import sys
import os
# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exit_nn import ExitNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from src.utils import *
from src.config import *

class HexNN(ExitNN):
    """
    Neural network implementation for the Hex game.
    This network uses convolutional and fully connected layers to predict optimal policies.
    """

    def __init__(self, dim=5, temperature=0.5):
        """
        Initializes the Hex neural network.
        
        Args:
            dim: Dimension of the Hex board (default: 5)
            temperature: Softmax temperature parameter to control distribution sharpness (default: 0.5)
                         Lower values make the distribution more peaked
        """
        self.dimension = dim
        self.num_actions = dim * dim
        self.temperature = temperature
        
        # Define layer sizes
        self.conv1_filters = 128  # 增加卷积层容量 (原来是64)
        self.conv2_filters = 128  # 增加卷积层容量 (原来是64)
        self.conv3_filters = 128  # 增加卷积层容量 (原来是64)
        self.fc1_nodes = 512  # 增加全连接层节点数 (原来是256)
        self.fc2_nodes = 256  # 增加全连接层节点数 (原来是128)
        
        # Build the network architecture
        self.buildNetwork()

    def buildGraph(self):
        """
        Build the neural network architecture and return important nodes.
        This method is called by ExitNN.train when training from scratch.
        """
        # Get all graph nodes from buildNetwork method
        self.buildNetwork()
        
        # Create a dictionary of important nodes to return
        nodes = {}
        
        # Get references to important nodes
        nodes['x'] = self.state_ph
        nodes['y'] = self.actions_ph
        nodes['output'] = self.policy_probs
        nodes['cost'] = self.loss
        nodes['turn_mask'] = tf.placeholder(tf.float32, [None, 2], name="turn_mask_placeholder")
        nodes['cost_vec'] = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.actions_ph, logits=self.logits)
        nodes['fc_output'] = self.logits
        
        return nodes
        
    def buildNetwork(self):
        """
        Build the neural network architecture for Hex game.
        """
        # 不再重置默认图，而是在外部管理图的作用域
        
        # Input placeholders
        # 自动适配 makeStateVector 输出长度（如164）
        self.input_dim = 164  # 与 HexState.makeStateVector 输出一致
        self.state_ph = tf.placeholder(tf.float32, [None, self.input_dim], name="state_placeholder")
        self.actions_ph = tf.placeholder(tf.float32, [None, self.num_actions], name="actions_placeholder")
        self.learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate_placeholder")
        # 兼容训练接口，暴露标准属性名
        self.input_ph = self.state_ph
        self.policy_ph = self.actions_ph
        self.value_ph = None  # 如有价值头可补充，这里暂设为None
        self.lr_ph = self.learning_rate_ph

        
        # 适配164维输入，前一半为棋盘一通道，后一半为棋盘另一通道，最后2位为turn mask
        # 解析输入
        channel_size = (self.input_dim - 2) // 2
        board1 = tf.slice(self.state_ph, [0, 0], [-1, channel_size])
        board2 = tf.slice(self.state_ph, [0, channel_size], [-1, channel_size])
        turn_info = tf.slice(self.state_ph, [0, 2 * channel_size], [-1, 2])
        # 还原成2通道棋盘
        board1_2d = tf.reshape(board1, [-1, self.dimension + 4, self.dimension + 4, 1])
        board2_2d = tf.reshape(board2, [-1, self.dimension + 4, self.dimension + 4, 1])
        board_2d = tf.concat([board1_2d, board2_2d], axis=3)  # shape: [batch, 9, 9, 2]
        # 后续卷积用 board_2d

        
        # 使用TensorFlow 1.x兼容的卷积层API
        # 创建卷积层变量作用域
        with tf.variable_scope('conv1'):
            # First convolutional layer (接受2通道输入)
            w_conv1 = tf.get_variable('weights', [3, 3, 2, self.conv1_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('bias', [self.conv1_filters],
                                     initializer=tf.constant_initializer(0.1))
            conv1 = tf.nn.conv2d(board_2d, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.bias_add(conv1, b_conv1)
            conv1 = tf.nn.relu(conv1)

        
        # 应用批量归一化
        conv1_mean, conv1_var = tf.nn.moments(conv1, [0, 1, 2])
        conv1_bn = tf.nn.batch_normalization(conv1, conv1_mean, conv1_var, None, None, 1e-5)
        
        with tf.variable_scope('conv2'):
            # Second convolutional layer
            w_conv2 = tf.get_variable('weights', [3, 3, self.conv1_filters, self.conv2_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('bias', [self.conv2_filters],
                                     initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.conv2d(conv1_bn, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, b_conv2)
            
            # 应用批量归一化
            conv2_mean, conv2_var = tf.nn.moments(conv2, [0, 1, 2])
            conv2_bn = tf.nn.batch_normalization(conv2, conv2_mean, conv2_var, None, None, 1e-5)
            
            # 添加残差连接 - 如果卷积层的通道数不同，需要使用投影
            if self.conv1_filters != self.conv2_filters:
                # 通道数不同时使用充1x1卷积进行调整
                shortcut = tf.layers.conv2d(conv1_bn, self.conv2_filters, kernel_size=1, strides=1, padding='SAME')
            else:
                # 通道数相同时直接使用输入
                shortcut = conv1_bn
                
            # 将卷积结果与残差连接相加
            conv2 = tf.nn.relu(conv2_bn + shortcut)
        
        with tf.variable_scope('conv3'):
            # Third convolutional layer
            w_conv3 = tf.get_variable('weights', [3, 3, self.conv2_filters, self.conv3_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('bias', [self.conv3_filters],
                                     initializer=tf.constant_initializer(0.1))
            conv3 = tf.nn.conv2d(conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, b_conv3)
            
            # 应用批量归一化
            conv3_mean, conv3_var = tf.nn.moments(conv3, [0, 1, 2])
            conv3_bn = tf.nn.batch_normalization(conv3, conv3_mean, conv3_var, None, None, 1e-5)
            
            # 添加第二个残差连接
            if self.conv2_filters != self.conv3_filters:
                # 通道数不同时使用1x1卷积进行调整
                shortcut = tf.layers.conv2d(conv2, self.conv3_filters, kernel_size=1, strides=1, padding='SAME')
            else:
                # 通道数相同时直接使用输入
                shortcut = conv2
                
            # 将卷积结果与残差连接相加
            conv3 = tf.nn.relu(conv3_bn + shortcut)
        
        # Flatten the output of the conv layers
        conv_flat = tf.reshape(conv3, [-1, (self.dimension + 4) * (self.dimension + 4) * self.conv3_filters])
        
        # Concatenate with the turn information
        conv_with_turn = tf.concat([conv_flat, turn_info], axis=1)
        
        # 使用TensorFlow 1.x兼容的全连接层API
        with tf.variable_scope('fc1'):
            # First fully connected layer
            input_size = conv_with_turn.get_shape().as_list()[1]
            w_fc1 = tf.get_variable('weights', [input_size, self.fc1_nodes],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_fc1 = tf.get_variable('bias', [self.fc1_nodes],
                                  initializer=tf.constant_initializer(0.1))
            fc1 = tf.matmul(conv_with_turn, w_fc1) + b_fc1
            fc1 = tf.nn.relu(fc1)
        
        # 应用批量归一化
        fc1_mean, fc1_var = tf.nn.moments(fc1, [0])
        fc1_bn = tf.nn.batch_normalization(fc1, fc1_mean, fc1_var, None, None, 1e-5)
        
        with tf.variable_scope('fc2'):
            # Second fully connected layer
            w_fc2 = tf.get_variable('weights', [self.fc1_nodes, self.fc2_nodes],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_fc2 = tf.get_variable('bias', [self.fc2_nodes],
                                  initializer=tf.constant_initializer(0.1))
            fc2 = tf.matmul(fc1_bn, w_fc2) + b_fc2
            fc2 = tf.nn.relu(fc2)
        
        # 应用批量归一化
        fc2_mean, fc2_var = tf.nn.moments(fc2, [0])
        fc2_bn = tf.nn.batch_normalization(fc2, fc2_mean, fc2_var, None, None, 1e-5)
        
        # 策略头 - 预测动作概率分布
        with tf.variable_scope('policy_head'):
            # 策略输出层
            w_policy = tf.get_variable('weights', [self.fc2_nodes, self.num_actions],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_policy = tf.get_variable('bias', [self.num_actions],
                                  initializer=tf.constant_initializer(0.1))
            self.logits = tf.matmul(fc2_bn, w_policy) + b_policy
            # 低温度会使分布更加尖锐，高温度会使分布更加平滑
            self.policy_probs = tf.nn.softmax(self.logits / self.temperature, name="policy_probs")
        
        # 价值头 - 预测胜率
        with tf.variable_scope('value_head'):
            # 从fc2_bn分支出价值网络
            w_value_fc = tf.get_variable('weights_fc', [self.fc2_nodes, 128],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_value_fc = tf.get_variable('bias_fc', [128],
                                  initializer=tf.constant_initializer(0.1))
            value_fc = tf.nn.relu(tf.matmul(fc2_bn, w_value_fc) + b_value_fc)
            
            # 价值输出层 (预测范围为[0,1]的胜率)
            w_value = tf.get_variable('weights_out', [128, 1],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_value = tf.get_variable('bias_out', [1],
                              initializer=tf.constant_initializer(0.1))
            self.value_logits = tf.matmul(value_fc, w_value) + b_value
            self.value_pred = tf.nn.sigmoid(self.value_logits, name="value_prediction")
        
        # 初始化value_ph接收胜率标签
        self.value_ph = tf.placeholder(tf.float32, [None, 1], name="value_placeholder")
        
        # 策略损失 (交叉熵)
        self.policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.actions_ph,
                logits=self.logits
            )
        )
        
        # 价值损失 (均方误差)
        self.value_loss = tf.reduce_mean(tf.square(self.value_ph - self.value_pred))
        
        # 总损失 = 策略损失 + 价值损失（加权）
        self.value_weight = 0.5  # 价值损失权重系数
        self.loss = self.policy_loss + self.value_weight * self.value_loss
        
        # 添加全局步数计数器
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # 添加学习率衰减
        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_ph,
            self.global_step,
            decay_steps=1000,  # 每1000步衰减一次
            decay_rate=0.96,   # 每次衰减为原来的96%
            staircase=True
        )
        
        # Define optimizer with decayed learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # 添加梯度裁剪，防止梯度爆炸
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # 限制梯度范数最大为5
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        
        # Define accuracy metric
        correct_predictions = tf.equal(
            tf.argmax(self.policy_probs, axis=1),
            tf.argmax(self.actions_ph, axis=1)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        
        # Define initializer
        self.init = tf.global_variables_initializer()
        
        # Setup summary operations for TensorBoard visualization
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()

    def extractFeaturesFromState(self, state_vec):
        """
        Extracts features from the state vector for the Hex game.
        Processes the input state vector to match the expected shape for the neural network.
        
        Args:
            state_vec: State vector representing a Hex board state
            
        Returns:
            Processed features suitable for the neural network
        """
        try:
            # 只允许输入164维特征，直接返回，无需任何映射。
            if isinstance(state_vec, np.ndarray) and len(state_vec.shape) > 1:
                state_vec = state_vec.flatten()
            if len(state_vec) == 164:
                return state_vec
            else:
                raise ValueError(f"HexNN expects 164-dim state vector, got {len(state_vec)}")
        except Exception as e:
            print(f"Error in extractFeaturesFromState: {e}")
            return np.zeros(164)


    def predict(self, state_vec, sess, model_ckpt_dir=None, nodes=None, sample=False):
        """
        Makes a prediction for the given state vector.
        
        Args:
            state_vec: State vector representing a Hex board state
            sess: TensorFlow session
            model_ckpt_dir: Directory to restore model from (optional)
            nodes: Dictionary of TensorFlow nodes (optional)
            sample: Whether to sample from the policy or take argmax (default: False)
            
        Returns:
            Policy prediction (action probabilities) or sampled/max action index
        """
        try:
            # 如果需要，从checkpoint恢复模型
            if model_ckpt_dir is not None and nodes is None:
                saver = tf.train.Saver()
                try:
                    saver.restore(sess, model_ckpt_dir)
                    print(f"Restored model from {model_ckpt_dir}")
                except Exception as e:
                    print(f"Error restoring model from {model_ckpt_dir}: {e}")
            
            # 获取输出节点
            output_node = None
            if nodes is not None and 'output' in nodes:
                output_node = nodes['output']
            else:
                output_node = self.policy_probs
            
            # 处理输入特征
            features = self.extractFeaturesFromState(state_vec)
            features_reshaped = np.reshape(features, (1, -1))
            
            feed_dict = {
                self.state_ph: features_reshaped
            }
            
            # 运行预测
            policy = sess.run(output_node, feed_dict=feed_dict)
            policy = policy[0]  # 获取第一个(唯一的)预测结果
            
            # 根据sample参数返回采样结果或直接返回概率分布
            if sample:
                # 从策略分布中采样一个动作
                action_idx = np.random.choice(len(policy), p=policy)
                return action_idx
            else:
                return policy
        except Exception as e:
            print(f"Error in predict: {e}")
            # 出错时返回均匀分布
            policy = np.ones(self.num_actions) / self.num_actions
            if sample:
                return np.random.choice(self.num_actions)
            return policy

    def getGraphNodes(self, sess):
        """
        获取图中的关键节点，用于预测。
        在模型初始化或恢复后调用。
        
        Args:
            sess: TensorFlow会话
            
        Returns:
            包含关键节点的字典
        """
        try:
            # 创建一个节点字典
            nodes = {}
            
            # 获取图中已经存在的关键节点
            graph = tf.get_default_graph()
            
            # 尝试获取输入占位符
            try:
                nodes['x'] = self.state_ph
                print("Successfully retrieved state_ph node")
            except Exception as e:
                try:
                    nodes['x'] = graph.get_tensor_by_name('state_placeholder:0')
                    print("Found x node by name in graph")
                except:
                    print("Could not find x node")
            
            # 尝试获取输出节点
            try:
                nodes['output'] = self.policy_probs
                print("Successfully retrieved policy_probs node")
            except Exception as e:
                try:
                    nodes['output'] = graph.get_tensor_by_name('policy_probs:0')
                    print("Found output node by name in graph")
                except:
                    print("Could not find output node")
            
            # 检查是否成功获取了关键节点
            if 'x' in nodes and 'output' in nodes:
                print("Successfully retrieved all necessary nodes for prediction")
            else:
                print("Warning: Missing some necessary nodes for prediction")
            
            return nodes
            
        except Exception as e:
            print(f"Error in getGraphNodes: {e}")
            return {}
        
    def train(self, data_dir, model_ckpt_dir=None, new_model_ckpt_dir=None, from_scratch=True, 
              dataset_size=None, split=0.8, begin_from=0, num_epochs=20, batch_size=32,
              learning_rate=0.001, log_every=8, save_every=1):
        """
        Trains the neural network on Hex game data.
        
        Args:
            data_dir: Directory containing training data
            model_ckpt_dir: Directory from which to restore a model (if not from_scratch)
            new_model_ckpt_dir: Directory to save the newly trained model
            from_scratch: Whether to train from scratch or continue training an existing model
            dataset_size: Size of the dataset to use for training
            split: Training/validation split ratio
            begin_from: Index to begin reading data from
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        """
        writeLog(f"Training HexNN model with dimension {self.dimension}")
        
        # Initialize or load session
        with tf.Session() as sess:
            # Initialize variables or restore model
            if from_scratch:
                writeLog("Initializing model from scratch")
                sess.run(self.init)
            else:
                if model_ckpt_dir is None:
                    raise ValueError("model_ckpt_dir must be provided if not training from scratch")
                
                writeLog(f"Restoring model from {model_ckpt_dir}")
                self.restoreModel(sess, model_ckpt_dir)
            
            # Create summary writer for TensorBoard
            if new_model_ckpt_dir is not None:
                summary_writer = tf.summary.FileWriter(new_model_ckpt_dir, sess.graph)
            
            # Load and preprocess data
            states, actions = self.loadData(data_dir, dataset_size, begin_from)
            
            # Shuffle and split data
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            split_idx = int(len(indices) * split)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            train_states = [states[i] for i in train_indices]
            train_actions = [actions[i] for i in train_indices]
            
            val_states = [states[i] for i in val_indices]
            val_actions = [actions[i] for i in val_indices]
            
            writeLog(f"Training set size: {len(train_states)}")
            writeLog(f"Validation set size: {len(val_states)}")
            
            # Training loop
            num_batches = len(train_states) // batch_size
            
            for epoch in range(num_epochs):
                # Shuffle training data
                epoch_indices = np.arange(len(train_states))
                np.random.shuffle(epoch_indices)
                
                # Track metrics
                epoch_loss = 0
                epoch_acc = 0
                
                for batch in range(num_batches):
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, len(train_states))
                    
                    batch_indices = epoch_indices[start_idx:end_idx]
                    
                    batch_states = np.array([train_states[i] for i in batch_indices])
                    batch_actions = np.array([train_actions[i] for i in batch_indices])
                    
                    # Train on batch
                    feed_dict = {
                        self.state_ph: batch_states,
                        self.actions_ph: batch_actions,
                        self.learning_rate_ph: learning_rate
                    }
                    
                    _, batch_loss, batch_acc, summary = sess.run(
                        [self.train_op, self.loss, self.accuracy, self.merged_summary],
                        feed_dict=feed_dict
                    )
                    
                    epoch_loss += batch_loss
                    epoch_acc += batch_acc
                    
                    # 根据log_every记录日志
                    if batch % log_every == 0:
                        writeLog(f"Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}")
                    
                    # Write summary
                    if new_model_ckpt_dir is not None and batch % log_every == 0:
                        global_step = epoch * num_batches + batch
                        summary_writer.add_summary(summary, global_step)
                
                # Calculate epoch metrics
                epoch_loss /= num_batches
                epoch_acc /= num_batches
                
                # Validation
                feed_dict = {
                    self.state_ph: np.array(val_states),
                    self.actions_ph: np.array(val_actions)
                }
                
                val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                
                writeLog(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save model checkpoint based on save_every parameter
                if new_model_ckpt_dir is not None and (epoch + 1) % save_every == 0:
                    self.saveCheckpoint(sess, new_model_ckpt_dir)
                    writeLog(f"Saved checkpoint at epoch {epoch+1} to {new_model_ckpt_dir}")
            
            # Save final model
            if new_model_ckpt_dir is not None:
                self.saveCheckpoint(sess, new_model_ckpt_dir)
                writeLog(f"Model saved to {new_model_ckpt_dir}")

    def saveCheckpoint(self, sess, model_ckpt_dir):
        """
        保存模型到检查点目录
        
        Args:
            sess: TensorFlow会话
            model_ckpt_dir: 保存模型的目录
        """
        try:
            os.makedirs(model_ckpt_dir, exist_ok=True)
            saver = tf.train.Saver()
            save_path = saver.save(sess, os.path.join(model_ckpt_dir, "model"))
            print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def restoreModel(self, sess, model_ckpt_dir):
        """
        从检查点恢复模型
        
        Args:
            sess: TensorFlow会话
            model_ckpt_dir: 模型检查点目录
        """
        try:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_ckpt_dir, "model"))
            print(f"Model restored from {model_ckpt_dir}")
        except Exception as e:
            print(f"Error restoring model: {e}")
            # 初始化所有变量，以防止未初始化错误
            sess.run(tf.global_variables_initializer())
            print("Initialized variables due to restore failure")
            
    def loadData(self, data_dir, dataset_size=None, begin_from=0):
        """
        Loads Hex game data from the specified directory.
        
        Args:
            data_dir: Directory containing data files
            dataset_size: Number of examples to load (None for all)
            begin_from: Index to begin loading from
            
        Returns:
            Tuple of (states, actions)
        """
        import os
        import numpy as np
        import csv
        
        # 确保输入路径以/结尾
        if data_dir[-1] != '/':
            data_dir += '/'
        
        # 定义状态和动作分布的目录路径
        states_dir = os.path.join(data_dir, 'states')
        actions_dir = os.path.join(data_dir, 'action_distributions')
        
        # 检查目录是否存在
        if not os.path.exists(states_dir) or not os.path.exists(actions_dir):
            raise FileNotFoundError(f"训练数据目录不存在: {states_dir} 或 {actions_dir}")
        
        # 获取目录中的文件列表
        state_files = sorted([f for f in os.listdir(states_dir) if f.endswith('.csv')])
        action_files = sorted([f for f in os.listdir(actions_dir) if f.endswith('.csv')])
        
        # 检查文件数量是否匹配
        if len(state_files) != len(action_files):
            raise ValueError(f"状态文件数量 ({len(state_files)}) 与动作分布文件数量 ({len(action_files)}) 不匹配")
        
        print(f"Found {len(state_files)} data files")
        
        # 决定加载哪些文件
        start_file_idx = begin_from
        file_count = 0
        
        # 加载数据
        states = []
        actions = []
        total_rows = 0
        
        # 从文件加载数据
        for i in range(start_file_idx, len(state_files)):
            state_file = os.path.join(states_dir, state_files[i])
            action_file = os.path.join(actions_dir, action_files[i])
            
            if os.path.exists(state_file) and os.path.exists(action_file):
                print(f"Loading file pair: {state_file} and {action_file}")
                
                # 加载状态
                with open(state_file, 'r') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if dataset_size is not None and total_rows >= dataset_size:
                            break
                        # 将CSV中的字符串转换为浮点数
                        state_vector = [float(x) for x in row]
                        
                        # 确保状态向量长度为27，与模型期望的输入形状匹配
                        expected_length = 27
                        if len(state_vector) < expected_length:
                            # 如果状态向量过短，用零填充
                            padded_vector = np.zeros(expected_length)
                            padded_vector[:len(state_vector)] = state_vector
                            state_vector = padded_vector
                        elif len(state_vector) > expected_length:
                            # 如果状态向量过长，只取前27个元素
                            state_vector = state_vector[:expected_length]
                            
                        states.append(state_vector)
                        total_rows += 1
                
                # 重置计数器用于动作文件
                row_idx = 0
                
                # 加载对应的动作分布
                with open(action_file, 'r') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if dataset_size is not None and len(actions) >= dataset_size:
                            break
                        # 将CSV中的字符串转换为浮点数
                        action_vector = [float(x) for x in row]
                        actions.append(action_vector)
                
                file_count += 1
                
                # 如果达到了所需的数据集大小，退出循环
                if dataset_size is not None and total_rows >= dataset_size:
                    break
        
        # 确保状态和动作数量匹配
        if len(states) != len(actions):
            raise ValueError(f"加载的状态数量 ({len(states)}) 与动作数量 ({len(actions)}) 不匹配")
        
        print(f"Successfully loaded {len(states)} examples from {file_count} files")
        
        return np.array(states), np.array(actions)
