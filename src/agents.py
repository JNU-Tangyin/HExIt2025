import numpy as np
import os
import tensorflow.compat.v1 as tf

# 优先尝试相对导入，然后尝试src前缀导入
try:
    from config import NN_MODEL_PATH as DEFAULT_NN_MODEL_PATH
    from config import DEFAULT_NUM_SIMULATIONS
    from exit_nn import ExitNN
    from hex import HexState
except ModuleNotFoundError:
    from src.config import NN_MODEL_PATH as DEFAULT_NN_MODEL_PATH
    from src.config import DEFAULT_NUM_SIMULATIONS
    from src.exit_nn import ExitNN
    from src.hex import HexState
try:
    from mcts import MCTS_Node, StateVector, ActionDistribution, runAllSimulations, runMCTS
    from mcts_thread_manager import parallel_mcts
except ModuleNotFoundError:
    from src.mcts import MCTS_Node, StateVector, ActionDistribution, runAllSimulations, runMCTS
    from src.mcts_thread_manager import parallel_mcts

class GameAgent(object):
    """
    The superclass of all Agent classes.
    All inheriting classes must implement a getAction method.
    Given a state, this method returns the index of the chosen action.
    """
    def __init__(self, game):
        """
        The "game" argument is a string that specifies what game this agent is playing.
        """
        self.game = game

    def getAction(self, state):
        """
        Returns a legal action for the given state.
        """
        return -1



class NNAgent(GameAgent):
    """ 
    The NNAgent class chooses actions by having a neural network predict optimal actions for the given state.
    """

    def __init__(self, nn, model_path, sample=True):
        """
        The nn argument specifies the NN that will be choosing actions.
        The model_path argument allows us to restore a model from a checkpoint file.
        The sample argument tells the NN whether to sample the best action.
        If false, it just chooses the action with the highest score.
        """
        self.model_path = model_path
        self.sample = sample
        self.nn = nn
        self.nodes = None # will be set by restoreModel

    def restoreModel(self, sess):
        """
        Restores a model from a checkpoint, if the checkpoint exists.
        Otherwise, initializes a new model.
        """
        try:
            # 首先尝试初始化所有变量，确保TensorFlow图中所有变量都被初始化
            sess.run(tf.global_variables_initializer())
            
            # 然后尝试恢复模型，这将覆盖初始化的变量
            saver = tf.train.Saver()
            try:
                saver.restore(sess, self.model_path)
                print("Successfully restored model from", self.model_path)
            except Exception as e:
                print("Could not restore model:", e)
                print("Using initialized model")
                # 保存新模型，以便下次可以恢复它
                import os
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                save_path = saver.save(sess, self.model_path)
                print("Saved new model to", save_path)
            
            # 保存网络节点以供predict使用
            self.nodes = self.nn.getGraphNodes(sess)
            
        except Exception as e:
            print("Error in restore model:", e)


    def predictBatch(self, states, sess):
        """
        Given a batch of state vectors.
        Makes predictions for them.
        Returns an array of chosen actions.
        """
        try:
            # 确保输入状态向量格式正确
            if len(states.shape) == 1:
                # 单个状态向量，需要添加批次维度
                states = np.expand_dims(states, axis=0)
            
            # 获取批次大小
            num_states = states.shape[0]
            
            # 如果是单个状态
            if num_states == 1:
                # 直接预测单个状态
                chosen_actions = self.nn.predict(states[0], sess, self.model_path, nodes=self.nodes, sample=self.sample)
                
                # 确保返回的是一个数组，即使是单个动作
                if isinstance(chosen_actions, (int, np.int32, np.int64)):
                    return [chosen_actions]
                return chosen_actions
            else:
                # 对每个状态进行预测，然后合并结果
                all_actions = []
                for i in range(num_states):
                    action = self.nn.predict(states[i], sess, self.model_path, nodes=self.nodes, sample=self.sample)
                    if isinstance(action, (int, np.int32, np.int64)):
                        all_actions.append(action)
                    else:
                        # 如果是概率分布，找到最可能的动作
                        all_actions.append(np.argmax(action))
                return all_actions
        except Exception as e:
            print(f"Error in predictBatch: {e}")
            # 出错时返回随机动作
            dim = self.nn.dimension
            return [np.random.randint(0, dim*dim) for _ in range(max(1, num_states))]


    def getAction(self, state, sess):
        """
        Returns a legal action for the given state. The action is chosen by the NN.
        If the NN keeps repeatedly suggesting illegal actions, eventually error.
        """

        # create np array, feed to predict method
        state_vector = state.makeStateVector()
        chosen_action = self.nn.predict(state_vector, sess, self.model_path, nodes=self.nodes, sample=self.sample)
        if not state.isLegalAction(chosen_action):
            chosen_action = state.randomAction()

        return chosen_action


class RandomAgent(GameAgent):
    """
    The RandomAgent class chooses actions uniformly at random from the set of legal actions.
    """

    def __init__(self, game):
        self.game = game

    def getAction(self, state):
        """
        Returns a legal action for the given state, chosen uniformly at random from the set of legal actions.
        """
        return state.randomAction()




class UserAgent(GameAgent):
    """
    The UserAgent class chooses actions by prompting a user to select one.
    """

    def __init__(self, game):
        self.game = game


    def getAction(self, state):
        """
        Returns the action chosen by the user for the given state.
        If the user chooses an illegal action, prompt again.
        If the user repeatedly chooses illegal actions, error.
        """

        chosen_action = -1
        legal_action = False
        t = 0
        while not legal_action and t < 10:
            chosen_action = input("Enter a legal action number: ")
            legal_action = state.isLegalAction(chosen_action)
            if not legal_action:
                print("Action", chosen_action, "is illegal.  Choose again")

        assert chosen_action != -1, "User entered too many illegal actions in UserAgent"
        return chosen_action



class MCTSAgent(GameAgent):
    """
    The MCTSAgent class uses Monte Carlo Tree Search to select actions.
    This is the Python implementation of MCTS agent (vs. the C++ implementation).
    """
    def __init__(self, game, nn_agent=None, num_simulations=DEFAULT_NUM_SIMULATIONS, 
                 num_threads=1, use_parallel=False, use_nn=False, sample=True):
        """
        Args:
            game: 游戏类型
            nn_agent: 神经网络代理，用于策略预测
            num_simulations: 每个状态进行的模拟次数
            num_threads: 并行线程数
            use_parallel: 是否使用并行MCTS
            use_nn: 是否使用神经网络辅助MCTS
            sample: 是否从动作分布中采样动作
        """
        super(MCTSAgent, self).__init__(game)
        self.nn_agent = nn_agent
        self.num_simulations = num_simulations
        self.num_threads = num_threads
        self.use_parallel = use_parallel
        self.use_nn = use_nn
        self.sample = sample
    
    def getAction(self, state, sess=None, return_distribution=False):
        """
        使用MCTS为给定状态选择最佳动作
        
        Args:
            state: 游戏状态
            sess: TensorFlow会话（当使用神经网络时需要）
            return_distribution: 是否同时返回访问分布（默认False）
        
        Returns:
            如果return_distribution=False，返回选择的动作索引
            如果return_distribution=True，返回(选择的动作索引, 访问分布)
        """
        # 创建MCTS节点
        requires_nn = self.use_nn and self.nn_agent is not None
        node = MCTS_Node(state, is_root=True, num_simulations=self.num_simulations, 
                       sample_actions=self.sample, requires_nn=requires_nn)
        
        # 如果使用神经网络，首先预测动作
        if requires_nn and sess is not None:
            state_vector = state.makeStateVector()
            # 将状态向量转换为NumPy数组
            state_vector_array = np.array([state_vector])
            predictions = self.nn_agent.predictBatch(state_vector_array, sess)[0]
            
            # 如果预测结果是单个整数，创建一个空分布并只给预测的动作赋值
            if isinstance(predictions, (int, np.int32, np.int64)):
                num_actions = state.numActions()
                action_probs = np.zeros(num_actions)
                action_probs[predictions] = 1.0
                action_dist = ActionDistribution(action_dist=action_probs)
            else:
                action_dist = ActionDistribution(action_dist=predictions)
                
            node.setNNActionDistribution(action_dist)
        
        # 在单线程模式下运行MCTS
        if not self.use_parallel:
            # 运行所有模拟
            runAllSimulations(node, self.num_simulations)
        # 运行MCTS模拟
        if self.use_parallel:
            parallel_mcts(node, self.num_threads, self.nn_agent)
        else:
            for _ in range(self.num_simulations):
                runMCTS(node)
        
        # 获取访问分布
        visit_distribution = node.get_visit_distribution()
        
        # 获取动作
        if self.sample:
            # 根据访问计数采样动作
            legal_actions = [i for i in range(len(visit_distribution)) if state.isLegalAction(i)]
            if not legal_actions:
                # 如果没有合法动作，返回随机动作
                action = state.randomAction()
            else:   
                # 重新标准化分布，只关注合法动作
                legal_dist = [visit_distribution[i] if i in legal_actions else 0.0 for i in range(len(visit_distribution))]
                total = sum(legal_dist)
                if total > 0:
                    legal_dist = [d / total for d in legal_dist]
                    action = np.random.choice(len(legal_dist), p=legal_dist)
                else:
                    # 如果分布不合法，返回随机动作
                    action = np.random.choice(legal_actions)
        else:
            # 选择访问次数最多的动作
            action_counts = node.getActionCounts()
            
            # 只考虑合法动作
            for i in range(len(action_counts)):
                if not state.isLegalAction(i):
                    action_counts[i] = -1
            
            # 返回访问次数最多的合法动作
            action = np.argmax(action_counts)
            
        # 根据参数决定返回动作还是同时返回分布
        if return_distribution:
            return action, visit_distribution
        else:
            return action


def declareWinner(winner, reward):
    """
    Prints the winner and reward in a neat format.
    """
    print("*" * 50)
    print("*" + " " * 48 + "*")
    print("*" + " " * 15 + "The winner is: " + winner + " " * 15 + "*")
    print("*" + " " * 48 + "*")
    print("*" * 50)
    print("\nThe reward is: " + str(reward))
    if winner == 1:
        print("Player 1 wins with a reward of", reward)
    elif winner == -1:
        print("Player 2 wins with a reward of", reward)
    else:
        print("Draw")
