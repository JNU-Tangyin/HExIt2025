#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进版Expert Iteration循环 - 支持真实MCTS访问分布和价值预测

- 使用真实MCTS访问分布替代独热编码
- 增加价值头预测获胜概率
- 数据管道优化
"""
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import json
import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TensorFlow设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()

# 使用统一模块定位系统
from src.module_locator import get_module, import_symbols

# 导入核心模块
hex_module = get_module('hex')
HexState = hex_module.HexState

agents_module = get_module('agents')
RandomAgent, MCTSAgent, NNAgent = import_symbols('agents', 'RandomAgent', 'MCTSAgent', 'NNAgent')

hex_nn_module = get_module('hex_nn')
HexNN = hex_nn_module.HexNN

config = get_module('config')

# 导入优化模块
value_module = get_module('value_prediction')
mcts_dist_module = get_module('mcts_distributions')

# ========== 参数区 ==========
BOARD_SIZE = 5
NUM_EXPERT_ITER = 3         # Expert Iteration轮数
NUM_SELFPLAY_GAMES = 20     # 每轮自我对弈局数
NUM_EVAL_GAMES = 20         # 每轮评估局数
TRAIN_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

CB_VALUES = [1.0]           # 简化为单一值进行测试
TEMPERATURE_VALUES = [0.2]  # 简化为单一值进行测试
NUM_SIMULATIONS_LIST = [500]  # 简化为单一值进行测试

MODEL_DIR = "models/hex/5/improved/"
LOG_FILE = "improved_expert_iteration_log.json"

# ========== 工具函数 ==========
def create_empty_state(dimension=BOARD_SIZE):
    """创建空棋盘状态"""
    board = np.zeros(dimension * dimension, dtype=np.int32)
    return HexState(dimension=dimension, board=board)

def save_model(sess, saver, model_path):
    """保存模型到指定路径"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    saver.save(sess, model_path)

def log_results(log_data):
    """记录训练结果到日志文件"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_data)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# ========== 自我对弈 ==========
def run_self_play(nn_agent, c_b, temperature, num_simulations, num_games, sess):
    """
    生成自我对弈数据
    
    使用MCTS真实访问分布替代独热编码，并记录当前玩家信息用于价值标签
    
    Args:
        nn_agent: 神经网络代理
        c_b: 探索常数
        temperature: 温度参数
        num_simulations: 模拟次数
        num_games: 游戏局数
        sess: TensorFlow会话
        
    Returns:
        包含(状态向量,动作分布,胜者,当前玩家)四元组的数据列表
    """
    # 定义独立的游戏执行函数，符合函数式和扁平化设计
    def play_single_game(game_idx):
        # 初始化游戏状态
        state = create_empty_state()
        
        # 创建MCTS代理
        mcts_agent = MCTSAgent("hex", nn_agent=nn_agent, num_simulations=num_simulations, use_nn=True)
        
        # 暂存游戏中每一步的数据
        game_data = []
        states, action_dists, current_players = [], [], []
        
        # 设置探索常数
        old_cb = config.DEFAULT_C_B
        config.DEFAULT_C_B = c_b
        
        try:
            # 游戏主循环
            while not state.isTerminalState():
                # 记录当前玩家（用于价值标签）
                current_player = state.turn()
                current_players.append(current_player)
                
                # 获取动作和访问分布
                action, visit_dist = mcts_agent.getAction(state, sess, return_distribution=True)
                
                # 使用新开发的MCTS分布模块处理分布
                if temperature != 1.0:
                    # 应用温度参数
                    processed_dist = mcts_dist_module.apply_temperature(visit_dist, temperature)
                else:
                    processed_dist = visit_dist
                
                # 添加分布分析
                if game_idx == 0 and len(states) < 3:  # 仅对第一局前几步进行分析
                    dist_metrics = mcts_dist_module.analyze_distribution(processed_dist)
                    print(f"\n步骤 {len(states)+1} 分布分析:")
                    print(f"  熵值: {dist_metrics['entropy']:.4f}")
                    print(f"  最大概率: {dist_metrics['max_prob']:.4f}")
                    print(f"  有效动作数: {dist_metrics['effective_actions']}")
                    top_actions = dist_metrics['top_actions'][:3]
                    top_probs = dist_metrics['top_probs'][:3]
                    print(f"  前3个动作: {[(a, f'{p:.3f}') for a, p in zip(top_actions, top_probs)]}")
                
                # 保存数据
                action_dists.append(processed_dist)
                states.append(state.makeStateVector())
                
                # 执行动作，更新棋盘
                state = state.nextState(action)
            
            # 获取游戏结果
            winner = state.winner()
            
            # 组装返回数据
            for s, d, p in zip(states, action_dists, current_players):
                game_data.append((s, d, winner, p))
                
        finally:
            # 恢复原始探索常数
            config.DEFAULT_C_B = old_cb
        
        return game_data
    
    # 使用tqdm显示进度
    all_data = []
    for i in tqdm(range(num_games), desc="Self-play"):
        game_data = play_single_game(i)
        all_data.extend(game_data)
    
    print(f"\n自我对弈完成: 生成了 {len(all_data)} 个训练样本")
    return all_data

# ========== 训练神经网络 ==========
def train_nn(nn_agent, data, sess, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """
    训练神经网络
    
    同时训练策略头和价值头
    
    Args:
        nn_agent: 神经网络代理
        data: 训练数据，包含(状态,动作分布,胜者,当前玩家)四元组
        sess: TensorFlow会话
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        
    Returns:
        包含训练指标的字典
    """
    # 拆分数据
    states = np.array([x[0] for x in data])
    action_dists = np.array([x[1] for x in data])
    winners = np.array([x[2] for x in data])
    players = np.array([x[3] for x in data])
    
    # 使用价值预测模块创建价值标签
    value_labels = value_module.create_value_labels(winners, players)
    
    # 分析数据统计信息
    print(f"\n训练数据分析:")
    print(f"  样本数量: {len(states)}")
    print(f"  状态向量形状: {states.shape}")
    print(f"  动作分布形状: {action_dists.shape}")
    print(f"  价值标签形状: {value_labels.shape}")
    
    # 检查动作分布特性
    dist_entropies = [mcts_dist_module.compute_entropy(dist) for dist in action_dists[:100]]
    avg_entropy = np.mean(dist_entropies)
    print(f"  动作分布平均熵值: {avg_entropy:.4f}")
    
    # 初始化训练指标
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_total_loss = 0.0
    
    # 评估初始性能
    initial_metrics = value_module.extract_training_metrics(
        sess, nn_agent.nn, {
            nn_agent.nn.input_ph: states[:batch_size],
            nn_agent.nn.policy_ph: action_dists[:batch_size],
            nn_agent.nn.value_ph: value_labels[:batch_size],
            nn_agent.nn.lr_ph: learning_rate
        }
    )
    
    print(f"\n初始模型性能:")
    print(f"  策略损失: {initial_metrics['policy_loss']:.4f}")
    print(f"  价值损失: {initial_metrics['value_loss']:.4f}")
    print(f"  总损失: {initial_metrics['total_loss']:.4f}")
    
    # 使用函数式编程处理批次训练
    def train_epoch(epoch_num):
        # 函数式打乱数据和创建批次
        def create_batches():
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # 生成标准大小的批次
            batches = []
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) >= 2:  # 避免过小的批次
                    batches.append(batch_idx)
            return batches
        
        # 处理单个批次
        def process_batch(batch_idx):
            # 使用价值预测模块创建训练feed_dict
            feed_dict = value_module.create_training_feed_dict(
                nn_agent.nn,
                states[batch_idx], 
                action_dists[batch_idx],
                value_labels[batch_idx],
                learning_rate
            )
            
            # 执行训练操作
            _, metrics = sess.run(
                [nn_agent.nn.train_op, nn_agent.nn.loss],
                feed_dict
            )
            
            # 获取详细损失指标
            return value_module.extract_training_metrics(sess, nn_agent.nn, feed_dict)
        
        # 创建批次并执行训练
        batches = create_batches()
        batch_metrics = [process_batch(idx) for idx in batches]
        
        # 计算平均指标
        if batch_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in batch_metrics])
                for key in ['total_loss', 'policy_loss', 'value_loss']
            }
            
            # 打印这个时期的指标
            print(f"Epoch {epoch_num+1}/{epochs}: "
                  f"Loss={avg_metrics['total_loss']:.4f} "
                  f"(Policy={avg_metrics['policy_loss']:.4f}, "
                  f"Value={avg_metrics['value_loss']:.4f})")
            
            return avg_metrics
        
        return None
    
    # 应用map函数式风格运行所有时期
    print(f"\n开始训练 {epochs} 轮...")
    epoch_results = [train_epoch(e) for e in range(epochs)]
    epoch_results = [r for r in epoch_results if r is not None]  # 过滤无效结果
    
    # 计算总体平均指标
    if epoch_results:
        avg_metrics = {
            key: np.mean([r[key] for r in epoch_results])
            for key in ['total_loss', 'policy_loss', 'value_loss']
        }
        avg_policy_loss = avg_metrics['policy_loss']
        avg_value_loss = avg_metrics['value_loss']
        avg_total_loss = avg_metrics['total_loss']
    else:
        avg_policy_loss = avg_value_loss = avg_total_loss = 0.0
    
    # 返回训练指标
    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "total_loss": avg_total_loss
    }

# ========== 评估 ==========
def evaluate_against_random(nn_agent, c_b, temperature, num_simulations, num_games, sess):
    """
    评估代理对抗随机代理的表现
    
    Args:
        nn_agent: 神经网络代理
        c_b: 探索常数
        temperature: 温度参数
        num_simulations: 模拟次数
        num_games: 游戏局数
        sess: TensorFlow会话
        
    Returns:
        (胜率, 先手胜场数, 后手胜场数)
    """
    mcts_agent = MCTSAgent("hex", nn_agent=nn_agent, num_simulations=num_simulations, use_nn=True)
    random_agent = RandomAgent("hex")
    old_cb = config.DEFAULT_C_B
    config.DEFAULT_C_B = c_b
    
    # 先手评估
    mcts_first_wins = 0
    for _ in tqdm(range(num_games // 2), desc="Eval-First"):
        state = create_empty_state()
        while not state.isTerminalState():
            if state.turn() == 1:
                action = mcts_agent.getAction(state, sess)
            else:
                action = random_agent.getAction(state)
            state = state.nextState(action)
        if state.winner() == 1:
            mcts_first_wins += 1
    
    # 后手评估
    mcts_second_wins = 0
    for _ in tqdm(range(num_games - num_games // 2), desc="Eval-Second"):
        state = create_empty_state()
        while not state.isTerminalState():
            if state.turn() == -1:
                action = mcts_agent.getAction(state, sess)
            else:
                action = random_agent.getAction(state)
            state = state.nextState(action)
        if state.winner() == -1:
            mcts_second_wins += 1
    
    # 恢复探索常数
    config.DEFAULT_C_B = old_cb
    
    # 计算胜率
    win_rate = (mcts_first_wins + mcts_second_wins) / num_games
    return win_rate, mcts_first_wins, mcts_second_wins

# ========== 主循环 ==========
def main():
    """主函数，运行改进版Expert Iteration训练"""
    results = []
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 遍历所有参数组合
    for c_b in CB_VALUES:
        for temperature in TEMPERATURE_VALUES:
            for num_simulations in NUM_SIMULATIONS_LIST:
                print(f"\n==== 改进版ExIt: c_b={c_b}, temperature={temperature}, num_simulations={num_simulations} ====")
                
                # 重置TensorFlow图
                tf.reset_default_graph()
                
                # 创建会话
                with tf.Session() as sess:
                    # 初始化模型
                    hex_nn = HexNN(dim=BOARD_SIZE, temperature=temperature)
                    nn_agent = NNAgent(hex_nn, model_path=None)
                    sess.run(tf.global_variables_initializer())
                    
                    # Expert Iteration循环
                    for iteration in range(NUM_EXPERT_ITER):
                        print(f"\n[Iteration {iteration+1}] 自我对弈...")
                        
                        # 1. 自我对弈生成数据
                        data = run_self_play(
                            nn_agent,
                            c_b,
                            temperature,
                            num_simulations,
                            NUM_SELFPLAY_GAMES,
                            sess
                        )
                        
                        print(f"[Iteration {iteration+1}] 训练NN...")
                        
                        # 2. 训练神经网络
                        train_metrics = train_nn(nn_agent, data, sess)
                        
                        print(f"[Iteration {iteration+1}] 评估...")
                        
                        # 3. 评估表现
                        win_rate, first_wins, second_wins = evaluate_against_random(
                            nn_agent,
                            c_b,
                            temperature,
                            num_simulations,
                            NUM_EVAL_GAMES,
                            sess
                        )
                        
                        print(f"[Iteration {iteration+1}] 胜率: {win_rate*100:.1f}% "
                              f"(先手{first_wins}/{NUM_EVAL_GAMES//2}, "
                              f"后手{second_wins}/{NUM_EVAL_GAMES-NUM_EVAL_GAMES//2})")
                        
                        # 保存模型
                        model_path = os.path.join(
                            MODEL_DIR,
                            f"improved_cb{c_b}_temp{temperature}_sim{num_simulations}_iter{iteration+1}"
                        )
                        saver = tf.train.Saver()
                        save_model(sess, saver, model_path)
                        
                        # 记录本次迭代的数据
                        log_data = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'c_b': c_b,
                            'temperature': temperature,
                            'num_simulations': num_simulations,
                            'iteration': iteration+1,
                            'win_rate': win_rate,
                            'first_wins': first_wins,
                            'second_wins': second_wins,
                            'policy_loss': float(train_metrics['policy_loss']),
                            'value_loss': float(train_metrics['value_loss']),
                            'total_loss': float(train_metrics['total_loss']),
                            'model_path': model_path
                        }
                        
                        # 记录到日志文件
                        log_results(log_data)
                        results.append(log_data)
                        
    print("\n全部实验完成！结果已写入日志。")

if __name__ == '__main__':
    main()
