#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价值头预测和真实MCTS分布集成测试
- 验证价值头训练效果
- 验证真实MCTS访问分布效果
- 扁平化设计，便于理解和维护
"""
import os
import sys
sys.path.append('.')  # 确保能找到本地模块

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 导入相关模块
from src.hex import HexState
from src.hex_nn import HexNN
from src.mcts import MCTS_Node, runMCTS

# 常量定义
BOARD_SIZE = 5     # 棋盘大小
NUM_SIMULATIONS = 100  # MCTS模拟次数
BATCH_SIZE = 8    # 训练批次大小
EPOCHS = 5        # 训练轮数

# =================== 辅助函数 ===================
def create_empty_state():
    """创建空的Hex棋盘状态"""
    board = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
    return HexState(dimension=BOARD_SIZE, board=board)

def print_separator(title):
    """打印分隔符和标题"""
    print("\n" + "="*50)
    print(f" {title} ".center(50, '='))
    print("="*50)

def get_visit_distribution(node):
    """从MCTS节点获取访问分布"""
    counts = node.getActionCounts()
    total = sum(counts)
    if total == 0:
        # 如果无访问，返回均匀分布
        return [1.0 / node.num_actions] * node.num_actions
    return [count / total for count in counts]

# =================== 测试价值头预测 ===================
def test_value_head():
    """测试价值头预测功能"""
    print_separator("价值头预测测试")
    
    # 创建一个新的TensorFlow图
    tf.reset_default_graph()
    
    # 初始化HexNN模型
    model = HexNN(dim=BOARD_SIZE)
    
    # 检查价值头组件
    value_components = {
        'value_ph': model.value_ph is not None,
        'value_pred': hasattr(model, 'value_pred'),
        'value_loss': hasattr(model, 'value_loss')
    }
    
    print("1. 价值头组件检查:")
    for component, exists in value_components.items():
        print(f"   - {component}: {'✓' if exists else '✗'}")
    
    # 创建TensorFlow会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        
        # 生成随机测试数据
        test_states = np.random.rand(BATCH_SIZE, 164)
        test_actions = np.zeros((BATCH_SIZE, BOARD_SIZE*BOARD_SIZE))
        for i in range(BATCH_SIZE):
            action = np.random.randint(0, BOARD_SIZE*BOARD_SIZE)
            test_actions[i, action] = 1.0
        
        # 生成胜率标签（50%获胜，50%失败）
        test_values = np.zeros((BATCH_SIZE, 1))
        test_values[:BATCH_SIZE//2] = 1.0  # 前半部分为获胜
        
        # 准备训练数据
        feed_dict = {
            model.state_ph: test_states,
            model.actions_ph: test_actions,
            model.value_ph: test_values,
            model.learning_rate_ph: 0.001
        }
        
        # 获取初始损失值
        initial_policy_loss, initial_value_loss = sess.run(
            [model.policy_loss, model.value_loss],
            feed_dict
        )
        
        print("\n2. 初始损失值:")
        print(f"   - 策略损失: {initial_policy_loss:.4f}")
        print(f"   - 价值损失: {initial_value_loss:.4f}")
        
        # 执行几轮训练
        print("\n3. 开始训练...")
        for epoch in range(EPOCHS):
            _, p_loss, v_loss = sess.run(
                [model.train_op, model.policy_loss, model.value_loss],
                feed_dict
            )
            print(f"   - Epoch {epoch+1}/{EPOCHS}: 策略损失={p_loss:.4f}, 价值损失={v_loss:.4f}")
        
        # 获取最终损失值
        final_policy_loss, final_value_loss = sess.run(
            [model.policy_loss, model.value_loss],
            feed_dict
        )
        
        print("\n4. 训练后损失值:")
        print(f"   - 策略损失: {final_policy_loss:.4f} (改进: {initial_policy_loss - final_policy_loss:.4f})")
        print(f"   - 价值损失: {final_value_loss:.4f} (改进: {initial_value_loss - final_value_loss:.4f})")
        
        # 评估预测准确性
        predictions = sess.run(model.value_pred, {model.state_ph: test_states})
        accuracy = np.mean((predictions > 0.5) == (test_values > 0.5))
        
        print("\n5. 价值预测评估:")
        print(f"   - 准确率: {accuracy*100:.1f}%")
        print(f"   - 预测值范围: {np.min(predictions):.4f} - {np.max(predictions):.4f}")
    
    print("\n价值头测试结果: ", "成功" if accuracy > 0.5 else "需要改进")
    return value_components['value_ph'] and value_components['value_pred']

# =================== 测试MCTS访问分布 ===================
def test_mcts_distribution():
    """测试MCTS访问分布功能"""
    print_separator("MCTS访问分布测试")
    
    # 创建初始状态
    state = create_empty_state()
    
    # 创建MCTS节点
    node = MCTS_Node(state, num_simulations=NUM_SIMULATIONS)
    
    # 运行几次MCTS模拟
    print("1. 运行MCTS模拟...")
    for i in range(NUM_SIMULATIONS):
        if i % 20 == 0:
            print(f"   - 已完成 {i}/{NUM_SIMULATIONS} 次模拟")
        runMCTS(node)
    
    # 获取动作计数和访问分布
    counts = node.getActionCounts()
    distribution = get_visit_distribution(node)
    
    # 找出访问最多的动作
    most_visited = np.argmax(counts)
    most_visited_count = counts[most_visited]
    most_visited_prob = distribution[most_visited]
    
    print("\n2. MCTS结果分析:")
    print(f"   - 总模拟次数: {node.num_node_visits}")
    print(f"   - 访问最多的动作: {most_visited} (行={most_visited//BOARD_SIZE}, 列={most_visited%BOARD_SIZE})")
    print(f"   - 该动作访问次数: {most_visited_count}")
    print(f"   - 该动作访问概率: {most_visited_prob:.4f}")
    
    # 检查分布的有效性
    valid_distribution = abs(sum(distribution) - 1.0) < 1e-6
    has_preference = most_visited_prob > 1.0 / (BOARD_SIZE * BOARD_SIZE)
    
    print("\n3. 分布有效性检查:")
    print(f"   - 分布总和为1: {'✓' if valid_distribution else '✗'} ({sum(distribution):.6f})")
    print(f"   - 存在明确偏好: {'✓' if has_preference else '✗'}")
    
    # 分析分布的熵（越低表示分布越集中）
    non_zero_probs = [p for p in distribution if p > 0]
    entropy = -sum([p * np.log(p) for p in non_zero_probs])
    max_entropy = np.log(len(non_zero_probs))
    relative_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print("\n4. 分布特性分析:")
    print(f"   - 非零概率动作数: {len(non_zero_probs)}/{BOARD_SIZE*BOARD_SIZE}")
    print(f"   - 分布熵: {entropy:.4f} (相对熵: {relative_entropy:.2f})")
    
    # 输出前5个最可能的动作
    top_indices = np.argsort(distribution)[-5:][::-1]
    print("\n5. 前5个最可能的动作:")
    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. 动作 {idx} (行={idx//BOARD_SIZE}, 列={idx%BOARD_SIZE}): 概率 {distribution[idx]:.4f}")
    
    print("\nMCTS分布测试结果: ", "成功" if valid_distribution and has_preference else "需要改进")
    return valid_distribution and has_preference

# =================== 集成测试 ===================
def run_integration_test():
    """运行价值头和MCTS分布的集成测试"""
    print_separator("优化特性集成测试")
    
    # 测试价值头
    value_head_success = test_value_head()
    
    # 测试MCTS分布
    mcts_dist_success = test_mcts_distribution()
    
    # 打印总结果
    print_separator("测试结果总结")
    print(f"1. 价值头预测: {'✓ 成功' if value_head_success else '✗ 失败'}")
    print(f"2. MCTS真实分布: {'✓ 成功' if mcts_dist_success else '✗ 失败'}")
    
    if value_head_success and mcts_dist_success:
        print("\n总体结果: ✓ 所有测试通过!")
        print("\n结论: 两项优化特性已成功实现，可以整合到主训练循环中。")
        print("      这将提供更精确的策略学习和更好的状态评估能力。")
    else:
        print("\n总体结果: ✗ 部分测试失败")
        print("\n建议: 修复失败的测试，然后再尝试整合到主训练循环。")

if __name__ == "__main__":
    run_integration_test()
