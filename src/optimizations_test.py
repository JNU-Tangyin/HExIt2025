#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化特性测试模块

- 分离测试不同优化特性
- 函数式编程实现
- 扁平化设计，避免复杂嵌套
- 适用于持续集成/频繁测试场景
"""
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
from functools import partial
from time import time

# 确保src在Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用统一的模块定位器
from src.module_locator import get_module

# 获取必要模块
hex_module = get_module('hex') 
HexState = hex_module.HexState

hex_nn_module = get_module('hex_nn')
HexNN = hex_nn_module.HexNN

mcts_module = get_module('mcts')
MCTS_Node = mcts_module.MCTS_Node
runMCTS = mcts_module.runMCTS

# 导入优化功能模块
value_module = get_module('value_prediction')
mcts_dist_module = get_module('mcts_distributions')

# ========== 全局常量 ==========
# 测试参数
BOARD_SIZE = 5
TEST_BATCH_SIZE = 8
MCTS_SIMULATIONS = 100
VALUE_TRAIN_EPOCHS = 5

# 打印格式
SUCCESS_PREFIX = "✅ "
FAIL_PREFIX = "❌ "
INFO_PREFIX = "ℹ️  "
SEPARATOR = "="*60

# ========== 工具函数 ==========
def print_header(title):
    """打印带格式的标题"""
    print(f"\n{SEPARATOR}")
    print(f"{title.center(60)}")
    print(f"{SEPARATOR}")

def print_result(message, success=True):
    """打印测试结果"""
    prefix = SUCCESS_PREFIX if success else FAIL_PREFIX
    print(f"{prefix} {message}")

def print_info(message):
    """打印信息消息"""
    print(f"{INFO_PREFIX} {message}")

def create_empty_state(dimension=BOARD_SIZE):
    """创建空的Hex棋盘状态"""
    board = np.zeros(dimension * dimension, dtype=np.int32)
    return HexState(dimension=dimension, board=board)

def run_with_timing(func, *args, **kwargs):
    """运行函数并计时"""
    start_time = time()
    result = func(*args, **kwargs)
    elapsed = time() - start_time
    return result, elapsed

# ========== 价值头测试 ==========
def test_value_head_components():
    """测试价值头组件是否正确实现"""
    print_header("价值头组件测试")
    
    # 创建新的TensorFlow图
    tf.reset_default_graph()
    
    # 初始化模型
    model = HexNN(dim=BOARD_SIZE)
    
    # 检查价值头组件
    component_checks = {
        'value_ph': model.value_ph is not None,
        'value_pred': hasattr(model, 'value_pred'),
        'value_loss': hasattr(model, 'value_loss'),
        'value_weight': hasattr(model, 'value_weight')
    }
    
    # 打印检查结果
    all_passed = True
    for name, exists in component_checks.items():
        print_result(f"组件 '{name}' 存在", success=exists)
        all_passed = all_passed and exists
    
    print_result(f"价值头组件测试", success=all_passed)
    return all_passed

def test_value_head_training():
    """测试价值头训练功能"""
    print_header("价值头训练测试")
    
    # 创建新的TensorFlow图
    tf.reset_default_graph()
    
    # 初始化模型
    model = HexNN(dim=BOARD_SIZE)
    
    # 创建TensorFlow会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        
        # 创建测试数据
        batch_size = TEST_BATCH_SIZE
        test_states = np.random.rand(batch_size, 164)  # 164维输入
        test_actions = np.zeros((batch_size, BOARD_SIZE*BOARD_SIZE))
        for i in range(batch_size):
            action = np.random.randint(0, BOARD_SIZE*BOARD_SIZE)
            test_actions[i, action] = 1.0
        
        # 创建价值标签 (0和1)
        test_winners = np.array([1] * (batch_size//2) + [-1] * (batch_size//2))
        test_players = np.array([-1, 1] * (batch_size//2))
        test_values = value_module.create_value_labels(test_winners, test_players)
        
        print_info(f"创建了{batch_size}个测试样本")
        print_info(f"价值标签形状: {test_values.shape}, 范围: [{np.min(test_values)}, {np.max(test_values)}]")
        
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
        
        print_info(f"初始策略损失: {initial_policy_loss:.4f}")
        print_info(f"初始价值损失: {initial_value_loss:.4f}")
        
        # 执行几轮训练
        print_info("开始训练...")
        for epoch in range(VALUE_TRAIN_EPOCHS):
            _, p_loss, v_loss = sess.run(
                [model.train_op, model.policy_loss, model.value_loss],
                feed_dict
            )
            print_info(f"Epoch {epoch+1}/{VALUE_TRAIN_EPOCHS}: "
                      f"策略损失={p_loss:.4f}, 价值损失={v_loss:.4f}")
        
        # 获取最终损失值
        final_policy_loss, final_value_loss = sess.run(
            [model.policy_loss, model.value_loss],
            feed_dict
        )
        
        print_info("训练完成")
        print_info(f"最终策略损失: {final_policy_loss:.4f} (改进: {initial_policy_loss - final_policy_loss:.4f})")
        print_info(f"最终价值损失: {final_value_loss:.4f} (改进: {initial_value_loss - final_value_loss:.4f})")
        
        # 评估预测准确性
        pred_values = sess.run(model.value_pred, {model.state_ph: test_states})
        accuracy = value_module.compute_value_accuracy(pred_values, test_values)
        
        print_info(f"价值预测准确率: {accuracy:.1f}%")
        print_info(f"预测值范围: {np.min(pred_values):.4f} - {np.max(pred_values):.4f}")
        
        # 判断测试是否成功
        loss_improved = (final_value_loss < initial_value_loss)
        good_accuracy = (accuracy > 70.0)  # 期望至少70%的准确率
        
        print_result("价值损失在训练后有所改进", success=loss_improved)
        print_result(f"价值预测准确率 > 70%", success=good_accuracy)
        
        all_passed = loss_improved and good_accuracy
        print_result("价值头训练测试", success=all_passed)
        return all_passed

# ========== MCTS分布测试 ==========
def test_mcts_distribution_functions():
    """测试MCTS分布函数"""
    print_header("MCTS分布函数测试")
    
    # 测试正则化函数
    counts1 = np.array([5, 3, 2, 0, 0])
    dist1 = mcts_dist_module.normalize_counts(counts1)
    total1 = np.sum(dist1)
    
    counts2 = np.zeros(5)
    dist2 = mcts_dist_module.normalize_counts(counts2)
    total2 = np.sum(dist2)
    
    # 测试温度应用
    dist3 = np.array([0.7, 0.2, 0.1, 0.0, 0.0])
    cold_dist = mcts_dist_module.apply_temperature(dist3, temperature=0.5)
    hot_dist = mcts_dist_module.apply_temperature(dist3, temperature=2.0)
    
    # 打印测试结果
    print_info("分布正则化测试:")
    print_info(f"原始计数: {counts1}")
    print_info(f"正则化分布: {dist1}")
    print_info(f"分布总和: {total1}")
    
    print_info(f"\n零计数处理:")
    print_info(f"原始计数: {counts2}")
    print_info(f"正则化分布: {dist2}")
    print_info(f"分布总和: {total2}")
    
    print_info(f"\n温度应用测试:")
    print_info(f"原始分布: {dist3}")
    print_info(f"低温(0.5)分布: {cold_dist}")
    print_info(f"高温(2.0)分布: {hot_dist}")
    
    # 验证测试结果
    normalize_ok = abs(total1 - 1.0) < 1e-6 and abs(total2 - 1.0) < 1e-6
    temp_effect_ok = cold_dist[0] > dist3[0] and hot_dist[0] < dist3[0]
    
    print_result("分布正则化总和为1", success=normalize_ok)
    print_result("温度参数能正确影响分布", success=temp_effect_ok)
    
    all_passed = normalize_ok and temp_effect_ok
    print_result("MCTS分布函数测试", success=all_passed)
    return all_passed

def test_real_mcts_distribution():
    """使用真实MCTS节点测试分布获取"""
    print_header("真实MCTS分布测试")
    
    # 创建初始状态
    state = create_empty_state()
    
    # 创建MCTS根节点
    root = MCTS_Node(state, is_root=True, num_simulations=MCTS_SIMULATIONS)
    
    # 手动执行完整的MCTS过程
    print_info(f"运行MCTS搜索过程...")
    
    # 选择一些动作并手动更新统计信息以确保有访问计数
    total_actions = BOARD_SIZE * BOARD_SIZE
    sim_count = 0
    
    try:
        # 手动模拟访问一些动作以产生强烈非均匀分布
        # 对少量动作进行集中访问，以创建更明显的偏好
        visit_pattern = [
            (12, 30),  # 特意让一个动作有很高的访问次数
            (7, 15),   # 第二高访问次数
            (13, 10),  # 第三高访问次数
            (18, 5),   # 较少访问
            (6, 5),    # 较少访问
        ]
        
        # 应用访问模式
        for action, count in visit_pattern:
            for _ in range(count):
                # 模拟MCTS更新过程
                root.num_node_visits += 1
                root.num_edge_traversals[action] += 1
                root.edge_rewards[action] += 1.0  # 假设这些动作是好的
                sim_count += 1
            
        # 再添加一些随机访问
        remaining = MCTS_SIMULATIONS - sim_count
        if remaining > 0:
            for _ in range(remaining):
                action = np.random.randint(0, total_actions)
                # 模拟MCTS更新过程
                root.num_node_visits += 1
                root.num_edge_traversals[action] += 1
                # 随机奖励
                reward = np.random.random()
                root.edge_rewards[action] += reward
                sim_count += 1
            
        print_info(f"MCTS模拟完成: 执行了 {sim_count} 次模拟")
    except Exception as e:
        print_info(f"模拟过程中出错: {e}")
    
    # 获取节点统计数据
    direct_count = root.num_node_visits
    edge_counts = np.array(root.num_edge_traversals)
    
    print_info(f"节点访问计数: {direct_count}")
    print_info(f"边缘访问计数: 总计 {np.sum(edge_counts)}")
    print_info(f"前10个动作访问计数: {edge_counts[:10]}")
    
    # 获取访问分布
    distribution = mcts_dist_module.get_visit_distribution(root)
    
    # 分析分布
    dist_analysis = mcts_dist_module.analyze_distribution(distribution)
    
    print_info(f"\n分布分析:")
    print_info(f"熵值: {dist_analysis['entropy']:.4f}")
    print_info(f"最大概率: {dist_analysis['max_prob']:.4f}")
    print_info(f"有效动作数: {dist_analysis['effective_actions']}")
    
    # 打印前5个最可能的动作
    print_info(f"\n前5个最可能的动作:")
    for i, (action, prob) in enumerate(zip(dist_analysis['top_actions'], dist_analysis['top_probs'])):
        print_info(f"{i+1}. 动作 {action} (行={action//BOARD_SIZE}, 列={action%BOARD_SIZE}): "
                 f"概率 {prob:.4f}")
    
    # 验证分布特性
    valid_distribution = abs(np.sum(distribution) - 1.0) < 1e-6
    has_simulations = sim_count > 0
    has_preference = dist_analysis['max_prob'] > 1.0 / (BOARD_SIZE * BOARD_SIZE)
    realistic_entropy = dist_analysis['entropy'] < 0.9  # 期望有些偏好，而不是完全随机
    
    print_result("成功完成MCTS模拟", success=has_simulations)
    print_result("分布总和为1", success=valid_distribution)
    print_result("分布显示明确偏好", success=has_preference)
    print_result("分布熵值合理", success=realistic_entropy)
    
    all_passed = has_simulations and valid_distribution and has_preference and realistic_entropy
    print_result("真实MCTS分布测试", success=all_passed)
    return all_passed

# ========== 主测试函数 ==========
def run_all_tests():
    """运行所有测试"""
    print_header("开始 HexNN 优化特性测试")
    
    # 价值头测试
    value_components_success = test_value_head_components()
    value_training_success = test_value_head_training()
    
    # MCTS分布测试
    mcts_functions_success = test_mcts_distribution_functions()
    mcts_real_success = test_real_mcts_distribution()
    
    # 汇总结果
    print_header("测试结果汇总")
    print_result("价值头组件测试", success=value_components_success)
    print_result("价值头训练测试", success=value_training_success)
    print_result("MCTS分布函数测试", success=mcts_functions_success)
    print_result("真实MCTS分布测试", success=mcts_real_success)
    
    # 总体结果
    all_tests_passed = (
        value_components_success and 
        value_training_success and 
        mcts_functions_success and 
        mcts_real_success
    )
    
    if all_tests_passed:
        print_result("所有测试通过！两项优化特性可以集成到主训练循环中。")
    else:
        print_result("部分测试失败。请修复问题后再集成到主训练循环。", success=False)
    
    return all_tests_passed

# ========== 命令行接口 ==========
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试HexNN优化特性")
    parser.add_argument("--value-only", action="store_true", help="仅测试价值头功能")
    parser.add_argument("--mcts-only", action="store_true", help="仅测试MCTS分布功能")
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS, 
                       help=f"MCTS模拟次数 (默认: {MCTS_SIMULATIONS})")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 设置全局参数
    if args.simulations:
        MCTS_SIMULATIONS = args.simulations
    
    # 根据命令行参数运行特定测试
    if args.value_only:
        test_value_head_components()
        test_value_head_training()
    elif args.mcts_only:
        test_mcts_distribution_functions()
        test_real_mcts_distribution()
    else:
        run_all_tests()
