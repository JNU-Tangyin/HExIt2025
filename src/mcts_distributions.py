#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCTS访问分布功能模块

- 处理MCTS访问计数和分布
- 纯函数式编程实现
- 模块化设计，高内聚低耦合
- 管道式数据流处理
"""
import numpy as np
from functools import partial

# ========== 常量定义 ==========
# 温度系数默认值
DEFAULT_TEMPERATURE = 1.0

# 最小概率以避免除零
EPSILON = 1e-8

# ========== 访问分布函数 ==========
def get_action_counts(mcts_node):
    """
    获取MCTS节点的动作访问计数
    
    Args:
        mcts_node: MCTS树节点
        
    Returns:
        动作访问计数数组
    """
    return np.array(mcts_node.num_edge_traversals)

def normalize_counts(counts, epsilon=EPSILON):
    """
    将访问计数归一化为概率分布
    
    Args:
        counts: 动作访问计数数组
        epsilon: 小数值，避免除零错误
        
    Returns:
        归一化后的概率分布
    """
    total = np.sum(counts)
    if total < epsilon:
        # 如果总访问次数接近零，返回均匀分布
        return np.ones_like(counts, dtype=float) / len(counts)
    return counts / total

def get_visit_distribution(mcts_node):
    """
    获取MCTS节点的访问概率分布
    
    Args:
        mcts_node: MCTS树节点
        
    Returns:
        动作访问概率分布
    """
    counts = get_action_counts(mcts_node)
    return normalize_counts(counts)

def apply_temperature(distribution, temperature=DEFAULT_TEMPERATURE, epsilon=EPSILON):
    """
    对分布应用温度调整
    
    Args:
        distribution: 原始概率分布(列表或NumPy数组)
        temperature: 温度参数，值越小分布越尖锐，值越大分布越平滑
        epsilon: 小数值，避免数值问题
        
    Returns:
        调整后的概率分布(始终返回NumPy数组)
    """
    # 确保分布是NumPy数组
    dist_array = np.array(distribution, dtype=float)
    
    if abs(temperature - 1.0) < epsilon:
        # 如果温度接近1，无需调整
        return dist_array
    
    # 应用温度
    scaled_probs = np.power(dist_array + epsilon, 1.0 / temperature)
    
    # 重新归一化
    return scaled_probs / np.sum(scaled_probs)

def filter_illegal_actions(distribution, legal_actions_mask):
    """
    过滤非法动作，只保留合法动作的概率
    
    Args:
        distribution: 原始概率分布
        legal_actions_mask: 布尔掩码，标记哪些动作是合法的
        
    Returns:
        过滤后的概率分布
    """
    # 将非法动作概率设为0
    filtered_dist = distribution * legal_actions_mask
    
    # 重新归一化
    total = np.sum(filtered_dist)
    if total > 0:
        return filtered_dist / total
    
    # 如果没有合法动作有概率，返回均匀分布
    return legal_actions_mask.astype(float) / np.sum(legal_actions_mask)

def sample_from_distribution(distribution):
    """
    从分布中采样一个动作
    
    Args:
        distribution: 概率分布
        
    Returns:
        采样的动作索引
    """
    return np.random.choice(len(distribution), p=distribution)

def get_best_action(distribution):
    """
    获取分布中概率最高的动作
    
    Args:
        distribution: 概率分布
        
    Returns:
        概率最高的动作索引
    """
    return np.argmax(distribution)

# ========== 分布处理管道 ==========
def process_mcts_distribution(mcts_node, temperature=DEFAULT_TEMPERATURE, 
                             use_sampling=True, legal_actions_mask=None):
    """
    处理MCTS分布的完整管道
    
    Args:
        mcts_node: MCTS树节点
        temperature: 温度参数
        use_sampling: 是否从分布中采样（否则选择最佳动作）
        legal_actions_mask: 合法动作掩码，如为None则假设所有动作合法
        
    Returns:
        选择的动作和处理后的概率分布
    """
    # 获取原始访问分布
    distribution = get_visit_distribution(mcts_node)
    
    # 应用温度
    if temperature != 1.0:
        distribution = apply_temperature(distribution, temperature)
    
    # 过滤非法动作
    if legal_actions_mask is not None:
        distribution = filter_illegal_actions(distribution, legal_actions_mask)
    
    # 选择动作
    if use_sampling:
        action = sample_from_distribution(distribution)
    else:
        action = get_best_action(distribution)
    
    return action, distribution

def compute_entropy(distribution, epsilon=EPSILON):
    """
    计算分布的熵
    
    Args:
        distribution: 概率分布
        epsilon: 小数值，避免log(0)
        
    Returns:
        分布的熵值
    """
    # 过滤掉零概率
    valid_probs = distribution[distribution > epsilon]
    
    # 如果没有有效概率，返回0
    if len(valid_probs) == 0:
        return 0
    
    # 计算熵: -sum(p * log(p))
    entropy = -np.sum(valid_probs * np.log(valid_probs))
    
    # 计算最大可能熵
    max_entropy = np.log(len(valid_probs))
    
    # 返回相对熵(0到1之间)
    return entropy / max_entropy if max_entropy > 0 else 0

def analyze_distribution(distribution):
    """
    分析分布的特性
    
    Args:
        distribution: 概率分布
        
    Returns:
        包含分布特性的字典
    """
    sorted_indices = np.argsort(distribution)[::-1]  # 降序
    top_actions = sorted_indices[:5]
    top_probs = distribution[top_actions]
    
    return {
        'entropy': compute_entropy(distribution),
        'max_prob': float(np.max(distribution)),
        'min_prob': float(np.min(distribution)),
        'std_dev': float(np.std(distribution)),
        'top_actions': top_actions.tolist(),
        'top_probs': top_probs.tolist(),
        'effective_actions': int(np.sum(distribution > 0.01))  # 有效动作数(>1%)
    }

# ========== 测试辅助函数 ==========
def create_legal_actions_mask(state):
    """
    为给定状态创建合法动作掩码
    
    Args:
        state: 游戏状态
        
    Returns:
        布尔掩码数组
    """
    num_actions = state.numActions()
    mask = np.zeros(num_actions, dtype=bool)
    
    for action in range(num_actions):
        mask[action] = state.isLegalAction(action)
    
    return mask
