#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价值预测功能模块

- 独立的价值预测数据处理
- 全函数式编程实现
- 无外部依赖，仅依赖numpy和标准库
- 符合扁平化设计原则
"""
import numpy as np
from functools import partial

# ========== 常量定义 ==========
WIN_VALUE = 1.0    # 获胜价值
LOSS_VALUE = 0.0   # 失败价值
DRAW_VALUE = 0.5   # 平局价值（Hex游戏中不存在，为完整性保留）

# ========== 纯函数实现 ==========
def create_value_labels(game_results, current_players):
    """
    基于游戏结果和当前玩家创建价值标签
    
    Args:
        game_results: 包含游戏结果的列表/数组，元素为胜者标识(-1或1)
        current_players: 每个状态对应的当前玩家
        
    Returns:
        形状为(n,1)的价值标签数组
    """
    # 使用函数式map生成结果
    value_labels = map(
        lambda pair: WIN_VALUE if pair[0] == pair[1] else LOSS_VALUE,
        zip(game_results, current_players)
    )
    
    # 转换为numpy数组并调整形状为(n,1)
    return np.array(list(value_labels)).reshape(-1, 1)

def create_value_batch(states, value_labels, batch_indices):
    """
    创建价值训练批次
    
    Args:
        states: 状态向量数组
        value_labels: 价值标签数组
        batch_indices: 批次索引
        
    Returns:
        批次状态和批次价值标签
    """
    return states[batch_indices], value_labels[batch_indices]

def compute_value_accuracy(predictions, labels, threshold=0.5):
    """
    计算价值预测准确率
    
    Args:
        predictions: 模型预测的价值，范围[0,1]
        labels: 真实价值标签，通常为0或1
        threshold: 分类阈值，默认0.5
        
    Returns:
        准确率百分比
    """
    binary_preds = predictions > threshold
    binary_labels = labels > threshold
    return np.mean(binary_preds == binary_labels) * 100.0

def create_training_feed_dict(model, states, action_dists, value_labels, learning_rate):
    """
    创建训练数据的feed_dict
    
    Args:
        model: 神经网络模型
        states: 状态向量批次
        action_dists: 动作分布批次
        value_labels: 价值标签批次
        learning_rate: 学习率
        
    Returns:
        TensorFlow feed_dict
    """
    return {
        model.input_ph: states,
        model.policy_ph: action_dists,
        model.value_ph: value_labels,
        model.lr_ph: learning_rate
    }

def extract_training_metrics(session, model, feed_dict):
    """
    提取训练指标
    
    Args:
        session: TensorFlow会话
        model: 神经网络模型
        feed_dict: 训练数据feed_dict
        
    Returns:
        包含各种损失值的字典
    """
    total_loss, policy_loss, value_loss = session.run(
        [model.loss, model.policy_loss, model.value_loss],
        feed_dict
    )
    
    return {
        'total_loss': float(total_loss),
        'policy_loss': float(policy_loss),
        'value_loss': float(value_loss)
    }

# ========== 训练工具函数 ==========
def train_with_value_head(session, model, data_generator, 
                         batch_size=32, epochs=5, learning_rate=0.001):
    """
    使用价值头训练模型
    
    Args:
        session: TensorFlow会话
        model: 神经网络模型
        data_generator: 生成训练数据的函数,返回(states, action_dists, winners, current_players)
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        
    Returns:
        训练指标字典
    """
    # 获取训练数据
    states, action_dists, winners, players = data_generator()
    
    # 创建价值标签
    value_labels = create_value_labels(winners, players)
    
    # 训练指标记录
    metrics_history = []
    
    # 训练循环
    for epoch in range(epochs):
        # 打乱数据
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        # 批次训练
        batch_metrics = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < 2:  # 避免过小的批次
                continue
                
            # 创建feed_dict
            feed_dict = create_training_feed_dict(
                model, states[batch_idx], action_dists[batch_idx], 
                value_labels[batch_idx], learning_rate
            )
            
            # 执行训练操作
            _, metrics = session.run(
                [model.train_op, model.loss], 
                feed_dict
            )
            
            # 记录批次指标
            batch_metrics.append(
                extract_training_metrics(session, model, feed_dict)
            )
        
        # 计算本轮平均指标
        if batch_metrics:
            epoch_metrics = {
                key: np.mean([m[key] for m in batch_metrics]) 
                for key in batch_metrics[0]
            }
            epoch_metrics['epoch'] = epoch + 1
            metrics_history.append(epoch_metrics)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={epoch_metrics['total_loss']:.4f} "
                  f"(Policy={epoch_metrics['policy_loss']:.4f}, "
                  f"Value={epoch_metrics['value_loss']:.4f})")
    
    # 返回训练历史
    return metrics_history

# ========== 评估函数 ==========
def evaluate_value_predictions(session, model, states, true_values):
    """
    评估价值预测性能
    
    Args:
        session: TensorFlow会话
        model: 神经网络模型
        states: 状态向量
        true_values: 真实价值标签
        
    Returns:
        包含评估指标的字典
    """
    # 获取预测值
    predictions = session.run(model.value_pred, {model.state_ph: states})
    
    # 计算各种指标
    accuracy = compute_value_accuracy(predictions, true_values)
    mae = np.mean(np.abs(predictions - true_values))
    mse = np.mean(np.square(predictions - true_values))
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'mse': mse,
        'min_pred': float(np.min(predictions)),
        'max_pred': float(np.max(predictions)),
        'avg_pred': float(np.mean(predictions))
    }
