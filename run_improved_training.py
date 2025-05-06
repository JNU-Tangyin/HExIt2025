#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hex AI 增强版训练启动脚本

这是一个安全的入口点，提供完整路径配置和导入修复，
确保改进的专家迭代训练能够正确运行。

功能:
- 解决模块导入和路径问题
- 启动改进版专家迭代训练
- 显示训练进度和结果汇总
"""
import os
import sys
import importlib
import numpy as np
import tensorflow.compat.v1 as tf

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入src包来设置模块别名和解决循环引用
import src

# TensorFlow设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()

# 训练参数 (可以根据需要调整)
BOARD_SIZE = 5
NUM_EXPERT_ITER = 2         # Expert Iteration轮数
NUM_SELFPLAY_GAMES = 10     # 每轮自我对弈局数
NUM_EVAL_GAMES = 10         # 每轮评估局数
TRAIN_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

CB_VALUES = [1.0]           # 探索常数
TEMPERATURE_VALUES = [0.2]  # 温度参数
NUM_SIMULATIONS_LIST = [100] # MCTS模拟次数

# 优化参数
VALUE_WEIGHT = 0.5          # 价值损失权重

def run_training():
    """执行优化版专家迭代训练"""
    # 导入训练模块 (导入放在函数内部避免循环引用)
    from improved_expert_iteration import main
    
    # 设置全局参数
    import src.config as config
    config.BOARD_SIZE = BOARD_SIZE
    
    # 确保模型目录存在
    model_dir = os.path.join(project_root, "models", "hex", str(BOARD_SIZE), "improved")
    os.makedirs(model_dir, exist_ok=True)
    
    # 打印训练配置
    print("\n" + "="*50)
    print("启动增强版Hex专家迭代训练")
    print("="*50)
    print(f"棋盘大小: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"专家迭代轮数: {NUM_EXPERT_ITER}")
    print(f"每轮自我对弈局数: {NUM_SELFPLAY_GAMES}")
    print(f"每轮评估局数: {NUM_EVAL_GAMES}")
    print(f"训练轮数: {TRAIN_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"探索常数: {CB_VALUES}")
    print(f"温度参数: {TEMPERATURE_VALUES}")
    print(f"MCTS模拟次数: {NUM_SIMULATIONS_LIST}")
    print(f"价值损失权重: {VALUE_WEIGHT}")
    print(f"模型保存目录: {model_dir}")
    print("="*50 + "\n")
    
    # 设置HexNN的价值权重
    from src.hex_nn import HexNN
    HexNN.VALUE_WEIGHT = VALUE_WEIGHT
    
    # 执行训练
    try:
        main()
        print("\n" + "="*50)
        print("训练成功完成!")
        print("="*50)
    except Exception as e:
        print("\n" + "="*50)
        print(f"训练过程中出错: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_training()
