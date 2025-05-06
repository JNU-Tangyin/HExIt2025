#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hexit模块包 - 统一入口点

解决模块间循环引用和路径问题
基于扁平化设计，最小化依赖耦合
"""
import os
import sys
import importlib

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 重要模块提前注册到sys.modules以解决循环引用
# 遵循扁平化设计，避免复杂的依赖图
core_modules = ['mcts', 'hex', 'config', 'hex_nn', 'agents']

# 为这些模块创建别名映射，解决直接导入问题
for module_name in core_modules:
    try:
        # 使用源模块路径
        full_name = f"src.{module_name}"
        module = importlib.import_module(full_name)
        
        # 创建顶级别名，让from mcts import xxx也能工作
        sys.modules[module_name] = module
    except ImportError:
        pass  # 忽略加载失败，后续导入时会再次尝试

# 导出所有核心模块供外部直接使用
__all__ = ['hex', 'mcts', 'hex_nn', 'agents', 'config']

