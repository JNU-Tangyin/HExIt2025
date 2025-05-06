#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块定位与导入工具

- 集中定位所有内部模块
- 提供统一导入接口
- 支持相对路径和绝对路径导入
- 符合扁平化设计原则

用法:
    from src.module_locator import get_module
    HexState = get_module('hex').HexState
"""
import os
import sys
import importlib
import functools

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 模块名称映射表
MODULE_MAPPING = {
    # 核心模块
    'config': 'src.config',
    'hex': 'src.hex',
    'mcts': 'src.mcts',
    'hex_nn': 'src.hex_nn',
    'exit_nn': 'src.exit_nn',
    'agents': 'src.agents',
    'mcts_thread_manager': 'src.mcts_thread_manager',
    
    # 工具模块
    'utils': 'src.utils',
    
    # 测试模块
    'test': 'test'
}

@functools.lru_cache(maxsize=64)
def patch_module_imports(module):
    """
    修复模块中的导入引用问题
    
    Args:
        module: 要修复的模块对象
    """
    # 在模块的全局命名空间中添加引用到其他模块
    for name, mapped_name in MODULE_MAPPING.items():
        if name != module.__name__.split('.')[-1]:  # 避免自引用
            try:
                imported_module = importlib.import_module(mapped_name)
                setattr(module, name, imported_module)
            except ImportError:
                pass  # 忽略失败的导入

def get_module(module_name):
    """
    获取指定名称的模块
    
    Args:
        module_name: 模块名称（简短名称，如'hex'）
        
    Returns:
        导入的模块
        
    Raises:
        ImportError: 如果模块不存在或无法导入
    """
    # 处理不同形式的模块名称
    if module_name in MODULE_MAPPING:
        full_module_name = MODULE_MAPPING[module_name]
    else:
        full_module_name = module_name
    
    # 尝试多种导入方式
    errors = []
    
    # 方式1: 直接完整模块路径
    try:
        module = importlib.import_module(full_module_name)
        # 成功导入后修复模块内的引用
        patch_module_imports(module)
        return module
    except ImportError as e:
        errors.append(f"\n尝试方式1 '{full_module_name}': {str(e)}")
    
    # 方式2: 尝试src.前缀
    if not full_module_name.startswith('src.'):
        try:
            module = importlib.import_module(f'src.{full_module_name}')
            patch_module_imports(module)
            return module
        except ImportError as e:
            errors.append(f"\n尝试方式2 'src.{full_module_name}': {str(e)}")
    
    # 方式3: 如果是相对导入，尝试绝对导入
    if '.' in full_module_name:
        base_name = full_module_name.split('.')[-1]
        try:
            module = importlib.import_module(base_name)
            patch_module_imports(module)
            return module
        except ImportError as e:
            errors.append(f"\n尝试方式3 '{base_name}': {str(e)}")
    
    # 方式4: 重支持误导入格式
    special_cases = {
        'mcts': 'src.mcts',
        'hex': 'src.hex',
        'config': 'src.config',
        'exit_nn': 'src.exit_nn'
    }
    
    if module_name in special_cases:
        try:
            module = importlib.import_module(special_cases[module_name])
            patch_module_imports(module)
            return module
        except ImportError as e:
            errors.append(f"\n尝试方式4 '{special_cases[module_name]}': {str(e)}")
    
    # 所有方式都失败了，打印详细错误并抛出异常
    error_msg = f"\n无法导入模块 '{module_name}':"
    for err in errors:
        error_msg += err
    print(error_msg)
    
    # 尝试一个最后的方法 - 直接monkeypatch全局sys.modules
    try:
        import sys
        if module_name == 'mcts':
            from src import mcts
            sys.modules['mcts'] = mcts
            return mcts
        elif module_name == 'hex':
            from src import hex
            sys.modules['hex'] = hex
            return hex
    except ImportError as e:
        print(f"\n最后的monkeypatch尝试也失败: {str(e)}")
    
    # 最终抛出异常
    raise ImportError(error_msg)

def import_symbols(module_name, *symbols):
    """
    从指定模块导入多个符号
    
    Args:
        module_name: 模块名称
        *symbols: 要导入的符号名称列表
        
    Returns:
        包含所有导入符号的元组
    """
    module = get_module(module_name)
    return tuple(getattr(module, symbol) for symbol in symbols)

# 示例：如何导入多个符号
# HexState, HexNN = import_symbols('hex', 'HexState', 'HexNN')
