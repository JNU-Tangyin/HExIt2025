#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import threading
import queue
import time
import tensorflow.compat.v1 as tf
import logging

# 确保使用TensorFlow 1.x兼容模式
tf.disable_v2_behavior()

# 导入自定义模块
from mcts import MCTS_Node, ActionDistribution, StateVector, runMCTS, runAllSimulations
from hex import HexState

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCTSThreadManager:
    """
    MCTS线程管理器负责创建和管理多个MCTS模拟线程
    """
    def __init__(self, num_threads=4, nn_model=None):
        """
        初始化MCTS线程管理器
        
        Args:
            num_threads: 线程数量
            nn_model: 神经网络模型实例，用于动作预测
        """
        self.num_threads = num_threads
        self.nn_model = nn_model
        self.nn_queue = queue.Queue()
        self.nn_results = {}
        self.threads = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.nodes_lock = threading.Lock()
        self.nodes = []
        self.is_running = False
        
        # 如果提供了模型，创建预测线程
        if self.nn_model is not None:
            self.nn_thread = threading.Thread(target=self._nn_worker)
            self.nn_thread.daemon = True
    
    def add_node(self, node):
        """
        添加一个节点到MCTS处理队列
        
        Args:
            node: 要添加的MCTS_Node实例
        """
        with self.nodes_lock:
            self.nodes.append(node)
    
    def add_nodes(self, nodes):
        """
        添加多个节点到MCTS处理队列
        
        Args:
            nodes: 要添加的MCTS_Node实例列表
        """
        with self.nodes_lock:
            self.nodes.extend(nodes)
    
    def start(self):
        """
        启动MCTS线程和神经网络预测线程
        """
        if self.is_running:
            logger.warning("MCTS线程管理器已经在运行")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # 如果有神经网络模型，启动预测线程
        if self.nn_model is not None:
            self.nn_thread.start()
        
        # 创建并启动MCTS工作线程
        self.threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=self._mcts_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        logger.info(f"MCTS线程管理器已启动，正在使用{self.num_threads}个线程进行模拟")
    
    def stop(self):
        """
        停止所有MCTS线程和神经网络预测线程
        """
        if not self.is_running:
            return
        
        logger.info("正在停止MCTS线程管理器...")
        
        self.stop_event.set()
        
        # 等待所有线程完成
        for thread in self.threads:
            thread.join()
        
        if self.nn_model is not None:
            self.nn_thread.join()
        
        self.is_running = False
        logger.info("MCTS线程管理器已停止")
    
    def wait_until_done(self):
        """
        等待所有节点完成模拟
        """
        while self.is_running:
            with self.nodes_lock:
                all_done = all(node.simulationsFinished() for node in self.nodes)
            
            if all_done:
                logger.info("所有节点模拟完成")
                return
            
            time.sleep(0.1)
    
    def get_results(self):
        """
        获取所有节点的最终动作分布
        
        Returns:
            节点列表和每个节点的动作分布
        """
        results = []
        with self.nodes_lock:
            for node in self.nodes:
                results.append((node, ActionDistribution(action_dist=node.getActionDistribution())))
        
        return results
    
    def _mcts_worker(self, thread_id):
        """
        MCTS工作线程函数
        
        Args:
            thread_id: 线程ID
        """
        logger.info(f"MCTS工作线程 {thread_id} 已启动")
        
        while not self.stop_event.is_set():
            node_to_process = None
            
            # 获取未完成的节点
            with self.nodes_lock:
                for node in self.nodes:
                    if not node.simulationsFinished():
                        if node.requiresNN():
                            if node.neverSubmittedToNN():
                                # 将节点提交到神经网络队列
                                self.nn_queue.put(node)
                                node.markSubmittedToNN()
                            elif node.awaitingNNResults():
                                # 检查是否有NN结果
                                node_id = id(node)
                                if node_id in self.nn_results:
                                    with self.lock:
                                        action_dist = self.nn_results.pop(node_id)
                                    node.setNNActionDistribution(action_dist)
                            else:
                                # 节点已准备好进行模拟
                                node_to_process = node
                                break
                        else:
                            # 节点不需要NN
                            node_to_process = node
                            break
            
            if node_to_process is not None:
                # 运行一次MCTS模拟
                runMCTS(node_to_process)
            else:
                # 没有找到可以处理的节点，休眠一段时间
                time.sleep(0.01)
        
        logger.info(f"MCTS工作线程 {thread_id} 已停止")
    
    def _nn_worker(self):
        """
        神经网络预测线程函数
        """
        logger.info("神经网络预测线程已启动")
        
        batch_size = 16  # 可调整的批处理大小
        
        while not self.stop_event.is_set() or not self.nn_queue.empty():
            batch = []
            batch_ids = []
            
            # 收集一批节点进行预测
            try:
                for _ in range(batch_size):
                    if self.nn_queue.empty():
                        break
                    
                    node = self.nn_queue.get(block=False)
                    batch.append(node.getStateVector())
                    batch_ids.append(id(node))
                    self.nn_queue.task_done()
            except queue.Empty:
                pass
            
            if not batch:
                time.sleep(0.01)
                continue
            
            # 使用神经网络进行预测
            try:
                predictions = self._predict_batch(batch)
                
                # 保存预测结果
                with self.lock:
                    for node_id, prediction in zip(batch_ids, predictions):
                        self.nn_results[node_id] = prediction
            except Exception as e:
                logger.error(f"神经网络预测出错: {e}")
        
        logger.info("神经网络预测线程已停止")
    
    def _predict_batch(self, state_vectors):
        """
        使用神经网络批量预测动作分布
        
        Args:
            state_vectors: 状态向量列表
        
        Returns:
            动作分布列表
        """
        # 转换为适合神经网络输入的格式
        states = [sv._sv for sv in state_vectors]
        
        # 使用模型预测
        predictions = self.nn_model.predict(states)
        
        # 转换为ActionDistribution
        return [ActionDistribution(action_dist=pred) for pred in predictions]

# 辅助函数，用于并行执行MCTS
def parallel_mcts(states, num_simulations=1000, num_threads=4, nn_model=None):
    """
    并行执行MCTS模拟
    
    Args:
        states: HexState对象列表或状态向量列表
        num_simulations: 每个状态的模拟次数
        num_threads: 并行线程数
        nn_model: 可选的神经网络模型
    
    Returns:
        每个状态的MCTS节点和动作分布
    """
    # 创建MCTS节点
    nodes = []
    
    for state in states:
        if isinstance(state, list) or isinstance(state, np.ndarray):
            # 如果是状态向量，转换为HexState
            state_vector = StateVector(state)
            hex_state = HexState.fromStateVector(state_vector)
            node = MCTS_Node(hex_state, num_simulations=num_simulations, requires_nn=(nn_model is not None))
        else:
            # 假设是HexState实例
            node = MCTS_Node(state, num_simulations=num_simulations, requires_nn=(nn_model is not None))
        
        nodes.append(node)
    
    # 创建线程管理器
    manager = MCTSThreadManager(num_threads=num_threads, nn_model=nn_model)
    
    # 添加节点并启动处理
    manager.add_nodes(nodes)
    manager.start()
    
    # 等待处理完成
    manager.wait_until_done()
    
    # 获取结果
    results = manager.get_results()
    
    # 停止线程管理器
    manager.stop()
    
    return results
