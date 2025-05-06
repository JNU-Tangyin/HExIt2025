#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import csv
import random
import math
from threading import Thread, Lock, Condition
import time
import tensorflow.compat.v1 as tf

# 确保使用TensorFlow 1.x兼容模式
tf.disable_v2_behavior()

# 导入必要的本地模块
# 优先尝试相对导入，然后尝试src前缀导入
try:
    from hex import HexState
except ModuleNotFoundError:
    from src.hex import HexState
# 配置导入也需要修复
try:
    from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_SAMPLE_ACTIONS, DEFAULT_REQUIRES_NN
    from config import DEFAULT_USE_RAVE, DEFAULT_C_B, DEFAULT_C_RAVE, DEFAULT_W_A, DEFAULT_MAX_DEPTH
    from config import DEFAULT_STATES_PER_FILE
except ModuleNotFoundError:
    from src.config import DEFAULT_NUM_SIMULATIONS, DEFAULT_SAMPLE_ACTIONS, DEFAULT_REQUIRES_NN
    from src.config import DEFAULT_USE_RAVE, DEFAULT_C_B, DEFAULT_C_RAVE, DEFAULT_W_A, DEFAULT_MAX_DEPTH
    from src.config import DEFAULT_STATES_PER_FILE

class StateVector:
    """
    状态向量类，作为神经网络输入的包装器
    """
    def __init__(self, vec=None):
        """创建状态向量，可选择复制给定向量的内容"""
        self._sv = [] if vec is None else list(vec)
    
    def at(self, k):
        """返回向量的第k个元素"""
        if k < 0 or k >= len(self._sv):
            raise IndexError(f"Index {k} out of range for StateVector of size {len(self._sv)}")
        return self._sv[k]
    
    def asCSVString(self):
        """返回CSV格式的字符串表示"""
        return ','.join(str(x) for x in self._sv)

class ActionDistribution:
    """
    动作分布类，处理神经网络预测的动作分布
    """
    def __init__(self, action_dist=None, num_actions=0, csv_line=None, delimiter=','):
        """
        初始化方式:
        1. 现有的动作分布向量
        2. 动作数量和CSV行
        """
        if action_dist is not None:
            self.action_dist = list(action_dist)
        elif csv_line is not None and num_actions > 0:
            self.action_dist = []
            elements = csv_line.strip().split(delimiter)
            for i in range(min(num_actions, len(elements))):
                self.action_dist.append(float(elements[i]))
            # 如有必要，用零填充
            while len(self.action_dist) < num_actions:
                self.action_dist.append(0.0)
        else:
            self.action_dist = []
    
    def at(self, k):
        """返回动作分布向量的第k个元素"""
        if k < 0 or k >= len(self.action_dist):
            raise IndexError(f"Index {k} out of range for ActionDistribution of size {len(self.action_dist)}")
        return self.action_dist[k]
    
    def asVector(self):
        """返回动作分布向量"""
        return self.action_dist
    
    def asCSVString(self):
        """返回CSV格式的字符串表示"""
        return ','.join(str(x) for x in self.action_dist)

class MCTS_Node:
    """
    蒙特卡洛树搜索节点实现
    """
    def __init__(self, state, is_root=True, num_simulations=DEFAULT_NUM_SIMULATIONS, 
                 sample_actions=DEFAULT_SAMPLE_ACTIONS, requires_nn=DEFAULT_REQUIRES_NN,
                 use_rave=DEFAULT_USE_RAVE, c_b=DEFAULT_C_B, c_rave=DEFAULT_C_RAVE, w_a=DEFAULT_W_A):
        """使用给定状态初始化MCTS节点"""
        self.state = state
        self.state_vector = StateVector(state.makeStateVector())
        self.is_root = is_root
        self.num_simulations_finished = 0
        self.total_num_simulations = num_simulations
        self.depth = 0
        self.submitted_to_nn = False
        self.received_nn_results = False
        self.nn_result = None
        self.num_actions = state.numActions()
        self.root = self
        self.parent = None
        self.children = [None] * self.num_actions
        self.child_index = -1 if is_root else 0
        
        # MCTS统计数据
        self.num_node_visits = 0
        self.num_edge_traversals = [0] * self.num_actions
        self.edge_rewards = [0.0] * self.num_actions
        
        # RAVE统计数据
        self.num_node_visits_rave = 0
        self.num_edge_traversals_rave = [0] * self.num_actions
        self.edge_rewards_rave = [0.0] * self.num_actions
        
        # 超参数
        self.c_b = c_b
        self.c_rave = c_rave
        self.w_a = w_a
        self.sample_actions = sample_actions
        self.requires_nn = requires_nn
        self.use_rave = use_rave
    
    def isTerminal(self):
        """检查节点状态是否是终止状态"""
        return self.state.isTerminalState()
    
    def isRoot(self):
        """检查节点是否是树的根节点"""
        return self.is_root
    
    def simulationsFinished(self):
        """检查根节点是否完成了所有模拟"""
        return self.num_simulations_finished >= self.total_num_simulations
    
    def markSimulationFinished(self):
        """标记完成了一次模拟"""
        self.num_simulations_finished += 1
    
    def usesRave(self):
        """检查节点是否使用RAVE"""
        return self.use_rave
    
    def requiresNN(self):
        """检查节点是否需要神经网络辅助"""
        return self.requires_nn
    
    def neverSubmittedToNN(self):
        """检查节点是否需要NN但从未提交其状态向量"""
        return self.requires_nn and not self.submitted_to_nn
    
    def markSubmittedToNN(self):
        """标记节点已提交其状态向量到NN"""
        self.submitted_to_nn = True
    
    def awaitingNNResults(self):
        """检查节点是否需要NN但尚未设置其nn_result"""
        return self.requires_nn and self.submitted_to_nn and not self.received_nn_results
    
    def markReceivedNNResults(self):
        """标记节点已设置其nn_result字段"""
        self.received_nn_results = True
    
    def getNumActions(self):
        """返回状态可用的动作数量"""
        return self.num_actions
    
    def getDepth(self):
        """返回节点在树中的深度"""
        return self.depth
    
    def setDepth(self, depth):
        """设置节点深度"""
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        self.depth = depth
        return self
    
    def getParent(self):
        """返回节点的父节点"""
        return self.parent
    
    def setParent(self, parent):
        """设置节点的父节点"""
        if parent is None:
            raise ValueError("Parent cannot be None")
        self.parent = parent
        return self
    
    def getChild(self, k):
        """返回第k个子节点"""
        if k < 0 or k >= self.num_actions:
            raise IndexError(f"Child index {k} out of range for node with {self.num_actions} actions")
        return self.children[k]
    
    def setChild(self, child, k):
        """设置第k个子节点"""
        if child is None:
            raise ValueError("Child cannot be None")
        if k < 0 or k >= self.num_actions:
            raise IndexError(f"Child index {k} out of range for node with {self.num_actions} actions")
        self.children[k] = child
        return self
    
    def getChildIndex(self):
        """返回子节点索引"""
        return self.child_index
    
    def setChildIndex(self, k):
        """设置子节点索引"""
        if k < 0 or k >= self.num_actions:
            raise IndexError(f"Child index {k} out of range for node with {self.num_actions} actions")
        self.child_index = k
        return self
    
    def getRoot(self):
        """返回节点树的根节点"""
        return self.root
    
    def setRoot(self, root):
        """设置节点的根指针"""
        if root is None:
            raise ValueError("Root cannot be None")
        self.root = root
        return self
    
    def makeChild(self, k):
        """
        返回第k个子节点，如果尚未创建则创建它
        """
        if self.isTerminal():
            return self
        
        if self.children[k] is not None:
            return self.children[k]
        
        # 执行动作k创建新状态
        next_state = self.state.nextState(k)
        
        # 为这个状态创建新节点
        child = MCTS_Node(next_state, is_root=False, 
                          num_simulations=self.total_num_simulations,
                          sample_actions=self.sample_actions,
                          requires_nn=self.requires_nn,
                          use_rave=self.use_rave,
                          c_b=self.c_b,
                          c_rave=self.c_rave,
                          w_a=self.w_a)
        
        # 建立这个节点和子节点之间的关系
        child.setParent(self)
        child.setRoot(self.root)
        child.setDepth(self.depth + 1)
        child.setChildIndex(k)
        
        # 更新此节点的子节点列表
        self.children[k] = child
        
        return child
    
    def getState(self):
        """返回节点的环境状态实例"""
        return self.state
    
    def getStateVector(self):
        """返回状态向量实例"""
        return self.state_vector
    
    def setNNActionDistribution(self, ad):
        """设置nn_result字段"""
        self.nn_result = ad
        self.markReceivedNNResults()
    
    def getActionCounts(self):
        """返回MCTS期间每个动作被选择的次数向量"""
        return self.num_edge_traversals
    
    def getActionDistribution(self):
        """
        返回MCTS期间每个动作被选择的比例向量
        """
        counts = self.getActionCounts()
        total = sum(counts)
        if total == 0:
            # 如果无访问，返回均匀分布
            return [1.0 / self.num_actions] * self.num_actions
        return [count / total for count in counts]
        
    def get_visit_distribution(self):
        """
        返回基于访问次数的动作概率分布
        与getActionDistribution相同，但使用更现代的命名约定
        """
        return self.getActionDistribution()
    
    def N(self, a=None):
        """
        如果a为None，返回MCTS期间此节点被访问的次数
        否则，返回MCTS期间从此节点执行动作a的次数
        """
        if a is None:
            return self.num_node_visits
        
        if a < 0 or a >= self.num_actions:
            raise IndexError(f"Action index {a} out of range for node with {self.num_actions} actions")
        
        return self.num_edge_traversals[a]
    
    def R(self, a):
        """返回从此节点执行动作a的所有模拟的总奖励"""
        if a < 0 or a >= self.num_actions:
            raise IndexError(f"Action index {a} out of range for node with {self.num_actions} actions")
        
        return self.edge_rewards[a]
    
    def NRave(self, a=None):
        """
        如果a为None，返回使用RAVE的MCTS期间此节点被访问的次数
        否则，返回使用RAVE的MCTS期间从此节点执行动作a的次数
        """
        if a is None:
            return self.num_node_visits_rave
        
        if a < 0 or a >= self.num_actions:
            raise IndexError(f"Action index {a} out of range for node with {self.num_actions} actions")
        
        return self.num_edge_traversals_rave[a]
    
    def RRave(self, a):
        """返回使用RAVE的从此节点执行动作a的所有模拟的总奖励"""
        if a < 0 or a >= self.num_actions:
            raise IndexError(f"Action index {a} out of range for node with {self.num_actions} actions")
        
        return self.edge_rewards_rave[a]
    
    def edgeTraversals(self):
        """返回边遍历向量"""
        return self.num_edge_traversals
    
    def edgeRewards(self):
        """返回边奖励向量"""
        return self.edge_rewards
    
    def edgeTraversalsRave(self):
        """返回RAVE的边遍历向量"""
        return self.num_edge_traversals_rave
    
    def edgeRewardsRave(self):
        """返回RAVE的边奖励向量"""
        return self.edge_rewards_rave
    
    def getMeanRewards(self):
        """返回每个动作的平均奖励向量"""
        mean_rewards = []
        for a in range(self.num_actions):
            if self.N(a) > 0:
                mean_rewards.append(self.R(a) / self.N(a))
            else:
                mean_rewards.append(0.0)
        return mean_rewards
    
    def computeUCT(self):
        """使用基本UCT公式计算每个动作的分数"""
        action_scores = []
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a):
                action_scores.append(float('-inf'))
                continue
                
            if self.N(a) == 0:
                action_scores.append(float('inf'))  # 未探索的动作具有无限价值
                continue
                
            # UCT公式: r(s,a)/n(s,a) + c_b * sqrt(log n(s) / n(s,a))
            exploit = self.R(a) / self.N(a)
            explore = self.c_b * math.sqrt(math.log(self.N()) / self.N(a))
            action_scores.append(exploit + explore)
        
        return action_scores
    
    def computeUCT_NN(self):
        """使用UCT_NN公式计算每个动作的分数"""
        if self.nn_result is None:
            raise ValueError("NN结果不可用于UCT_NN计算")
            
        action_scores = []
        nn_predictions = self.nn_result.asVector()
        
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a):
                action_scores.append(float('-inf'))
                continue
                
            if self.N(a) == 0:
                action_scores.append(float('inf'))  # 未探索的动作具有无限价值
                continue
                
            # UCT_NN公式: UCT(s,a) + w_a * pi(a|s) / (n(s,a) + 1)
            uct = self.R(a) / self.N(a) + self.c_b * math.sqrt(math.log(self.N()) / self.N(a))
            nn_factor = self.w_a * nn_predictions[a] / (self.N(a) + 1)
            action_scores.append(uct + nn_factor)
        
        return action_scores
    
    def computeUCT_Rave(self):
        """使用UCT_RAVE公式计算每个动作的分数"""
        action_scores = []
        
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a):
                action_scores.append(float('-inf'))
                continue
                
            if self.NRave(a) == 0:
                action_scores.append(float('inf'))  # 未探索的动作具有无限价值
                continue
                
            # UCT_RAVE公式: r_rave(s,a)/n_rave(s,a) + c_b * sqrt(log n_rave(s) / n_rave(s, a))
            exploit = self.RRave(a) / self.NRave(a)
            explore = self.c_b * math.sqrt(math.log(self.NRave()) / self.NRave(a))
            action_scores.append(exploit + explore)
        
        return action_scores
    
    def computeUCT_U_Rave(self):
        """使用UCT_U_RAVE公式计算每个动作的分数"""
        uct_scores = self.computeUCT()
        rave_scores = self.computeUCT_Rave()
        action_scores = []
        
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a):
                action_scores.append(float('-inf'))
                continue
                
            if self.N(a) == 0 or self.NRave(a) == 0:
                action_scores.append(float('inf'))  # 未探索的动作具有无限价值
                continue
                
            # beta(s, a) = sqrt(c_rave / (3 * n(s) + c_rave))
            beta = math.sqrt(self.c_rave / (3 * self.N() + self.c_rave))
            
            # UCT_U,RAVE(s,a) = beta(s,a) * UCT_RAVE(s,a) + (1 - beta(s,a)) * UCT(s,a)
            score = beta * rave_scores[a] + (1 - beta) * uct_scores[a]
            action_scores.append(score)
        
        return action_scores
    
    def computeUCT_NN_Rave(self):
        """使用UCT_NN_RAVE公式计算每个动作的分数"""
        if self.nn_result is None:
            raise ValueError("NN结果不可用于UCT_NN_RAVE计算")
            
        urave_scores = self.computeUCT_U_Rave()
        nn_predictions = self.nn_result.asVector()
        action_scores = []
        
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a):
                action_scores.append(float('-inf'))
                continue
                
            if self.N(a) == 0 or self.NRave(a) == 0:
                action_scores.append(float('inf'))  # 未探索的动作具有无限价值
                continue
                
            # UCT_NN_RAVE(s, a) = UCT_U,RAVE(s, a) + w_a * pi(a|s) / (n(s,a) + 1)
            nn_factor = self.w_a * nn_predictions[a] / (self.N(a) + 1)
            score = urave_scores[a] + nn_factor
            action_scores.append(score)
        
        return action_scores
    
    def computeActionScores(self):
        """根据配置计算每个动作的分数"""
        # 根据配置确定使用哪种评分方法
        if self.requires_nn and self.use_rave:
            return self.computeUCT_NN_Rave()
        elif self.requires_nn and not self.use_rave:
            return self.computeUCT_NN()
        elif not self.requires_nn and self.use_rave:
            return self.computeUCT_U_Rave()
        else:  # not self.requires_nn and not self.use_rave
            return self.computeUCT()
    
    def sampleLegalAction(self, action_scores):
        """基于动作分数的softmax概率抽样一个动作"""
        # 对分数应用softmax
        max_score = max([s for s in action_scores if s != float('-inf')])
        exp_scores = []
        total_exp_score = 0.0
        
        for a in range(self.num_actions):
            if not self.state.isLegalAction(a) or action_scores[a] == float('-inf'):
                exp_scores.append(0.0)
            else:
                # 减去max_score以保持数值稳定性
                exp_score = math.exp(action_scores[a] - max_score)
                exp_scores.append(exp_score)
                total_exp_score += exp_score
        
        # 基于softmax概率抽样一个动作
        if total_exp_score == 0:
            # 如果所有合法动作的分数都是-inf（不应该发生），均匀抽样
            legal_actions = [a for a in range(self.num_actions) if self.state.isLegalAction(a)]
            return random.choice(legal_actions)
        
        r = random.random() * total_exp_score
        cumulative = 0.0
        
        for a in range(self.num_actions):
            cumulative += exp_scores[a]
            if cumulative > r:
                return a
        
        # 备选方案（不应该到达这里）
        legal_actions = [a for a in range(self.num_actions) if self.state.isLegalAction(a)]
        return legal_actions[-1] if legal_actions else 0
    
    def bestLegalAction(self, action_scores):
        """返回得分最高的合法动作"""
        best_score = float('-inf')
        best_action = -1
        
        for a in range(self.num_actions):
            if self.state.isLegalAction(a) and action_scores[a] > best_score:
                best_score = action_scores[a]
                best_action = a
        
        if best_action == -1:
            raise ValueError("没有可用的合法动作")
        
        return best_action
    
    def chooseBestAction(self):
        """从此节点的状态中选择最佳动作"""
        # 检查节点是否需要神经网络但尚未收到结果
        if self.requires_nn and self.nn_result is None:
            raise ValueError("节点需要NN但nn_result为None")
        
        # 计算动作分数
        action_scores = self.computeActionScores()
        
        # 根据分数选择动作
        if self.sample_actions:
            chosen_action = self.sampleLegalAction(action_scores)
        else:
            chosen_action = self.bestLegalAction(action_scores)
        
        # 为所选动作创建或获取子节点
        child = self.makeChild(chosen_action)
        
        return child
    
    def updateStats(self, chosen_action, reward, update_rave_stats=False):
        """更新模拟的必要统计信息"""
        # 更新标准统计信息
        self.num_node_visits += 1
        self.num_edge_traversals[chosen_action] += 1
        self.edge_rewards[chosen_action] += reward
        
        # 如果需要，更新RAVE统计信息
        if update_rave_stats and self.use_rave:
            self.num_node_visits_rave += 1
            self.num_edge_traversals_rave[chosen_action] += 1
            self.edge_rewards_rave[chosen_action] += reward
    
    def updateStatsRave(self, chosen_actions, reward):
        """更新列表中所有动作的RAVE统计信息"""
        if not self.use_rave:
            return
            
        for action in chosen_actions:
            if action < 0 or action >= self.num_actions:
                continue
                
            self.num_node_visits_rave += 1
            self.num_edge_traversals_rave[action] += 1
            self.edge_rewards_rave[action] += reward


# MCTS核心函数实现
def propagateStats(node, reward, actions_taken=[]):
    """
    日h求的节点的统计信息回游到根节点
    
    Args:
        node: 开始回游的节点
        reward: 回游的奖励值
        actions_taken: 模拟过程中执行的动作列表（用于RAVE）
    """
    # 获取该节点的根节点
    root = node.getRoot()
    
    # 如果节点是根节点，则回游完成
    if node.isRoot():
        return
    
    # 获取父节点和当前节点的动作索引
    parent = node.getParent()
    action_index = node.getChildIndex()
    
    # 更新父节点的标准统计信息
    parent.updateStats(action_index, reward, update_rave_stats=True)
    
    # 如果使用RAVE，更新RAVE统计信息
    if parent.usesRave():
        parent.updateStatsRave(actions_taken, reward)
    
    # 递归回游到根节点
    propagateStats(parent, reward, actions_taken)
    
    # 如果当前节点为根节点的直接子节点，标记一次模拟完成
    if parent.isRoot():
        root.markSimulationFinished()

def rolloutSimulation(node, max_depth=DEFAULT_MAX_DEPTH):
    """
    从给定节点开始随机选择动作并执行直到达到终止状态
    
    Args:
        node: 起始节点
        max_depth: 最大搜索深度
    
    Returns:
        终止节点
    """
    # 记录模拟过程中执行的动作
    actions_taken = []
    current_node = node
    depth = 0
    
    # 执行随机策略直到到达终止状态或最大深度
    while not current_node.isTerminal() and depth < max_depth:
        # 获取当前状态的所有合法动作
        state = current_node.getState()
        legal_actions = [a for a in range(current_node.getNumActions()) if state.isLegalAction(a)]
        
        if not legal_actions:
            break
        
        # 随机选择一个合法动作
        action = random.choice(legal_actions)
        actions_taken.append(action)
        
        # 创建或获取子节点
        current_node = current_node.makeChild(action)
        depth += 1
    
    # 返回终止节点与模拟中执行的动作
    return current_node, actions_taken

def runMCTS(node):
    """
    继一个MCTS模拟并更新统计信息
    
    Args:
        node: 模拟的起始节点
    """
    # 如果节点需要神经网络且没有提交其状态向量
    if node.neverSubmittedToNN():
        # 在真实实现中，这里应该提交到神经网络队列
        node.markSubmittedToNN()
        return

    # 如果节点已提交到NN但尚未收到结果
    if node.awaitingNNResults():
        return

    current_node = node
    
    # 选择阶段：使用UCT选择策略向下遍历树
    while not current_node.isTerminal() and all(current_node.getChild(a) is not None 
                                           for a in range(current_node.getNumActions()) 
                                           if current_node.getState().isLegalAction(a)):
        current_node = current_node.chooseBestAction()
    
    # 扩展阶段：如果节点不是终止节点，选择一个未扩展的子节点
    if not current_node.isTerminal():
        current_node = current_node.chooseBestAction()
    
    # 模拟阶段：从当前节点进行随机模拟直到终止状态
    terminal_node, actions_taken = rolloutSimulation(current_node)
    
    # 回游阶段：将结果回游到根节点
    reward = terminal_node.getState().reward()
    propagateStats(current_node, reward, actions_taken)

def runAllSimulations(node, num_simulations=None):
    """
    为给定节点运行所有模拟
    
    Args:
        node: 模拟的起始节点
        num_simulations: 要运行的模拟数，如果为None则使用节点的默认值
    
    Returns:
        最佳动作分布
    """
    if num_simulations is None:
        num_simulations = node.total_num_simulations
    
    # 运行指定数量的模拟
    for i in range(num_simulations):
        runMCTS(node)
    
    # 返回动作分布
    return ActionDistribution(action_dist=node.getActionDistribution())

# 输入输出函数
def readInputData(input_data_path, num_simulations=DEFAULT_NUM_SIMULATIONS, num_states=0):
    """
    从给定路径读取输入数据并创建MCTS节点
    
    Args:
        input_data_path: 输入数据路径
        num_simulations: 每个状态运行的模拟数
        num_states: 要读取的状态数量，0表示读取所有状态
    
    Returns:
        MCTS节点列表
    """
    nodes = []
    
    # 确保路径存在
    if not os.path.exists(input_data_path):
        raise ValueError(f"输入路径不存在：{input_data_path}")
    
    # 检查路径是目录还是文件
    if os.path.isdir(input_data_path):
        # 找到目录中的所有CSV文件
        csv_files = [os.path.join(input_data_path, f) for f in os.listdir(input_data_path) 
                    if f.endswith('.csv')]
        
        # 处理每个CSV文件
        for csv_file in csv_files:
            file_nodes = readCSVFile(csv_file, num_simulations, num_states)
            nodes.extend(file_nodes)
            
            # 如果达到指定的状态数量，停止读取
            if num_states > 0 and len(nodes) >= num_states:
                nodes = nodes[:num_states]
                break
    else:
        # 直接读取单个文件
        nodes = readCSVFile(input_data_path, num_simulations, num_states)
    
    return nodes

def readCSVFile(csv_file_path, num_simulations=DEFAULT_NUM_SIMULATIONS, num_states=0):
    """
    从给定CSV文件读取状态并创建MCTS节点
    
    Args:
        csv_file_path: CSV文件路径
        num_simulations: 每个状态运行的模拟数
        num_states: 要读取的状态数量，0表示读取所有状态
    
    Returns:
        MCTS节点列表
    """
    nodes = []
    
    with open(csv_file_path, 'r') as f:
        csv_reader = csv.reader(f)
        
        for row in csv_reader:
            if not row or not row[0]:
                continue
                
            # 从每一行创建状态向量
            state_vector = StateVector(map(float, row))
            
            # 从状态向量创建HexState实例
            try:
                state = HexState.fromStateVector(state_vector)
                node = MCTS_Node(state, num_simulations=num_simulations)
                
                # 运行指定数量的模拟
                runAllSimulations(node, num_simulations)
                nodes.append(node)
                
                # 如果达到指定的状态数量，停止读取
                if num_states > 0 and len(nodes) >= num_states:
                    break
            except Exception as e:
                print(f"无法从状态向量创建HexState: {e}")
    
    return nodes

def writeBatchToFile(nodes, output_dir, begin_file_num=0):
    """
    将节点列表写入CSV文件
    
    Args:
        nodes: MCTS节点列表
        output_dir: 输出目录
        begin_file_num: 起始文件编号
    """
    # 创建必要的目录
    states_dir = os.path.join(output_dir, 'states')
    actions_dir = os.path.join(output_dir, 'action_distributions')
    
    os.makedirs(states_dir, exist_ok=True)
    os.makedirs(actions_dir, exist_ok=True)
    
    # 写入状态和动作分布
    writeStatesToFile(nodes, states_dir, begin_file_num)
    writeActionDistributionsToFile(nodes, actions_dir, begin_file_num)

def writeStatesToFile(nodes, output_dir, next_file_num=0):
    """
    将状态向量写入CSV文件
    
    Args:
        nodes: MCTS节点列表
        output_dir: 输出目录
        next_file_num: 起始文件编号
    """
    states_per_file = DEFAULT_STATES_PER_FILE
    file_num = next_file_num
    
    for i in range(0, len(nodes), states_per_file):
        batch = nodes[i:i+states_per_file]
        file_path = os.path.join(output_dir, f'states{file_num}.csv')
        
        with open(file_path, 'w', newline='') as f:
            for node in batch:
                f.write(node.getStateVector().asCSVString() + '\n')
        
        file_num += 1

def writeActionDistributionsToFile(nodes, output_dir, next_file_num=0):
    """
    将动作分布写入CSV文件
    
    Args:
        nodes: MCTS节点列表
        output_dir: 输出目录
        next_file_num: 起始文件编号
    """
    states_per_file = DEFAULT_STATES_PER_FILE
    file_num = next_file_num
    
    for i in range(0, len(nodes), states_per_file):
        batch = nodes[i:i+states_per_file]
        file_path = os.path.join(output_dir, f'action_distributions{file_num}.csv')
        
        with open(file_path, 'w', newline='') as f:
            for node in batch:
                action_distribution = ActionDistribution(action_dist=node.getActionDistribution())
                f.write(action_distribution.asCSVString() + '\n')
        
        file_num += 1
