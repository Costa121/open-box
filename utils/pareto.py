"""
Pareto工具函数：用于多目标优化中的Pareto前沿计算和管理
迁移自OpenBox的pareto相关功能
"""

import numpy as np
from typing import List, Tuple, Optional, Union


def is_dominated(point: np.ndarray, other_point: np.ndarray) -> bool:
    """
    检查point是否被other_point支配（假设最小化所有目标）
    
    Args:
        point: 待检查的点
        other_point: 参考点
        
    Returns:
        True如果point被other_point支配
    """
    return np.all(other_point <= point) and np.any(other_point < point)


def is_non_dominated(point: np.ndarray, points: np.ndarray) -> bool:
    """
    检查point是否为非支配解
    
    Args:
        point: 待检查的点
        points: 所有点的集合
        
    Returns:
        True如果point是非支配的
    """
    for other in points:
        if is_dominated(point, other):
            return False
    return True


def get_pareto_front(objectives: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从目标值集合中提取Pareto前沿
    
    Args:
        objectives: 形状为(n_points, n_objectives)的目标值数组
        
    Returns:
        pareto_front: Pareto前沿上的点
        pareto_indices: Pareto前沿点在原数组中的索引
    """
    if len(objectives) == 0:
        return np.array([]), np.array([], dtype=int)
    
    n_points = objectives.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            # 检查i是否被j支配
            if is_dominated(objectives[i], objectives[j]):
                is_pareto[i] = False
                break
    
    pareto_indices = np.where(is_pareto)[0]
    pareto_front = objectives[is_pareto]
    
    return pareto_front, pareto_indices


def update_pareto_set(
    current_pareto: np.ndarray,
    current_indices: np.ndarray,
    new_point: np.ndarray,
    new_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    增量更新Pareto集合
    
    Args:
        current_pareto: 当前Pareto前沿
        current_indices: 当前Pareto前沿的索引
        new_point: 新增的点
        new_index: 新点的索引
        
    Returns:
        更新后的Pareto前沿和索引
    """
    if len(current_pareto) == 0:
        return np.array([new_point]), np.array([new_index])
    
    # 检查新点是否被现有Pareto点支配
    new_point_dominated = False
    dominated_mask = np.ones(len(current_pareto), dtype=bool)
    
    for i, pareto_point in enumerate(current_pareto):
        if is_dominated(new_point, pareto_point):
            new_point_dominated = True
            break
        if is_dominated(pareto_point, new_point):
            dominated_mask[i] = False
    
    if new_point_dominated:
        return current_pareto, current_indices
    
    # 移除被新点支配的点
    new_pareto = current_pareto[dominated_mask]
    new_indices = current_indices[dominated_mask]
    
    # 添加新点
    new_pareto = np.vstack([new_pareto, new_point]) if len(new_pareto) > 0 else np.array([new_point])
    new_indices = np.append(new_indices, new_index)
    
    return new_pareto, new_indices


def compute_crowding_distance(pareto_front: np.ndarray) -> np.ndarray:
    """
    计算Pareto前沿上每个点的拥挤距离（用于解的多样性选择）
    
    Args:
        pareto_front: Pareto前沿点集
        
    Returns:
        每个点的拥挤距离
    """
    n_points, n_objectives = pareto_front.shape
    
    if n_points <= 2:
        return np.full(n_points, np.inf)
    
    distances = np.zeros(n_points)
    
    for obj_idx in range(n_objectives):
        # 按当前目标排序
        sorted_indices = np.argsort(pareto_front[:, obj_idx])
        sorted_values = pareto_front[sorted_indices, obj_idx]
        
        # 边界点距离设为无穷大
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # 计算中间点的距离
        obj_range = sorted_values[-1] - sorted_values[0]
        if obj_range > 0:
            for i in range(1, n_points - 1):
                distances[sorted_indices[i]] += (
                    sorted_values[i + 1] - sorted_values[i - 1]
                ) / obj_range
    
    return distances


def select_from_pareto(
    pareto_front: np.ndarray,
    pareto_indices: np.ndarray,
    n_select: int
) -> np.ndarray:
    """
    从Pareto前沿中选择指定数量的解（基于拥挤距离）
    
    Args:
        pareto_front: Pareto前沿
        pareto_indices: 对应的原始索引
        n_select: 要选择的数量
        
    Returns:
        选中点的原始索引
    """
    if n_select >= len(pareto_front):
        return pareto_indices
    
    crowding = compute_crowding_distance(pareto_front)
    selected_mask = np.argsort(crowding)[::-1][:n_select]
    
    return pareto_indices[selected_mask]