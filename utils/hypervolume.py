"""
超体积(Hypervolume)计算工具
迁移自OpenBox的hypervolume计算功能
"""

import numpy as np
from typing import Optional, List


class HypervolumeCalculator:
    """
    超体积计算器
    支持2D和3D的精确计算，更高维度使用Monte Carlo近似
    """
    
    def __init__(self, ref_point: np.ndarray):
        """
        Args:
            ref_point: 参考点，应支配所有Pareto前沿点
        """
        self.ref_point = np.array(ref_point)
        self.n_objectives = len(ref_point)
    
    def compute(self, pareto_front: np.ndarray) -> float:
        """
        计算Pareto前沿的超体积
        
        Args:
            pareto_front: 形状为(n_points, n_objectives)的Pareto前沿
            
        Returns:
            超体积值
        """
        if len(pareto_front) == 0:
            return 0.0
        
        # 过滤掉被参考点支配的点
        valid_mask = np.all(pareto_front < self.ref_point, axis=1)
        valid_front = pareto_front[valid_mask]
        
        if len(valid_front) == 0:
            return 0.0
        
        if self.n_objectives == 2:
            return self._compute_2d(valid_front)
        elif self.n_objectives == 3:
            return self._compute_3d(valid_front)
        else:
            return self._compute_monte_carlo(valid_front)
    
    def _compute_2d(self, pareto_front: np.ndarray) -> float:
        """2D精确计算"""
        # 按第一个目标排序
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        hv = 0.0
        prev_y = self.ref_point[1]
        
        for point in sorted_front:
            hv += (self.ref_point[0] - point[0]) * (prev_y - point[1])
            prev_y = point[1]
        
        return hv
    
    def _compute_3d(self, pareto_front: np.ndarray) -> float:
        """3D精确计算 - 使用WFG算法的简化版本"""
        # 按第一个目标排序
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        hv = 0.0
        n = len(sorted_front)
        
        for i in range(n):
            # 计算每个切片的贡献
            if i == n - 1:
                x_width = self.ref_point[0] - sorted_front[i, 0]
            else:
                x_width = sorted_front[i + 1, 0] - sorted_front[i, 0]
            
            # 获取当前切片中的2D Pareto前沿
            slice_points = sorted_front[:i + 1, 1:]
            slice_front, _ = self._get_2d_pareto(slice_points)
            
            # 计算2D超体积
            slice_hv = self._compute_2d_with_ref(slice_front, self.ref_point[1:])
            hv += x_width * slice_hv
        
        return hv
    
    def _get_2d_pareto(self, points: np.ndarray):
        """获取2D Pareto前沿"""
        from utils.pareto import get_pareto_front
        return get_pareto_front(points)
    
    def _compute_2d_with_ref(self, pareto_front: np.ndarray, ref: np.ndarray) -> float:
        """使用指定参考点计算2D超体积"""
        if len(pareto_front) == 0:
            return 0.0
        
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        hv = 0.0
        prev_y = ref[1]
        
        for point in sorted_front:
            if point[0] < ref[0] and point[1] < ref[1]:
                hv += (ref[0] - point[0]) * (prev_y - point[1])
                prev_y = point[1]
        
        return hv
    
    def _compute_monte_carlo(
        self, 
        pareto_front: np.ndarray, 
        n_samples: int = 100000
    ) -> float:
        """高维Monte Carlo近似"""
        # 确定采样边界
        lower = np.min(pareto_front, axis=0)
        upper = self.ref_point
        
        # 生成随机样本
        samples = np.random.uniform(lower, upper, size=(n_samples, self.n_objectives))
        
        # 检查每个样本是否被Pareto前沿支配
        dominated_count = 0
        for sample in samples:
            for point in pareto_front:
                if np.all(point <= sample):
                    dominated_count += 1
                    break
        
        # 计算超体积
        box_volume = np.prod(upper - lower)
        return box_volume * dominated_count / n_samples
    
    def compute_contribution(
        self, 
        pareto_front: np.ndarray, 
        point_index: int
    ) -> float:
        """
        计算单个点对超体积的贡献
        
        Args:
            pareto_front: Pareto前沿
            point_index: 要计算贡献的点的索引
            
        Returns:
            该点的超体积贡献
        """
        total_hv = self.compute(pareto_front)
        
        # 移除该点后的超体积
        reduced_front = np.delete(pareto_front, point_index, axis=0)
        reduced_hv = self.compute(reduced_front)
        
        return total_hv - reduced_hv


def compute_hypervolume(
    pareto_front: np.ndarray, 
    ref_point: Optional[np.ndarray] = None
) -> float:
    """
    便捷函数：计算超体积
    
    Args:
        pareto_front: Pareto前沿
        ref_point: 参考点（如未提供，自动计算）
        
    Returns:
        超体积值
    """
    if len(pareto_front) == 0:
        return 0.0
    
    if ref_point is None:
        # 自动计算参考点：取每个目标的最大值 + 10%边距
        ref_point = np.max(pareto_front, axis=0) * 1.1
    
    calculator = HypervolumeCalculator(ref_point)
    return calculator.compute(pareto_front)


def get_default_ref_point(
    objectives: np.ndarray, 
    margin: float = 0.1
) -> np.ndarray:
    """
    根据当前观测值计算默认参考点
    
    Args:
        objectives: 所有目标值
        margin: 边距比例
        
    Returns:
        参考点
    """
    if len(objectives) == 0:
        raise ValueError("无法从空数据计算参考点")
    
    max_vals = np.max(objectives, axis=0)
    min_vals = np.min(objectives, axis=0)
    ranges = max_vals - min_vals
    
    # 参考点 = 最大值 + 边距
    ref_point = max_vals + margin * np.maximum(ranges, 1e-6)
    
    return ref_point