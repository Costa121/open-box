"""
多目标采集函数基类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from itertools import product

from ..base import AcquisitionFunction, SurrogateModel


class MultiObjectiveAcquisition(AcquisitionFunction):
    """
    多目标采集函数基类
    
    与新框架的 AcquisitionFunction 统一接口，同时支持多目标优化特有功能
    """
    
    def __init__(
        self, 
        model: List[SurrogateModel],  # 多目标：每个目标一个模型
        ref_point: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Args:
            model: 代理模型列表，每个目标一个模型
            ref_point: 参考点（用于计算超体积）
        """
        # 注意：不调用 super().__init__，因为父类期望单个 model
        # 我们直接设置属性
        self.model = model
        self.long_name = self.__class__.__name__
        
        self.num_objectives = len(model)
        self.ref_point = np.asarray(ref_point) if ref_point is not None else None
        
        # 多目标特有属性（由外部更新）
        self.pareto_front = None
        self.cell_lower_bounds = None  # EHVI 需要
        self.cell_upper_bounds = None  # EHVI 需要
        
        # 训练数据
        self.X = None
        self.Y = None
    
    def update(self, context=None, **kwargs) -> None:
        """
        更新采集函数状态
        
        支持两种更新方式：
        1. 通过 context（新框架）
        2. 通过 kwargs（兼容 OpenBox）
        """
        if context is not None:
            # 新框架：从 context 更新
            self._update_from_context(context, **kwargs)
        else:
            # 兼容 OpenBox：直接从 kwargs 更新
            self._update_from_kwargs(**kwargs)
    
    def _update_from_context(self, context, **kwargs) -> None:
        """从 AcquisitionContext 更新（新框架）"""
        target_task = context.get_target_task()
        
        # 更新模型（如果 context 提供了）
        if hasattr(context, 'get_main_surrogate'):
            main_surrogate = context.get_main_surrogate()
            if main_surrogate is not None and isinstance(main_surrogate, list):
                self.model = main_surrogate
        
        # 更新训练数据
        if hasattr(target_task, 'history'):
            self.X = target_task.history.get_config_array()
            self.Y = target_task.history.get_objectives()
        
        # 从 kwargs 更新其他属性
        self._update_from_kwargs(**kwargs)
    
    def _update_from_kwargs(self, **kwargs) -> None:
        """从 kwargs 更新（兼容 OpenBox）"""
        for key in ['X', 'Y', 'pareto_front', 'ref_point', 
                    'cell_lower_bounds', 'cell_upper_bounds']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        # 处理 model 更新
        if 'model' in kwargs:
            self.model = kwargs['model']
    
    def __call__(self, X: np.ndarray, convert: bool = True, **kwargs) -> np.ndarray:
        """
        调用接口，统一与单目标
        
        Args:
            X: 候选点
            convert: 是否转换配置（占位，多目标通常不需要）
            
        Returns:
            采集函数值
        """
        return self._compute(X, **kwargs)
    
    @abstractmethod
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        计算采集函数值（子类必须实现）
        
        Args:
            X: 候选点，形状为 (n_points, n_features)
            
        Returns:
            采集值，形状为 (n_points, 1)
        """
        pass
    
    def predict_marginalized(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取所有目标的预测均值和方差
        
        Args:
            X: 输入点，形状为 (n_points, n_features)
            
        Returns:
            means: (n_points, n_objectives)
            stds: (n_points, n_objectives)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_points = X.shape[0]
        means = np.zeros((n_points, self.num_objectives))
        stds = np.zeros((n_points, self.num_objectives))
        
        for i in range(self.num_objectives):
            # 兼容不同的预测接口
            if hasattr(self.model[i], 'predict_marginalized_over_instances'):
                # OpenBox 接口
                m, v = self.model[i].predict_marginalized_over_instances(X)
                means[:, i] = m.flatten()
                stds[:, i] = np.sqrt(v.flatten())
            else:
                # 标准接口
                m, v = self.model[i].predict(X)
                means[:, i] = m.flatten()
                stds[:, i] = np.sqrt(v.flatten())
        
        return means, stds


class MultiObjectiveConstrainedAcquisition(MultiObjectiveAcquisition):
    """
    带约束的多目标采集函数基类
    """
    
    def __init__(
        self,
        model: List[SurrogateModel],
        constraint_models: List[SurrogateModel],
        ref_point: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Args:
            model: 目标模型列表
            constraint_models: 约束模型列表
            ref_point: 参考点
        """
        super().__init__(model, ref_point, **kwargs)
        self.constraint_models = constraint_models
        self.num_constraints = len(constraint_models)
        self.constraint_perfs = None
    
    def _update_from_kwargs(self, **kwargs) -> None:
        """扩展父类，增加约束相关更新"""
        super()._update_from_kwargs(**kwargs)
        
        if 'constraint_models' in kwargs:
            self.constraint_models = kwargs['constraint_models']
        if 'constraint_perfs' in kwargs:
            self.constraint_perfs = kwargs['constraint_perfs']
    
    def compute_constraint_probability(self, X: np.ndarray) -> np.ndarray:
        """
        计算约束满足概率 P(c(x) <= 0)
        
        Args:
            X: 输入点
            
        Returns:
            约束满足的联合概率，形状为 (n_points, 1)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        prob = np.ones((X.shape[0], 1))
        
        for c_model in self.constraint_models:
            if hasattr(c_model, 'predict_marginalized_over_instances'):
                m, v = c_model.predict_marginalized_over_instances(X)
            else:
                m, v = c_model.predict(X)
            
            s = np.sqrt(v)
            s[s < 1e-10] = 1e-10
            
            # P(c(x) <= 0) = Φ(-m/s)
            from scipy.stats import norm
            prob *= norm.cdf(-m / s)
        
        return prob