"""
多目标优化历史管理器
扩展现有的HistoryManager以支持多目标优化
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import os

from utils.pareto import get_pareto_front, update_pareto_set
from utils.hypervolume import compute_hypervolume, get_default_ref_point


@dataclass
class MultiObjectiveObservation:
    """多目标观测记录"""
    config: Dict[str, Any]
    objectives: np.ndarray  # 多个目标值
    constraints: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    
    def is_feasible(self, constraint_threshold: float = 0.0) -> bool:
        """检查是否满足约束"""
        if self.constraints is None:
            return True
        return np.all(self.constraints <= constraint_threshold)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'config': self.config,
            'objectives': self.objectives.tolist(),
            'constraints': self.constraints.tolist() if self.constraints is not None else None,
            'info': self.info,
            'iteration': self.iteration
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MultiObjectiveObservation':
        """从字典创建"""
        return cls(
            config=data['config'],
            objectives=np.array(data['objectives']),
            constraints=np.array(data['constraints']) if data.get('constraints') else None,
            info=data.get('info', {}),
            iteration=data.get('iteration', 0)
        )


class MultiObjectiveHistoryManager:
    """
    多目标优化历史管理器
    
    负责：
    - 存储所有观测记录
    - 维护Pareto前沿
    - 计算超体积等指标
    - 支持历史数据持久化
    """
    
    def __init__(
        self,
        n_objectives: int,
        n_constraints: int = 0,
        ref_point: Optional[np.ndarray] = None,
        task_id: Optional[str] = None
    ):
        """
        Args:
            n_objectives: 目标数量
            n_constraints: 约束数量
            ref_point: 超体积计算的参考点
            task_id: 任务标识（用于持久化）
        """
        self.n_objectives = n_objectives
        self.n_constraints = n_constraints
        self.ref_point = np.array(ref_point) if ref_point is not None else None
        self.task_id = task_id
        
        # 观测历史
        self.observations: List[MultiObjectiveObservation] = []
        
        # Pareto前沿缓存
        self._pareto_front: Optional[np.ndarray] = None
        self._pareto_indices: Optional[np.ndarray] = None
        self._pareto_configs: Optional[List[Dict]] = None
        
        # 性能指标历史
        self.hypervolume_history: List[float] = []
        
        # 迭代计数
        self.iteration = 0
    
    def add_observation(
        self,
        config: Dict[str, Any],
        objectives: np.ndarray,
        constraints: Optional[np.ndarray] = None,
        info: Optional[Dict] = None
    ) -> None:
        """
        添加新的观测记录
        
        Args:
            config: 配置参数
            objectives: 目标值（数组）
            constraints: 约束值（可选）
            info: 额外信息
        """
        objectives = np.atleast_1d(objectives)
        
        if len(objectives) != self.n_objectives:
            raise ValueError(
                f"目标数量不匹配：期望{self.n_objectives}，实际{len(objectives)}"
            )
        
        if constraints is not None:
            constraints = np.atleast_1d(constraints)
            if len(constraints) != self.n_constraints:
                raise ValueError(
                    f"约束数量不匹配：期望{self.n_constraints}，实际{len(constraints)}"
                )
        
        obs = MultiObjectiveObservation(
            config=config,
            objectives=objectives,
            constraints=constraints,
            info=info or {},
            iteration=self.iteration
        )
        
        self.observations.append(obs)
        self.iteration += 1
        
        # 更新Pareto前沿
        self._update_pareto()
        
        # 更新超体积历史
        self._update_hypervolume()
    
    def _update_pareto(self) -> None:
        """更新Pareto前沿"""
        if len(self.observations) == 0:
            self._pareto_front = None
            self._pareto_indices = None
            self._pareto_configs = None
            return
        
        # 只考虑可行解
        feasible_indices = [
            i for i, obs in enumerate(self.observations)
            if obs.is_feasible()
        ]
        
        if len(feasible_indices) == 0:
            self._pareto_front = None
            self._pareto_indices = None
            self._pareto_configs = None
            return
        
        # 获取可行解的目标值
        feasible_objectives = np.array([
            self.observations[i].objectives for i in feasible_indices
        ])
        
        # 计算Pareto前沿
        pareto_front, pareto_local_indices = get_pareto_front(feasible_objectives)
        
        # 映射回原始索引
        self._pareto_indices = np.array([
            feasible_indices[i] for i in pareto_local_indices
        ])
        self._pareto_front = pareto_front
        self._pareto_configs = [
            self.observations[i].config for i in self._pareto_indices
        ]
    
    def _update_hypervolume(self) -> None:
        """更新超体积指标"""
        if self._pareto_front is None or len(self._pareto_front) == 0:
            self.hypervolume_history.append(0.0)
            return
        
        # 确定参考点
        ref_point = self.ref_point
        if ref_point is None:
            all_objectives = self.get_all_objectives()
            ref_point = get_default_ref_point(all_objectives)
        
        hv = compute_hypervolume(self._pareto_front, ref_point)
        self.hypervolume_history.append(hv)
    
    def get_pareto_front(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        获取当前Pareto前沿
        
        Returns:
            pareto_objectives: Pareto前沿的目标值
            pareto_configs: 对应的配置
        """
        return self._pareto_front, self._pareto_configs
    
    def get_all_objectives(self) -> np.ndarray:
        """获取所有观测的目标值"""
        if len(self.observations) == 0:
            return np.array([]).reshape(0, self.n_objectives)
        return np.array([obs.objectives for obs in self.observations])
    
    def get_all_configs(self) -> List[Dict]:
        """获取所有配置"""
        return [obs.config for obs in self.observations]
    
    def get_feasible_observations(self) -> List[MultiObjectiveObservation]:
        """获取所有可行的观测"""
        return [obs for obs in self.observations if obs.is_feasible()]
    
    def get_hypervolume(self) -> float:
        """获取当前超体积"""
        if len(self.hypervolume_history) == 0:
            return 0.0
        return self.hypervolume_history[-1]
    
    def get_best_observation(
        self, 
        objective_index: int = 0
    ) -> Optional[MultiObjectiveObservation]:
        """
        获取单个目标上的最优观测（用于参考）
        
        Args:
            objective_index: 目标索引
            
        Returns:
            在该目标上最优的可行观测
        """
        feasible = self.get_feasible_observations()
        if len(feasible) == 0:
            return None
        
        best_idx = np.argmin([obs.objectives[objective_index] for obs in feasible])
        return feasible[best_idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'n_observations': len(self.observations),
            'n_feasible': len(self.get_feasible_observations()),
            'n_pareto': len(self._pareto_front) if self._pareto_front is not None else 0,
            'current_hypervolume': self.get_hypervolume(),
            'hypervolume_history': self.hypervolume_history.copy()
        }
        
        if self._pareto_front is not None and len(self._pareto_front) > 0:
            stats['pareto_front'] = self._pareto_front.tolist()
            stats['pareto_objective_ranges'] = {
                f'obj_{i}': {
                    'min': float(np.min(self._pareto_front[:, i])),
                    'max': float(np.max(self._pareto_front[:, i]))
                }
                for i in range(self.n_objectives)
            }
        
        return stats
    
    def save(self, filepath: str) -> None:
        """保存历史到文件"""
        data = {
            'n_objectives': self.n_objectives,
            'n_constraints': self.n_constraints,
            'ref_point': self.ref_point.tolist() if self.ref_point is not None else None,
            'task_id': self.task_id,
            'observations': [obs.to_dict() for obs in self.observations],
            'hypervolume_history': self.hypervolume_history
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MultiObjectiveHistoryManager':
        """从文件加载历史"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        manager = cls(
            n_objectives=data['n_objectives'],
            n_constraints=data.get('n_constraints', 0),
            ref_point=np.array(data['ref_point']) if data.get('ref_point') else None,
            task_id=data.get('task_id')
        )
        
        for obs_data in data['observations']:
            obs = MultiObjectiveObservation.from_dict(obs_data)
            manager.observations.append(obs)
        
        manager.hypervolume_history = data.get('hypervolume_history', [])
        manager._update_pareto()
        manager.iteration = len(manager.observations)
        
        return manager
    
    def print_summary(self) -> None:
        """打印历史摘要"""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("多目标优化历史摘要")
        print("=" * 60)
        print(f"总观测数: {stats['n_observations']}")
        print(f"可行解数: {stats['n_feasible']}")
        print(f"Pareto解数: {stats['n_pareto']}")
        print(f"当前超体积: {stats['current_hypervolume']:.6f}")
        
        if 'pareto_objective_ranges' in stats:
            print("\nPareto前沿目标范围:")
            for obj_name, ranges in stats['pareto_objective_ranges'].items():
                print(f"  {obj_name}: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
        
        print("=" * 60)


def create_multi_objective_history(
    config: Dict[str, Any]
) -> MultiObjectiveHistoryManager:
    """
    工厂函数：根据配置创建多目标历史管理器
    
    Args:
        config: 配置字典，包含n_objectives, n_constraints, ref_point等
        
    Returns:
        MultiObjectiveHistoryManager实例
    """
    return MultiObjectiveHistoryManager(
        n_objectives=config.get('n_objectives', 2),
        n_constraints=config.get('n_constraints', 0),
        ref_point=config.get('ref_point'),
        task_id=config.get('task_id')
    )