"""
多目标贝叶斯优化Advisor
基于OpenBox实现，集成到现有框架中
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ConfigSpace import ConfigurationSpace, Configuration
from scipy.stats import norm

# 使用OpenBox的核心组件
from openbox.core.base import build_surrogate, build_acq_func
from openbox.acq_optimizer import build_acq_optimizer
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS, FAILED
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox import logger

# 继承现有的BaseAdvisor
from .base import BaseAdvisor


class MultiObjectiveBOAdvisor(BaseAdvisor):
    """
    多目标贝叶斯优化建议器
    
    基于OpenBox实现，支持:
    - 多个目标的同时优化
    - EHVI/ParEGO等多目标采集函数
    - Pareto前沿管理
    - 可选的约束处理
    """
    
    def __init__(
        self,
        config_space: ConfigurationSpace,
        num_objectives: int = 2,
        num_constraints: int = 0,
        ref_point: Optional[List[float]] = None,
        surrogate_type: str = 'gp',
        acq_type: str = 'ehvi',
        acq_optimizer_type: str = 'random_scipy',
        init_strategy: str = 'random_explore_first',
        initial_trials: int = None,
        random_state: int = None,
        task_id: str = 'MOBO',
        output_dir: str = 'logs',
        **kwargs
    ):
        # 保存多目标特有的参数，在调用父类之前
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self._ref_point = ref_point
        self.surrogate_type = surrogate_type
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type
        self.init_strategy = init_strategy
        self.task_id = task_id
        self.output_dir = output_dir
        
        # 随机状态
        if random_state is None:
            random_state = np.random.randint(0, 10000)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # 初始化样本数
        if initial_trials is None:
            initial_trials = max(5, 2 * len(config_space.get_hyperparameters()))
        self.init_num = initial_trials
        
        # 调用父类初始化
        super().__init__(config_space, **kwargs)
        
        # 使用OpenBox的History - 在父类初始化之后
        try:
            self.history = History(
                task_id=task_id,
                num_objectives=num_objectives,
                num_constraints=num_constraints,
                config_space=config_space,
                ref_point=ref_point
            )
        except TypeError:
            # 如果参数不匹配，尝试其他参数组合
            try:
                self.history = History(
                    num_objectives=num_objectives,
                    num_constraints=num_constraints,
                    config_space=config_space,
                    meta_info={'task_id': task_id}
                )
            except:
                # 最简化的初始化
                self.history = History(
                    num_objectives=num_objectives,
                    num_constraints=num_constraints,
                    config_space=config_space
                )
        
        # 初始化代理模型
        self.surrogate_models = []
        self._init_surrogates()
        
        # 约束代理模型
        self.constraint_models = []
        if num_constraints > 0:
            self._init_constraint_surrogates()
        
        # 初始化采集函数
        self.acquisition_function = None
        self._init_acquisition_function()
        
        # 采集函数优化器
        self.acq_optimizer = None
        self._init_acq_optimizer()
        
        # 初始配置
        self.initial_configurations = self._create_initial_design()
        
        logger.info(f"MultiObjectiveBOAdvisor initialized: "
                   f"objectives={num_objectives}, constraints={num_constraints}, "
                   f"surrogate={surrogate_type}, acq={acq_type}")
    
    def _init_surrogates(self) -> None:
        """初始化代理模型"""
        for i in range(self.num_objectives):
            surrogate = build_surrogate(
                func_str=self.surrogate_type,
                config_space=self.config_space,
                rng=np.random.RandomState(self.random_state + i)
            )
            self.surrogate_models.append(surrogate)
    
    def _init_constraint_surrogates(self) -> None:
        """初始化约束代理模型"""
        for i in range(self.num_constraints):
            surrogate = build_surrogate(
                func_str=self.surrogate_type,
                config_space=self.config_space,
                rng=np.random.RandomState(self.random_state + self.num_objectives + i)
            )
            self.constraint_models.append(surrogate)
    
    def _init_acquisition_function(self) -> None:
        """初始化采集函数"""
        try:
            if self.acq_type == 'parego':
                self.acquisition_function = build_acq_func(
                    func_str='ei',
                    model=None,
                    constraint_models=self.constraint_models if self.num_constraints > 0 else None
                )
            else:
                self.acquisition_function = build_acq_func(
                    func_str=self.acq_type,
                    model=self.surrogate_models,
                    constraint_models=self.constraint_models if self.num_constraints > 0 else None,
                    ref_point=self._ref_point
                )
        except Exception as e:
            logger.warning(f"Failed to build acq_func '{self.acq_type}': {e}. Using fallback.")
            self.acquisition_function = None
    
    def _init_acq_optimizer(self) -> None:
        """初始化采集函数优化器"""
        try:
            self.acq_optimizer = build_acq_optimizer(
                func_str=self.acq_optimizer_type,
                config_space=self.config_space,
                rng=self.rng
            )
        except Exception as e:
            logger.warning(f"Failed to build acq_optimizer: {e}. Using random sampling.")
            self.acq_optimizer = None
    
    def _create_initial_design(self) -> List[Configuration]:
        """创建初始设计"""
        try:
            from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler
            
            if self.init_strategy == 'sobol':
                sampler = SobolSampler(
                    self.config_space, 
                    self.init_num,
                    random_state=self.random_state
                )
                return sampler.generate(return_config=True)
            elif self.init_strategy == 'latin_hypercube':
                sampler = LatinHypercubeSampler(
                    self.config_space,
                    self.init_num,
                    random_state=self.random_state
                )
                return sampler.generate(return_config=True)
        except Exception as e:
            logger.debug(f"Sampler not available: {e}")
        
        # 默认：随机采样
        configs = []
        try:
            default_config = self.config_space.get_default_configuration()
            configs.append(default_config)
        except:
            pass
        
        while len(configs) < self.init_num:
            config = self.config_space.sample_configuration()
            if config not in configs:
                configs.append(config)
        
        return configs
    
    def get_suggestion(self, history: History = None) -> Configuration:
        """
        获取下一个配置建议
        
        Args:
            history: 可选的外部历史记录
            
        Returns:
            建议的配置
        """
        if history is None:
            history = self.history
        
        num_evaluated = len(history)
        
        # 初始阶段
        if num_evaluated < self.init_num:
            if num_evaluated < len(self.initial_configurations):
                return self.initial_configurations[num_evaluated]
            else:
                return self.config_space.sample_configuration()
        
        # 优化阶段
        return self._get_bo_suggestion(history)
    
    def _get_bo_suggestion(self, history: History) -> Configuration:
        """使用贝叶斯优化获取建议"""
        # 准备训练数据
        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        
        # 训练代理模型
        self._train_surrogates(X, Y)
        
        # 训练约束模型
        if self.num_constraints > 0:
            cY = history.get_constraints(transform='bilog')
            self._train_constraint_models(X, cY)
        
        # 更新采集函数
        self._update_acquisition_function(history)
        
        # 优化采集函数
        if self.acq_optimizer is not None and self.acquisition_function is not None:
            try:
                challengers = self.acq_optimizer.maximize(
                    acquisition_function=self.acquisition_function,
                    history=history,
                    num_points=5000
                )
                return challengers[0]
            except Exception as e:
                logger.warning(f"Acq optimization failed: {e}. Using random sampling.")
        
        # Fallback: 随机采样
        return self._random_sample_with_acquisition(history)
    
    def _random_sample_with_acquisition(self, history: History) -> Configuration:
        """使用随机采样 + 采集函数评估"""
        n_candidates = 5000
        candidates = [self.config_space.sample_configuration() for _ in range(n_candidates)]
        X_candidates = np.array([c.get_array() for c in candidates])
        
        # 计算采集函数值
        acq_values = self._compute_acquisition_values(X_candidates, history)
        
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]
    
    def _compute_acquisition_values(self, X: np.ndarray, history: History) -> np.ndarray:
        """计算采集函数值"""
        if self.acq_type == 'parego':
            return self._compute_parego(X, history)
        else:
            return self._compute_ehvi_simple(X, history)
    
    def _compute_parego(self, X: np.ndarray, history: History) -> np.ndarray:
        """计算ParEGO采集函数值"""
        n = X.shape[0]
        weights = self.rng.dirichlet(np.ones(self.num_objectives))
        
        ei_values = np.zeros(n)
        
        Y_current = history.get_objectives(transform='none')
        if len(Y_current) > 0:
            ideal_point = np.min(Y_current, axis=0)
        else:
            ideal_point = np.zeros(self.num_objectives)
        
        for i in range(n):
            means = []
            vars_ = []
            for model in self.surrogate_models:
                m, v = model.predict(X[i:i+1])
                means.append(m.flatten()[0])
                vars_.append(v.flatten()[0])
            
            means = np.array(means)
            vars_ = np.array(vars_)
            
            scalarized_mean = np.max(weights * (means - ideal_point))
            scalarized_var = np.sum((weights ** 2) * vars_)
            scalarized_std = np.sqrt(max(scalarized_var, 1e-10))
            
            if len(Y_current) > 0:
                scalarized_current = np.min([np.max(weights * (y - ideal_point)) for y in Y_current])
            else:
                scalarized_current = float('inf')
            
            ei_values[i] = self._compute_ei(scalarized_mean, scalarized_std, scalarized_current)
        
        return ei_values
    
    def _compute_ehvi_simple(self, X: np.ndarray, history: History) -> np.ndarray:
        """简化版EHVI计算"""
        n = X.shape[0]
        
        # 获取Pareto前沿
        pareto_Y = self._get_pareto_front_objectives(history)
        ref_point = self._get_ref_point(history)
        
        ehvi_values = np.zeros(n)
        
        for i in range(n):
            means = []
            stds = []
            for model in self.surrogate_models:
                m, v = model.predict(X[i:i+1])
                means.append(m.flatten()[0])
                stds.append(np.sqrt(np.maximum(v.flatten()[0], 1e-10)))
            
            means = np.array(means)
            stds = np.array(stds)
            
            ehvi_values[i] = self._mc_ehvi(means, stds, pareto_Y, ref_point)
        
        return ehvi_values
    
    def _mc_ehvi(
        self, 
        means: np.ndarray, 
        stds: np.ndarray,
        pareto_front: np.ndarray,
        ref_point: np.ndarray,
        n_samples: int = 100
    ) -> float:
        """Monte Carlo EHVI估计"""
        samples = np.zeros((n_samples, len(means)))
        for i in range(len(means)):
            samples[:, i] = self.rng.normal(means[i], stds[i], n_samples)
        
        if len(pareto_front) > 0:
            current_hv = self._compute_hypervolume(pareto_front, ref_point)
        else:
            current_hv = 0.0
        
        improvements = []
        for sample in samples:
            if len(pareto_front) > 0:
                new_front = np.vstack([pareto_front, sample])
            else:
                new_front = sample.reshape(1, -1)
            
            pareto_mask = self._get_pareto_mask(new_front)
            new_pareto = new_front[pareto_mask]
            new_hv = self._compute_hypervolume(new_pareto, ref_point)
            improvements.append(max(0, new_hv - current_hv))
        
        return np.mean(improvements)
    
    def _compute_ei(self, mean: float, std: float, current_best: float) -> float:
        """计算Expected Improvement"""
        if std < 1e-10:
            return 0.0
        z = (current_best - mean) / std
        ei = (current_best - mean) * norm.cdf(z) + std * norm.pdf(z)
        return max(0, ei)
    
    def _train_surrogates(self, X: np.ndarray, Y: np.ndarray) -> None:
        """训练代理模型"""
        for i, model in enumerate(self.surrogate_models):
            model.train(X, Y[:, i] if Y.ndim == 2 else Y)
    
    def _train_constraint_models(self, X: np.ndarray, cY: np.ndarray) -> None:
        """训练约束模型"""
        for i, model in enumerate(self.constraint_models):
            model.train(X, cY[:, i])
    
    def _update_acquisition_function(self, history: History) -> None:
        """更新采集函数"""
        if self.acquisition_function is None:
            return
        
        try:
            mo_incumbent_values = history.get_mo_incumbent_values()
            ref_point = self._get_ref_point(history)
            
            if self.acq_type == 'parego':
                self.acquisition_function.update(
                    model=self.surrogate_models[0],
                    eta=mo_incumbent_values[0] if mo_incumbent_values else 0
                )
            else:
                self.acquisition_function.update(
                    model=self.surrogate_models,
                    constraint_models=self.constraint_models if self.num_constraints > 0 else None,
                    eta=mo_incumbent_values,
                    num_data=len(history),
                    ref_point=ref_point
                )
        except Exception as e:
            logger.debug(f"Acq function update warning: {e}")
    
    def _get_pareto_front_objectives(self, history: History) -> np.ndarray:
        """获取Pareto前沿目标值"""
        Y = history.get_objectives(transform='none')
        if len(Y) == 0:
            return np.array([]).reshape(0, self.num_objectives)
        
        pareto_mask = self._get_pareto_mask(Y)
        return Y[pareto_mask]
    
    def _get_pareto_mask(self, Y: np.ndarray) -> np.ndarray:
        """获取Pareto前沿掩码"""
        n = len(Y)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    is_pareto[i] = False
                    break
        
        return is_pareto
    
    def _get_ref_point(self, history: History) -> np.ndarray:
        """获取参考点"""
        if self._ref_point is not None:
            return np.array(self._ref_point)
        
        Y = history.get_objectives(transform='none')
        if len(Y) == 0:
            return np.ones(self.num_objectives) * 10.0
        
        max_vals = np.max(Y, axis=0)
        min_vals = np.min(Y, axis=0)
        margin = 0.1 * (max_vals - min_vals + 1e-6)
        return max_vals + margin
    
    def _compute_hypervolume(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """计算超体积"""
        if len(pareto_front) == 0:
            return 0.0
        
        if self.num_objectives == 2:
            return self._compute_2d_hypervolume(pareto_front, ref_point)
        else:
            return self._compute_nd_hypervolume_approx(pareto_front, ref_point)
    
    def _compute_2d_hypervolume(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """2D超体积计算"""
        valid_points = pareto_front[np.all(pareto_front < ref_point, axis=1)]
        
        if len(valid_points) == 0:
            return 0.0
        
        sorted_indices = np.argsort(valid_points[:, 0])
        sorted_front = valid_points[sorted_indices]
        
        hv = 0.0
        prev_y = ref_point[1]
        
        for point in sorted_front:
            hv += (ref_point[0] - point[0]) * (prev_y - point[1])
            prev_y = point[1]
        
        return hv
    
    def _compute_nd_hypervolume_approx(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """N维超体积近似计算"""
        hv = 0.0
        for point in pareto_front:
            if np.all(point < ref_point):
                contribution = np.prod(ref_point - point)
                hv += contribution
        return hv / max(1, len(pareto_front))
    
    def update_observation(self, observation: Observation) -> None:
        """更新观测结果"""
        self.history.update_observation(observation)
    
    def update_observations(self, observations: List[Observation]) -> None:
        """批量更新观测"""
        for obs in observations:
            self.update_observation(obs)
    
    def get_history(self) -> History:
        """获取历史记录"""
        return self.history
    
    def get_pareto_front(self) -> Tuple[np.ndarray, List[Configuration]]:
        """获取Pareto前沿"""
        try:
            pareto_observations = self.history.get_pareto()
            if len(pareto_observations) == 0:
                return np.array([]).reshape(0, self.num_objectives), []
            
            pareto_Y = np.array([obs.objectives for obs in pareto_observations])
            pareto_configs = [obs.config for obs in pareto_observations]
            return pareto_Y, pareto_configs
        except:
            # Fallback
            Y = self.history.get_objectives(transform='none')
            configs = list(self.history.configurations)
            
            if len(Y) == 0:
                return np.array([]).reshape(0, self.num_objectives), []
            
            pareto_mask = self._get_pareto_mask(Y)
            pareto_Y = Y[pareto_mask]
            pareto_configs = [configs[i] for i in range(len(configs)) if pareto_mask[i]]
            return pareto_Y, pareto_configs
    
    def get_hypervolume(self, ref_point: List[float] = None) -> float:
        """获取当前超体积"""
        try:
            return self.history.compute_hypervolume(ref_point=ref_point or self._ref_point)
        except:
            pareto_Y, _ = self.get_pareto_front()
            ref = np.array(ref_point) if ref_point else self._get_ref_point(self.history)
            return self._compute_hypervolume(pareto_Y, ref)
    
    def print_summary(self) -> None:
        """打印优化摘要"""
        pareto_Y, _ = self.get_pareto_front()
        hv = self.get_hypervolume()
        
        print("=" * 60)
        print("多目标贝叶斯优化摘要")
        print("=" * 60)
        print(f"任务ID: {self.task_id}")
        print(f"目标数量: {self.num_objectives}")
        print(f"约束数量: {self.num_constraints}")
        print(f"总迭代次数: {len(self.history)}")
        print(f"Pareto前沿大小: {len(pareto_Y)}")
        print(f"当前超体积: {hv:.6f}")
        
        if len(pareto_Y) > 0:
            print("\nPareto前沿目标值:")
            for i, y in enumerate(pareto_Y[:5]):
                print(f"  解 {i+1}: {y}")
            if len(pareto_Y) > 5:
                print(f"  ... 共 {len(pareto_Y)} 个Pareto最优解")
        
        print("=" * 60)


def create_multi_objective_advisor(
    config_space: ConfigurationSpace,
    num_objectives: int = 2,
    **kwargs
) -> MultiObjectiveBOAdvisor:
    """工厂函数：创建多目标优化Advisor"""
    return MultiObjectiveBOAdvisor(
        config_space=config_space,
        num_objectives=num_objectives,
        **kwargs
    )