import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Protocol, List, Sequence
from dataclasses import dataclass
from itertools import product

class HistoryLike(Protocol):
    @property
    def observations(self) -> List:
        ...
    
    def get_config_array(self, transform: Optional[str] = None) -> np.ndarray:
        ...
    
    def get_objectives(self, transform: Optional[str] = None) -> np.ndarray:
        ...

    def get_incumbent_value(self) -> float:
        ...

class SurrogateModel(Protocol):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


class AcquisitionFunction(ABC):
    def __init__(self, model: SurrogateModel, **kwargs):
        self.model = model
        self.long_name = self.__class__.__name__
    
    @abstractmethod
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def __call__(self, X: np.ndarray, convert: bool = True, **kwargs) -> np.ndarray:
        return self._compute(X, **kwargs)
    
    def update(self, context=None, **kwargs) -> None:
        if context is not None:
            self.model = context.get_main_surrogate()
            self._update_from_context(context, **kwargs)
            return
            
    def _update_from_context(self, context, **kwargs) -> None:
        pass


class SingleObjectiveAcquisition(AcquisitionFunction):
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.eta = None
    
    def update(self, context=None, **kwargs) -> None:
        super().update(context=context, **kwargs)
        
        if context is not None:
            target_task = context.get_target_task()
            self.eta = target_task.get_incumbent_value()
            return


class TransferLearningAcquisition(AcquisitionFunction):    
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.source_acq_funcs = []
        self.target_acq_func = None
        self.weights = None
    
    @abstractmethod
    def _combine_acquisitions(self, source_values: np.ndarray, 
                            target_values: np.ndarray) -> np.ndarray:
        pass


class MultiObjectiveAcquisition(AcquisitionFunction):
    def __init__(self, model: Sequence[SurrogateModel], ref_point: Sequence[float], **kwargs):
        self.model = list(model)
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.pareto_front: Optional[np.ndarray] = None
        self.cell_lower_bounds: Optional[np.ndarray] = None
        self.cell_upper_bounds: Optional[np.ndarray] = None
        super().__init__(model=self.model[0] if self.model else None, **kwargs)

    def update(self, context=None, **kwargs) -> None:
        if context is not None:
            target_task = context.get_target_task()
            objectives = target_task.history.get_objectives()
            if objectives.ndim == 1:
                objectives = objectives.reshape(-1, 1)
            self.pareto_front = self._get_non_dominated(objectives)
            self.cell_lower_bounds, self.cell_upper_bounds = self._cell_bounds_from_pareto(
                self.pareto_front,
                self.ref_point
            )

    @staticmethod
    def _get_non_dominated(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        num_points = points.shape[0]
        is_dominated = np.zeros(num_points, dtype=bool)
        for i in range(num_points):
            if is_dominated[i]:
                continue
            for j in range(num_points):
                if i == j:
                    continue
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    is_dominated[i] = True
                    break
        return points[~is_dominated]

    @staticmethod
    def _cell_bounds_from_pareto(
        pareto_front: np.ndarray,
        ref_point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if pareto_front.size == 0:
            return np.empty((0, ref_point.shape[0])), np.empty((0, ref_point.shape[0]))

        if not np.all(ref_point >= pareto_front):
            raise ValueError("Reference point must be dominated by the Pareto front for minimization.")

        bounds_per_obj = []
        for idx in range(ref_point.shape[0]):
            values = np.unique(np.concatenate([pareto_front[:, idx], [ref_point[idx]]]))
            values.sort()
            bounds_per_obj.append(values)

        index_ranges = [range(len(values) - 1) for values in bounds_per_obj]
        lower_bounds = []
        upper_bounds = []
        for indices in product(*index_ranges):
            lower = np.array([bounds_per_obj[dim][i] for dim, i in enumerate(indices)])
            upper = np.array([bounds_per_obj[dim][i + 1] for dim, i in enumerate(indices)])
            dominated = np.any(np.all(pareto_front <= lower, axis=1))
            if dominated:
                lower_bounds.append(lower)
                upper_bounds.append(upper)

        return np.asarray(lower_bounds), np.asarray(upper_bounds)


@dataclass
class TaskContext:
    surrogate: SurrogateModel
    history: HistoryLike
    eta: float
    num_data: int
    
    def get_incumbent_value(self):
        return self.eta


@dataclass
class AcquisitionContext:
    tasks: List[TaskContext]
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.array([1.0])
        else:
            self.weights = np.array(self.weights)
        
        if len(self.weights) != len(self.tasks):
            raise ValueError(
                f"Weights length ({len(self.weights)}) must match tasks length ({len(self.tasks)})"
            )
        
        self._main_surrogate: Optional[SurrogateModel] = None
    
    def is_multi_task(self) -> bool:
        return len(self.tasks) > 1
    
    def get_target_task(self) -> TaskContext:
        return self.tasks[-1]
    
    def get_source_tasks(self) -> List[TaskContext]:
        return self.tasks[:-1] if self.is_multi_task() else []
    
    def get_main_surrogate(self):
        if self._main_surrogate is not None:
            return self._main_surrogate
        return self.get_target_task().surrogate

    def set_main_surrogate(self, surrogate: SurrogateModel) -> None:
        self._main_surrogate = surrogate