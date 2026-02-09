import numpy as np
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any, Callable
from ConfigSpace import ConfigurationSpace

from Advisor.utils import map_source_hpo_data, build_observation
from .config_manager import ConfigManager
from .history_manager import HistoryManager
from .component_registry import ComponentRegistry
from core.interfaces import TargetSystem

from openbox import logger
from openbox.utils.history import History


class TaskManager:    
    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, 
                config_space: ConfigurationSpace,
                config_manager: Optional[ConfigManager] = None,
                config_dict: Optional[Dict[str, Any]] = None,
                logger_kwargs: Optional[Dict[str, Any]] = None,
                target_system: Optional[TargetSystem] = None,
                component_registry: Optional[ComponentRegistry] = None,
                history_sync_callbacks: Optional[List[Callable[[str, Dict[str, Any]], None]]] = None,
                **kwargs):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self._config_manager = config_manager
        if config_manager is not None:
            self._config = deepcopy(config_manager.config)
        elif config_dict is not None:
            self._config = deepcopy(config_dict)
        else:
            raise ValueError('TaskManager requires config_manager or config_dict.')

        # Keep legacy helper methods available even when caller does not instantiate ConfigManager.
        self._config_accessor = config_manager or ConfigManager.from_dict(self._config)

        method_args = self._config.get('method_args', {})
        self.ws_args = method_args.get('ws_args')
        self.tl_args = method_args.get('tl_args')
        self.scheduler_kwargs = method_args.get('scheduler_kwargs') or {}
        self.logger_kwargs = logger_kwargs or {}
        self.random_kwargs = method_args.get('random_kwargs') or {}
        self.config_space = config_space
        self.target_system = target_system
        self._history_sync_callbacks = history_sync_callbacks or []

        paths = self._config.get('paths', {})
        root_dir = getattr(self._config_accessor, 'root_dir', '')
        history_dir = paths.get('history_dir', '')
        if root_dir and history_dir and not history_dir.startswith('/'):
            history_dir = f'{root_dir}/{history_dir}'

        similarity_threshold = self._config.get('similarity_threshold', 0.0)
        current_database = self._config.get('database')
        
        self.history_manager = HistoryManager(
            config_space=config_space,
            history_dir=history_dir,
            similarity_threshold=similarity_threshold,
            current_database=current_database
        )
        
        self.component_registry = component_registry or ComponentRegistry()
        
        self._setup_listeners()
        
        logger.info("TaskManager initialized with modular architecture")
    
    def _setup_listeners(self):
        def mark_plan_dirty(component):
            if self.target_system:
                self.target_system.on_component_update('scheduler', component)
        
        self.component_registry.add_listener('scheduler', mark_plan_dirty)
        # self.component_registry.add_listener('sql_partitioner', mark_plan_dirty) # Handled by target_system if needed
    

    def calculate_meta_feature(self, eval_func: Callable, task_id: str = "default", **kwargs):
        # skip meta_feature collecting and default config evaluation
        if kwargs.get('resume', None) is not None:
            self.history_manager.resume_current_task(kwargs.get('resume'))
            self._update_similarity()
            return
        
        default_config = self.config_space.get_default_configuration()
        default_config.origin = 'Default Configuration'
        result = eval_func(config=default_config, resource_ratio=1.0)
        
        if kwargs.get('test_mode', False):
            logger.info("Using test mode meta feature")
            meta_feature = np.random.rand(38)  # 34 base + 4 additional (CPU, Memory, Nodes, DB size)
            self.history_manager.initialize_current_task(task_id, meta_feature)
            self.history_manager.update_current_history(build_observation(default_config, result))
            self._update_similarity()
            return
        
        logger.info("Computing current task meta feature using target system...")

        if self.target_system:
            meta_feature = self.target_system.get_meta_feature(task_id, test_mode=kwargs.get('test_mode', False))
        else:
            logger.warning("No target system configured, using random meta feature")
            meta_feature = np.random.rand(38)  # 34 base + 4 additional (CPU, Memory, Nodes, DB size)
        
        self.history_manager.initialize_current_task(task_id, meta_feature)
        self.history_manager.update_current_history(build_observation(default_config, result))
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")
        
        self._update_similarity()
    
    def _update_similarity(self):
        ws_args_with_cache = (self.ws_args or {}).copy()
        ws_args_with_cache['use_cached_model'] = self._config.get('use_cached_model', False)
        if self._config_manager is not None:
            ws_args_with_cache['use_cached_model'] = self._config_manager.use_cached_model
        self.history_manager.compute_similarity(
            similarity_func=map_source_hpo_data,
            **ws_args_with_cache
        )
        self._mark_sql_plan_dirty()

    def _notify_history_sync(self, event: str, payload: Dict[str, Any]) -> None:
        for callback in self._history_sync_callbacks:
            try:
                callback(event, payload)
            except Exception as e:
                logger.warning(f'History sync callback failed on event={event}: {e}')
    
    def _mark_sql_plan_dirty(self):
        if self.target_system:
            self.target_system.on_component_update('scheduler', None)
    
    
    @property
    def current_task_history(self) -> Optional[History]:
        return self.history_manager.get_current_history()
    
    def update_current_task_history(self, config, results):
        obs = build_observation(config, results)
        self.history_manager.update_current_history(obs)
        self._notify_history_sync('update_current_task_history', {
            'config': config,
            'results': results,
            'observation': obs,
        })
        self._update_similarity()
    
    def update_history_meta_info(self, meta_info: dict):
        self.history_manager.update_history_meta_info(meta_info)
        self._notify_history_sync('update_history_meta_info', {
            'meta_info': meta_info,
            'history': self.history_manager.get_current_history(),
        })
    
    def get_similar_tasks(
        self,
        topk: Optional[int] = None,
        filter_by_sql_type: bool = False
    ) -> Tuple[List[History], List[Tuple[int, float]]]:
        if topk is None:
            topk = self.tl_args.get('topk') if self.tl_args else None
        return self.history_manager.get_similar_tasks(topk, filter_by_sql_type=filter_by_sql_type)
    
    def get_current_task_history(self) -> Optional[History]:
        return self.history_manager.get_current_history()
    
    
    def register_scheduler(self, scheduler):
        self.component_registry.register('scheduler', scheduler, replace=False)
    
    def get_scheduler(self) -> Optional[object]:
        return self.component_registry.get('scheduler')
    
    def register_sql_partitioner(self, partitioner) -> None:
        self.component_registry.register('sql_partitioner', partitioner, replace=True)
    
    def get_sql_partitioner(self):
        return self.component_registry.get('sql_partitioner')
    
    def register_planner(self, planner) -> None:
        self.component_registry.register('planner', planner, replace=True)
    
    def get_planner(self):
        return self.component_registry.get('planner')
    
    def register_compressor(self, compressor) -> None:
        self.component_registry.register('compressor', compressor, replace=True)
    
    def get_compressor(self):
        return self.component_registry.get('compressor')
    
    
    def get_cp_string(self, config_space) -> str:
        return self._config_accessor.get_cp_string(config_space)
    
    def generate_task_id(self, task_name: str, method_id: str, ws_strategy: str,
                        tl_strategy: str, scheduler_type: str, config_space,
                        rand_mode: str = 'ran', seed: int = 42) -> str:
        return self._config_accessor.generate_task_id(
            task_name, method_id, ws_strategy, tl_strategy, 
            scheduler_type, config_space, rand_mode, seed
        )
    
    def get_ws_args(self) -> Dict[str, Any]:
        return dict(self.ws_args or {})
    
    def get_tl_args(self) -> Dict[str, Any]:
        return dict(self.tl_args or {})
    
    def get_cp_args(self, config_space=None) -> Dict[str, Any]:
        if config_space is None:
            config_space = self.config_space
        return self._config_accessor.get_cp_args(config_space)
    
    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        return dict(self.scheduler_kwargs)
    
    def get_logger_kwargs(self) -> Dict[str, Any]:
        return dict(self.logger_kwargs)
    
    def get_random_kwargs(self) -> Dict[str, Any]:
        return dict(self.random_kwargs)
