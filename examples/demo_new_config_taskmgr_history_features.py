"""Demo for newly added config/task-manager/history features.

Run locally (recommended):

    python examples/demo_new_config_taskmgr_history_features.py

This demo validates:
1) ConfigManager static utilities for direct dict workflows.
2) TaskManager initialization from config_dict (without caller-side ConfigManager instance).
3) External ComponentRegistry injection.
4) TaskManager history sync callbacks.
5) Optimizer legacy history + optional TaskManager history sync together.
6) Compact history JSON save/load.
"""

import json
import os
import tempfile
from argparse import Namespace

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from openbox.manager.component_registry import ComponentRegistry
from openbox.manager.config_manager import ConfigManager
from openbox.manager.task_manager import TaskManager
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.history import History


def build_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('x', 0.0, 1.0, default_value=0.5))
    return cs


def build_base_config(tmp_dir):
    return {
        'paths': {
            'history_dir': os.path.join(tmp_dir, 'history'),
            'log_dir': os.path.join(tmp_dir, 'logs'),
            'data_dir': os.path.join(tmp_dir, 'data'),
            'save_dir': os.path.join(tmp_dir, 'save'),
            'target': 'demo',
        },
        'method_args': {
            'ws_args': {'init_num': 1},
            'tl_args': {'topk': 1},
            'cp_args': {'strategy': 'none'},
            'scheduler_kwargs': {},
            'random_kwargs': {'seed': 0},
            'logger_kwargs': {},
        },
        'config_spaces': {
            'config_space': 'unused.json',
            'expert_space': 'unused.json',
        },
        'similarity_threshold': 0.0,
        'database': 'demo_db',
        'target_system': 'spark',
    }


def objective(config):
    x = float(config['x'])
    return {'objectives': [x * x], 'extra_info': {'x_value': x}}


def main():
    with tempfile.TemporaryDirectory(prefix='openbox_feature_demo_') as tmp_dir:
        # --------- 1) ConfigManager static utilities on plain dict ---------
        config = build_base_config(tmp_dir)

        # Merge patch config without instantiation.
        config = ConfigManager.merge_config(config, {'method_args': {'ws_args': {'topk': 3}}})

        # Set a nested value by dotted path.
        ConfigManager.set_config_value(config, 'method_args.tl_args.topk', 2)

        # Apply CLI-like overrides from Namespace.
        fake_args = Namespace(
            config='unused',
            opt='MFES_SMAC',
            task='demo_task',
            log_level='info',
            iter_num=2,
            warm_start='none',
            transfer='none',
            backup_flag=False,
            test_mode=True,
            debug=False,
            resume=None,
            use_cached_model=False,
            ws_topk=4,
            tl_topk=5,
            cp_topk=6,
            compress='none',
        )
        ConfigManager.apply_args_overrides(config, fake_args)

        assert config['method_args']['ws_args']['topk'] == 4
        assert config['method_args']['tl_args']['topk'] == 5

        # --------- 2) TaskManager from config_dict + external registry ---------
        TaskManager._instance = None
        callback_events = []

        def on_history_event(event_name, payload):
            callback_events.append((event_name, payload))

        registry = ComponentRegistry()
        cs = build_config_space()
        tm = TaskManager.instance(
            config_space=cs,
            config_dict=config,
            logger_kwargs={'name': 'demo_tm'},
            component_registry=registry,
            history_sync_callbacks=[on_history_event],
        )

        assert tm.component_registry is registry

        # --------- 3) Run optimizer and verify dual history sync ---------
        tm.history_manager.initialize_current_task(task_id='demo_task')

        optimizer = SMBO(
            objective_function=objective,
            config_space=cs,
            advisor_type='random',
            max_runs=2,
            initial_runs=1,
            logging_dir=os.path.join(tmp_dir, 'opt_logs'),
            task_id='demo_task',
        )
        optimizer.iterate()

        # Original/legacy advisor history is still required and updated.
        advisor_history_len = len(optimizer.get_history())
        assert advisor_history_len == 1

        # TaskManager is optional; when present, it is also updated.
        tm_history = tm.get_current_task_history()
        assert tm_history is not None and len(tm_history) >= 1
        assert any(name == 'update_current_task_history' for name, _ in callback_events)

        # --------- 4) Compact history JSON ---------
        compact_path = os.path.join(tmp_dir, 'history_compact.json')
        tm_history.save_json(compact_path, compact=True)

        with open(compact_path, 'r', encoding='utf-8') as f:
            compact_data = json.load(f)

        # Optional keys are omitted when not needed.
        assert 'meta_info' not in compact_data or compact_data['meta_info']
        if compact_data['observations']:
            first_obs = compact_data['observations'][0]
            assert 'constraints' not in first_obs

        reloaded = History.load_json(compact_path, config_space=cs)
        assert len(reloaded) == len(tm_history)

        print('Demo passed ✅')
        print(f'  advisor history len: {advisor_history_len}')
        print(f'  task manager history len: {len(tm_history)}')
        print(f'  callback events: {[name for name, _ in callback_events]}')
        print(f'  compact history file: {compact_path}')


if __name__ == '__main__':
    main()
