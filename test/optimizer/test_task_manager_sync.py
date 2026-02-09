import pytest
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from openbox.manager.task_manager import TaskManager
from openbox.optimizer.generic_smbo import SMBO


class DummyTaskManager:
    def __init__(self):
        self.calls = []

    def update_current_task_history(self, config, results):
        self.calls.append((config, results))


def _build_config_space():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter('x', 0.0, 1.0))
    return config_space


def test_smbo_syncs_task_manager_history(monkeypatch, tmp_path):
    dummy = DummyTaskManager()
    monkeypatch.setattr(TaskManager, "_instance", dummy)
    config_space = _build_config_space()

    def objective_function(config):
        return {'objectives': [float(config['x'])]}

    optimizer = SMBO(
        objective_function=objective_function,
        config_space=config_space,
        advisor_type='random',
        max_runs=1,
        initial_runs=1,
        logging_dir=str(tmp_path),
    )
    optimizer.iterate()

    assert len(dummy.calls) == 1
    config, results = dummy.calls[0]
    assert results['result']['objective'] == pytest.approx(float(config['x']))


def test_smbo_runs_without_task_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(TaskManager, "_instance", None)
    config_space = _build_config_space()

    def objective_function(config):
        return {'objectives': [float(config['x'])]}

    optimizer = SMBO(
        objective_function=objective_function,
        config_space=config_space,
        advisor_type='random',
        max_runs=1,
        initial_runs=1,
        logging_dir=str(tmp_path),
    )
    optimizer.iterate()
