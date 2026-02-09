from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from openbox.manager.task_manager import TaskManager
from openbox.optimizer.generic_smbo import SMBO


class DummyTaskManager:
    def __init__(self):
        self.calls = []

    def update_current_task_history(self, config, results):
        self.calls.append((config, results))


def build_config_space():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter('x', 0.0, 1.0))
    return config_space


def objective_function(config):
    return {'objectives': [float(config['x'])]}


if __name__ == '__main__':
    TaskManager._instance = DummyTaskManager()
    optimizer = SMBO(
        objective_function=objective_function,
        config_space=build_config_space(),
        advisor_type='random',
        max_runs=1,
        initial_runs=1,
        logging_dir='logs',
        task_id='legacy-taskmanager-sync-demo',
    )
    optimizer.iterate()
    print(f"TaskManager synced calls: {len(TaskManager._instance.calls)}")
