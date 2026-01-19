multique_fidelity_spark-new_ref/
├── Advisor/
│   ├── acq_function/
│   │   ├── multi_objective/           # 新增目录
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # 多目标采集函数基类
│   │   │   ├── ehvi.py               # EHVI实现
│   │   │   ├── parego.py             # ParEGO实现（可选）
│   │   │   └── mesmo.py              # MESMO实现（可选）
│   ├── MOBO.py                        # 新增：多目标贝叶斯优化Advisor
├── utils/
│   ├── pareto.py                      # 新增：Pareto工具函数
│   ├── hypervolume.py                 # 新增：超体积计算
├── manager/
│   ├── multi_objective_history.py     # 新增：多目标历史管理