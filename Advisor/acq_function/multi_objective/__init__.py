"""
多目标采集函数模块
"""

from .base import MultiObjectiveAcquisition
from .ehvi import EHVI

__all__ = [
    'MultiObjectiveAcquisition',
    'EHVI',
]