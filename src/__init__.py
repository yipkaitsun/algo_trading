"""
Algorithmic Trading Package
"""

__version__ = "0.1.0"

from .utils.data_loader import DataLoader
from .strategies.strategy_example import ExampleStrategy

__all__ = ['DataLoader', 'ExampleStrategy']
