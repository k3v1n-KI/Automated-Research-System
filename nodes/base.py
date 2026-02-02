"""
Base class for research pipeline nodes.
Provides common interface and utilities.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


class BaseNode(ABC):
    """Abstract base class for all research nodes"""
    
    def __init__(self):
        self.node_name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Execute the node's logic. Must be implemented by subclasses."""
        pass
