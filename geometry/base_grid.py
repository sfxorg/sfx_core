import abc
from sfx_core.config import GridConfig

class BaseGrid(abc.ABC):
    """Abstract Base Class defining the geometry engine contract."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.setup_grid()

    @abc.abstractmethod
    def setup_grid(self):
        """Generates spatial coordinates and geometry-specific metrics."""
        pass

    @abc.abstractmethod
    def compute_metric_tensor(self):
        """Computes inverse Jacobians and scale transformations."""
        pass
