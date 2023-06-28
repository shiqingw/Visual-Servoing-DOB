import numpy as np

class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # dtype for computations
        self.np_dtype = np.float32
        self.dpi = 100
        self.figsize = (10,10)

