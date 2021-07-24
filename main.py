import numpy as np
import jax
from jax import pmap
out = pmap(lambda x: x ** 2)(np.arange(8))
print(out)
