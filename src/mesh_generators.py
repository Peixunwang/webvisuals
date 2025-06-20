import numpy as np
from mesh import Mesh
from typing import List

def gen_block_mesh(
        dims: List[List[float]], 
        shape: List[int], 
        ) -> Mesh:
    '''
        dims: [[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]]
        shape: number of nodes in [x, y, z]
    '''
    pass