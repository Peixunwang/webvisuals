import numpy as np
from typing import List, Tuple, Union
from .mesh import Mesh

def gen_block_mesh(
        dims: List[List[float]], 
        shape: List[int], 
        ) -> Mesh:
    '''
        dims: [[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]]
        shape: number of nodes in [x, y, z]
    '''
    x = np.linspace(*dims[0], shape[0])
    y = np.linspace(*dims[1], shape[1])
    X, Y = np.meshgrid(x, y)
    p = np.vstack((X.flatten(), Y.flatten())).T
    nt = (shape[0] - 1) * (shape[1] - 1)                # number of cells

    t = np.zeros([nt, 4], dtype=int)
    point_order = np.arange(shape[0] * shape[1]).reshape(tuple(reversed(shape)))
    t[:, 0] = point_order[0:-1, 0:-1].flatten()
    t[:, 1] = point_order[0:-1, 1:].flatten()
    t[:, 2] = point_order[1:, 1:].flatten()
    t[:, 3] = point_order[1:, 0:-1].flatten()
    element = 'quad'
    facet = []
    facet_points = []
    facet_points.append(point_order[0, :].tolist())
    facet_points.append(point_order[:, -1].tolist())
    facet_points.append(list(reversed(point_order[-1, :].tolist())))
    facet_points.append(list(reversed(point_order[:, 0].tolist())))
    for l in facet_points:
        facet.extend([[l[i], l[i+1]] for i in range(len(l)-1)])
    print(facet)


    return Mesh(p.tolist(), t.tolist(), element, facet)
def gen_cylinder_mesh(): pass
def gen_spherical_mesh(): pass
def gen_torus_mesh(): pass