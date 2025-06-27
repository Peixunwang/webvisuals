import numpy as np
from typing import List, Tuple, Union
from .mesh import Mesh, Face, Edge

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
    nt = (shape[0] - 1) * (shape[1] - 1)                    # number of cells

    t = np.zeros([nt, 4], dtype=int)
    point_order = np.arange(shape[0] * shape[1]).reshape((shape[1], shape[0]))
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

    if len(shape) == 2:
        return Mesh(p.tolist(), t.tolist(), element, facet)
    # generate 3D mesh

    z = np.linspace(*dims[2], shape[2])
    p3d = np.zeros((shape[0] * shape[1] * shape[2], 3))
    for i in range(shape[2]):
        p3d[i*X.size : (i+1)*X.size, :2] = p
        p3d[i*X.size : (i+1)*X.size, 2] = z[i]
    nt = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1)   # number of cells
    t = np.zeros([nt, 8], dtype=int)
    point_order = np.arange(shape[0] * shape[1] * shape[2]).reshape((shape[2], shape[1], shape[0]))
    t[:, 0] = point_order[0:-1, 0:-1, 0:-1].flatten()
    t[:, 1] = point_order[0:-1, 0:-1, 1:].flatten()
    t[:, 2] = point_order[0:-1, 1:, 1:].flatten()
    t[:, 3] = point_order[0:-1, 1:, 0:-1].flatten()
    t[:, 4] = point_order[1:, 0:-1, 0:-1].flatten()
    t[:, 5] = point_order[1:, 0:-1, 1:].flatten()
    t[:, 6] = point_order[1:, 1:, 1:].flatten()
    t[:, 7] = point_order[1:, 1:, 0:-1].flatten()
    element = 'hexahedron'
    mesh = Mesh(p3d.tolist(), t.tolist(), element)
    boundaries = {}
    boundaries.update({'left' : mesh.nodes_satisfy(lambda u: u[0] <= dims[0][0])})
    boundaries.update({'right' : mesh.nodes_satisfy(lambda u: u[0] >= dims[0][1])})
    boundaries.update({'front' : mesh.nodes_satisfy(lambda u: u[1] <= dims[1][0])})
    boundaries.update({'back' : mesh.nodes_satisfy(lambda u: u[1] >= dims[1][1])})
    boundaries.update({'bottom' : mesh.nodes_satisfy(lambda u: u[2] <= dims[2][0])})
    boundaries.update({'top' : mesh.nodes_satisfy(lambda u: u[2] >= dims[2][1])})
    mesh.boundaries = boundaries

    return mesh

def gen_cylinder_mesh(): return
def gen_spherical_mesh(): return
def gen_torus_mesh(): return