from typing import List
import numpy as np
# from src.mesh_generators import gen_block_mesh
# is git working?
# mesh = gen_block_mesh([[0, 1], [0, 2]], [3, 4])
# print(mesh)
from src.mesh.mesh import Mesh
from src.mesh.mesh_generators import gen_block_mesh

mesh_2d = Mesh([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
                        [0.0, 1.5], [0.5, 1.5], [1.0, 1.5],
                        [0.0, 2.0], [0.5, 2.0], [1.0, 2.0]],
                       [[0, 1, 4, 3],
                        [1, 2, 5, 4], 
                        [3, 4, 7, 6],
                        [4, 5, 8, 7], 
                        [6, 7, 10, 9],
                        [7, 8, 11, 10]],
                        'quad',
                        [[0, 1], [1, 2], 
                         [2, 5], [5, 8], [8, 11], 
                         [11, 10], [10, 9], 
                         [9, 6], [6, 3], [3, 0]])

mesh_3d = gen_block_mesh([[0, 1], [0, 2], [0, 0.5]], (3, 3, 3))
# idx = filter(lambda x: x[1][2] <= 0, enumerate(mesh_3d.p))
# indices = [index for index, value in idx]
indices = mesh_3d.nodes_satisfy(lambda u: u[2] <= 0)
facet = mesh_3d.facet_from_nodes(indices)
from src.mesh.mesh_helpers import face_to_meshgrid
print(face_to_meshgrid(mesh_3d, facet))
