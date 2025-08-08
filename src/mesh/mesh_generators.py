import numpy as np
from typing import List, Tuple, Union
from .mesh import Mesh, Face, Edge
import math

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

def gen_circular_mesh(nnode, r):
    """
    Generates a structured quadrilateral mesh that morphs from a central square
    to an outer circle.

    Args:
        nnode (int): The number of nodes along one edge of the grid.
        r (float): The radius of the outer circular boundary.

    Returns:
        Mesh: An object containing the mesh's points, topology, and facets.
    """
    def _transform_point(X, Y, r, node_r, row, col, sign_num, sign_den):
        """
        Transforms a single point in the mesh.

        This function abstracts the core transformation logic, which interpolates a point's
        position between a circle and a square boundary.

        Args:
            X (np.ndarray): The 2D array of X coordinates.
            Y (np.ndarray): The 2D array of Y coordinates.
            r (float): The outer radius of the circular mesh.
            node_r (float): The radius of the concentric square being processed.
            row (int): The row index of the point to transform.
            col (int): The column index of the point to transform.
            sign_num (int): Sign multiplier for the numerator in the limit calculation (+1 or -1).
            sign_den (int): Sign multiplier for the cosine term in the denominator (+1 or -1).
        """
        x, y = X[row, col], Y[row, col]
        theta = math.atan2(y, x)

        # Calculate the "limit point" (x_l, y_l) on the square boundary
        denominator = math.sin(theta) + sign_den * math.cos(theta)
        # Avoid division by zero, though unlikely for non-corner points
        if abs(denominator) < 1e-9:
            return
            
        limit_radius = (sign_num * node_r) / denominator
        x_l = limit_radius * math.cos(theta)
        y_l = limit_radius * math.sin(theta)

        # The new point is a weighted average of a point on a perfect circle
        # and the calculated limit point on the square boundary.
        weight_circle = (1 + node_r / r) / 2
        weight_limit = (1 - node_r / r) / 2

        x_on_circle = node_r * math.cos(theta)
        y_on_circle = node_r * math.sin(theta)

        new_x = x_on_circle * weight_circle + x_l * weight_limit
        new_y = y_on_circle * weight_circle + y_l * weight_limit

        # Update the mesh arrays in place
        X[row, col] = new_x
        Y[row, col] = new_y
    # 1. Create a standard Cartesian grid
    X, Y = np.meshgrid(np.linspace(-r, r, nnode), np.linspace(-r, r, nnode))

    # 2. Scale and rotate the grid so the diagonals align with the axes.
    # This simplifies processing the grid in concentric square "layers".
    k = 1 / np.sqrt(2)
    X *= k
    Y *= k
    phi = 3 * np.pi / 4
    rot_matix = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), +np.cos(phi)]])
    X, Y = np.einsum('ji, mni -> jmn', rot_matix, np.dstack([X, Y]))

    # 3. Process the grid in concentric square layers, from outside-in
    for i in range(nnode // 2):
        # 'node_r' is the characteristic "radius" of the current square layer
        node_r = X[i, i]
        
        # Iterate over the nodes on the perimeter of the current layer
        for j in range(nnode - 2 * i):
            j_inx = j + i

            # Transform points on each of the four sides of the square
            # (Top, Left, Bottom, Right) by calling the helper function.
            _transform_point(X, Y, r, node_r, row=i,      col=j_inx,  sign_num=+1, sign_den=+1) # Top
            _transform_point(X, Y, r, node_r, row=j_inx,  col=i,      sign_num=-1, sign_den=-1) # Left
            _transform_point(X, Y, r, node_r, row=-i - 1, col=j_inx,  sign_num=-1, sign_den=+1) # Bottom
            _transform_point(X, Y, r, node_r, row=j_inx,  col=-i - 1, sign_num=+1, sign_den=-1) # Right

    # 4. Rotate the mesh back to its original orientation
    X, Y = np.einsum('ji, mni -> jmn', rot_matix.T, np.dstack([X, Y]))

    # 5. Assemble the final mesh data structure
    p = np.vstack((X.flatten(), Y.flatten())).T
    shape = [nnode, nnode]
    nt = (shape[0] - 1) * (shape[1] - 1)  # number of cells

    t = np.zeros([nt, 4], dtype=int)
    point_order = np.arange(shape[0] * shape[1]).reshape((shape[1], shape[0]))
    t[:, 0] = point_order[0:-1, 0:-1].flatten()
    t[:, 1] = point_order[0:-1, 1:].flatten()
    t[:, 2] = point_order[1:, 1:].flatten()
    t[:, 3] = point_order[1:, 0:-1].flatten()

    element = 'quad'
    facet_points = [
        point_order[0, :].tolist(),
        point_order[:, -1].tolist(),
        list(reversed(point_order[-1, :].tolist())),
        list(reversed(point_order[:, 0].tolist())),
    ]
    
    facet = [
        [l[i], l[i + 1]] for l in facet_points for i in range(len(l) - 1)
    ]

    return Mesh(p.tolist(), t.tolist(), element, facet)

def gen_cylinder_mesh(): return

def gen_spherical_mesh(): return

def gen_torus_mesh(): return
