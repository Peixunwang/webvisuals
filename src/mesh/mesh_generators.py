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

def gen_circular_mesh_old(nnode, r):
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

def gen_circular_mesh_squircle(nnode, r, f_square=0.0):
    """
    Generates a high-quality, structured quadrilateral mesh within a circle.

    This function starts with a Cartesian grid and transforms it into a circular
    domain using a low-distortion elliptical grid mapping ("squircle" method).
    This avoids the element skewing common with simpler methods.

    Args:
        nnode (int): The number of nodes along one edge of the grid.
                     Should be an odd number for a center node.
        r (float): The radius of the outer circular boundary.
        f_square (float, optional): Fraction of the radius (from 0.0 to 1.0) 
                                    that defines a central square region which 
                                    resists transformation. Defaults to 0.0 for a
                                    fully circular transformation.

    Returns:
        Mesh: An object containing the mesh's points, topology, and facets.
    """
    # 1. Create a standard Cartesian grid from -r to r
    X, Y = np.meshgrid(np.linspace(-r, r, nnode), np.linspace(-r, r, nnode))

    # 2. Apply the superior square-to-disk mapping
    # This transformation is vectorized and replaces the complex loops.
    
    # Calculate Euclidean distance (hypotenuse) and "square" distance for each point
    r_euclidean = np.hypot(X, Y)  # More stable than sqrt(X**2 + Y**2)
    r_square = np.maximum(np.abs(X), np.abs(Y))

    # Calculate the scaling factor to map the square to a disk
    # Use np.divide to safely handle the origin (0/0 -> 0)
    scale = np.divide(r_square, r_euclidean, out=np.ones_like(r_euclidean), where=(r_euclidean != 0))

    # Apply the base transformation
    X_new = X * scale
    Y_new = Y * scale
    
    # 3. Blend between the original grid and transformed grid for a central square
    if f_square > 0.0:
        # Define the radius of the inner square region
        r_inner_square = r * f_square
        # Calculate a blend factor `alpha` that is 0 inside the square and 1 outside
        # It smoothly transitions from 0 to 1 between the inner square and the outer boundary
        alpha = np.clip((r_square - r_inner_square) / (r - r_inner_square), 0, 1)
        
        # Interpolate between the original grid (X,Y) and the new circular grid (X_new, Y_new)
        X = X * (1 - alpha) + X_new * alpha
        Y = Y * (1 - alpha) + Y_new * alpha
    else:
        # If no central square is needed, use the transformed grid directly
        X, Y = X_new, Y_new

    # 4. Assemble the final mesh data structure
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

def solve_elliptic_grid_generator(X, Y, tol=1e-5, max_iter=10000, ortho_weight=2.0):
    """
    Solves the elliptic grid generation equations to create a smooth, orthogonal grid.

    This function uses the Thompson-Thames-Mastin (TTM) system to iteratively
    adjust the interior grid points, balancing smoothness and orthogonality.

    Args:
        X (np.ndarray): Initial X-coordinate grid with boundary conditions set.
        Y (np.ndarray): Initial Y-coordinate grid with boundary conditions set.
        tol (float): Convergence tolerance for the maximum change in coordinates.
        max_iter (int): Maximum number of iterations.
        ortho_weight (float): Weight for the orthogonality term (0.0 to 1.0).
                              0.0 is equivalent to the original Laplace solver.
                              1.0 enforces strong orthogonality.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The smoothed and orthogonalized X and Y grids.
    """
    X_new, Y_new = X.copy(), Y.copy()
    
    # A small epsilon to prevent division by zero in the denominator
    epsilon = 1e-12

    for k in range(max_iter):
        X_old, Y_old = X_new.copy(), Y_new.copy()

        # Calculate first derivatives using central differences for interior points
        # Derivatives with respect to xi (along columns, index j)
        X_xi = (X_old[1:-1, 2:] - X_old[1:-1, 0:-2]) / 2.0
        Y_xi = (Y_old[1:-1, 2:] - Y_old[1:-1, 0:-2]) / 2.0
        
        # Derivatives with respect to eta (along rows, index i)
        X_eta = (X_old[2:, 1:-1] - X_old[0:-2, 1:-1]) / 2.0
        Y_eta = (Y_old[2:, 1:-1] - Y_old[0:-2, 1:-1]) / 2.0

        # Calculate the coefficients alpha, beta, gamma
        alpha = X_eta**2 + Y_eta**2
        beta  = X_xi * X_eta + Y_xi * Y_eta
        gamma = X_xi**2 + Y_xi**2

        # The term multiplying the mixed derivative (X_xi_eta)
        # We apply the ortho_weight here to control orthogonality
        C = ortho_weight * beta

        # Calculate the mixed derivatives
        X_xi_eta = (X_old[2:, 2:] - X_old[0:-2, 2:] - X_old[2:, 0:-2] + X_old[0:-2, 0:-2]) / 4.0
        Y_xi_eta = (Y_old[2:, 2:] - Y_old[0:-2, 2:] - Y_old[2:, 0:-2] + Y_old[0:-2, 0:-2]) / 4.0

        # Jacobi iteration update formula for the TTM equations
        # This formula is derived by discretizing the PDEs and solving for the center point (i,j)
        denom = 2 * (alpha + gamma + epsilon)
        
        term1_X = alpha * (X_old[2:, 1:-1] + X_old[0:-2, 1:-1])
        term2_X = gamma * (X_old[1:-1, 2:] + X_old[1:-1, 0:-2])
        term3_X = C * 2 * X_xi_eta # Note: the 2*beta term from the PDE
        X_new[1:-1, 1:-1] = (term1_X + term2_X - term3_X) / denom

        term1_Y = alpha * (Y_old[2:, 1:-1] + Y_old[0:-2, 1:-1])
        term2_Y = gamma * (Y_old[1:-1, 2:] + Y_old[1:-1, 0:-2])
        term3_Y = C * 2 * Y_xi_eta
        Y_new[1:-1, 1:-1] = (term1_Y + term2_Y - term3_Y) / denom


        # Check for convergence
        error_x = np.abs(X_new - X_old).max()
        error_y = np.abs(Y_new - Y_old).max()

        if error_x < tol and error_y < tol:
            print(f"Converged after {k+1} iterations.")
            return X_new, Y_new

    print(f"Warning: Did not converge after {max_iter} iterations.")
    return X_new, Y_new

# Keep the original function for comparison
def _solve_laplace_for_grid(X, Y, tol=1e-5, max_iter=10000):
    n = X.shape[0]
    for i in range(max_iter):
        X_old, Y_old = X.copy(), Y.copy()
        X[1:-1, 1:-1] = 0.25 * (X_old[0:-2, 1:-1] + X_old[2:, 1:-1] + X_old[1:-1, 0:-2] + X_old[1:-1, 2:])
        Y[1:-1, 1:-1] = 0.25 * (Y_old[0:-2, 1:-1] + Y_old[2:, 1:-1] + Y_old[1:-1, 0:-2] + Y_old[1:-1, 2:])
        error_x = np.abs(X - X_old).max()
        error_y = np.abs(Y - Y_old).max()
        if error_x < tol and error_y < tol:
            print(f"Laplace converged after {i+1} iterations.")
            return X, Y
    print(f"Laplace Warning: Did not converge after {max_iter} iterations.")
    return X, Y

def gen_circular_mesh(nnode, r):
    """
    Generates a high-quality, orthogonal-style quadrilateral mesh within a circle
    by solving Laplace's equation for the node positions.

    Args:
        nnode (int): The number of nodes along one edge. Must be odd.
        r (float): The radius of the outer circular boundary.

    Returns:
        Mesh: An object containing the mesh's points, topology, and facets.
    """
    if nnode % 2 == 0:
        raise ValueError("nnode must be an odd number for this method.")
    """
    Generates a high-quality, structured quadrilateral mesh within a circle.

    This function starts with a Cartesian grid and transforms it into a circular
    domain using a low-distortion elliptical grid mapping ("squircle" method).
    This avoids the element skewing common with simpler methods.

    Args:
        nnode (int): The number of nodes along one edge of the grid.
                     Should be an odd number for a center node.
        r (float): The radius of the outer circular boundary.
        f_square (float, optional): Fraction of the radius (from 0.0 to 1.0) 
                                    that defines a central square region which 
                                    resists transformation. Defaults to 0.0 for a
                                    fully circular transformation.

    Returns:
        Mesh: An object containing the mesh's points, topology, and facets.
    """
    # 1. Create a standard Cartesian grid from -r to r
    X, Y = np.meshgrid(np.linspace(-r, r, nnode), np.linspace(-r, r, nnode))

    # 2. Apply the superior square-to-disk mapping
    # This transformation is vectorized and replaces the complex loops.
    
    # Calculate Euclidean distance (hypotenuse) and "square" distance for each point
    r_euclidean = np.hypot(X, Y)  # More stable than sqrt(X**2 + Y**2)
    r_square = np.maximum(np.abs(X), np.abs(Y))

    # Calculate the scaling factor to map the square to a disk
    # Use np.divide to safely handle the origin (0/0 -> 0)
    scale = np.divide(r_square, r_euclidean, out=np.ones_like(r_euclidean), where=(r_euclidean != 0))

    # Apply the base transformation
    X_new = X * scale
    Y_new = Y * scale
    
    X, Y = X_new, Y_new

    # 3. Solve for the interior node positions using the PDE solver
    X, Y = solve_elliptic_grid_generator(X, Y)

    # 4. Assemble the final mesh data structure
    p = np.vstack((X.flatten(), Y.flatten())).T
    shape = [nnode, nnode]
    nt = (shape[0] - 1) * (shape[1] - 1)

    t = np.zeros([nt, 4], dtype=int)
    point_order = np.arange(nnode * nnode).reshape((nnode, nnode))
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
