import numpy as np
from typing import List, Tuple, Union
from .mesh import Mesh, Face, Edge
import math

def gen_tensor_topology(shape: List[int]):
    nt = (shape[0] - 1) * (shape[1] - 1)                    # number of cells
    t = np.zeros([nt, 4], dtype=int)
    point_order = np.arange(shape[0] * shape[1]).reshape((shape[1], shape[0]))
    t[:, 0] = point_order[0:-1, 0:-1].flatten()
    t[:, 1] = point_order[0:-1, 1:].flatten()
    t[:, 2] = point_order[1:, 1:].flatten()
    t[:, 3] = point_order[1:, 0:-1].flatten()
    if len(shape) == 2:
        return t
    
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
    return t

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
    t = gen_tensor_topology(shape)
    element = 'quad'
    nt = (shape[0] - 1) * (shape[1] - 1)                    # number of cells
    point_order = np.arange(shape[0] * shape[1]).reshape((shape[1], shape[0]))
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
    t = gen_tensor_topology(shape)
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

def solve_elliptic_grid_generator3D(X, Y, Z, tol=1e-5, max_iter=2000, ortho_weight=0.0):
    """
    Solves the 3D elliptic grid generation equations for a smooth, orthogonal grid.

    This function uses a 3D extension of the Thompson-Thames-Mastin (TTM) system
    to iteratively adjust the interior grid points, balancing grid smoothness
    and orthogonality. Boundary points are assumed to be fixed.

    Args:
        X (np.ndarray): 3D array of initial X-coordinates.
        Y (np.ndarray): 3D array of initial Y-coordinates.
        Z (np.ndarray): 3D array of initial Z-coordinates.
        tol (float): Convergence tolerance for the maximum change in coordinates.
        max_iter (int): Maximum number of iterations.
        ortho_weight (float): Weight for the orthogonality terms.
                              0.0 is equivalent to a Laplace solver.
                              Values > 0 increase orthogonality.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Smoothed X, Y, and Z grids.
    """
    X_new, Y_new, Z_new = X.copy(), Y.copy(), Z.copy()

    # A small epsilon to prevent division by zero
    epsilon = 1e-12
    
    # We define computational coordinates xi, eta, zeta to align with array
    # indices j, i, k respectively for clarity with the slicing.
    # i corresponds to axis 0, j to axis 1, k to axis 2.

    for iteration in range(max_iter):
        X_old, Y_old, Z_old = X_new.copy(), Y_new.copy(), Z_new.copy()

        # === 1. Calculate first derivatives on the interior grid ===
        # Derivatives are calculated at the center of each interior cell.
        # Slicing like [1:-1, 2:, 1:-1] selects the interior in i and k,
        # and the relevant neighbors in j.
        
        # Derivatives w.r.t. xi (along axis 1, index j)
        X_xi = (X_old[1:-1, 2:  , 1:-1] - X_old[1:-1, 0:-2, 1:-1]) / 2.0
        Y_xi = (Y_old[1:-1, 2:  , 1:-1] - Y_old[1:-1, 0:-2, 1:-1]) / 2.0
        Z_xi = (Z_old[1:-1, 2:  , 1:-1] - Z_old[1:-1, 0:-2, 1:-1]) / 2.0

        # Derivatives w.r.t. eta (along axis 0, index i)
        X_eta = (X_old[2:  , 1:-1, 1:-1] - X_old[0:-2, 1:-1, 1:-1]) / 2.0
        Y_eta = (Y_old[2:  , 1:-1, 1:-1] - Y_old[0:-2, 1:-1, 1:-1]) / 2.0
        Z_eta = (Z_old[2:  , 1:-1, 1:-1] - Z_old[0:-2, 1:-1, 1:-1]) / 2.0

        # Derivatives w.r.t. zeta (along axis 2, index k)
        X_zeta = (X_old[1:-1, 1:-1, 2:  ] - X_old[1:-1, 1:-1, 0:-2]) / 2.0
        Y_zeta = (Y_old[1:-1, 1:-1, 2:  ] - Y_old[1:-1, 1:-1, 0:-2]) / 2.0
        Z_zeta = (Z_old[1:-1, 1:-1, 2:  ] - Z_old[1:-1, 1:-1, 0:-2]) / 2.0

        # === 2. Calculate metric tensor components (g_ij) ===
        # These define the transformation from computational to physical space.
        g11 = X_xi**2 + Y_xi**2 + Z_xi**2
        g22 = X_eta**2 + Y_eta**2 + Z_eta**2
        g33 = X_zeta**2 + Y_zeta**2 + Z_zeta**2

        g12 = X_xi * X_eta + Y_xi * Y_eta + Z_xi * Z_eta
        g13 = X_xi * X_zeta + Y_xi * Y_zeta + Z_xi * Z_zeta
        g23 = X_eta * X_zeta + Y_eta * Y_zeta + Z_eta * Z_zeta

        # === 3. Calculate coefficients (A_ij) for the PDE ===
        # These are related to the cofactors of the metric tensor matrix.
        A11 = g22 * g33 - g23**2
        A22 = g11 * g33 - g13**2
        A33 = g11 * g22 - g12**2

        A12 = g13 * g23 - g12 * g33
        A13 = g12 * g23 - g13 * g22
        A23 = g12 * g13 - g23 * g11

        # === 4. Calculate mixed derivatives ===
        X_xi_eta = (X_old[2:, 2:, 1:-1] - X_old[0:-2, 2:, 1:-1] - X_old[2:, 0:-2, 1:-1] + X_old[0:-2, 0:-2, 1:-1]) / 4.0
        Y_xi_eta = (Y_old[2:, 2:, 1:-1] - Y_old[0:-2, 2:, 1:-1] - Y_old[2:, 0:-2, 1:-1] + Y_old[0:-2, 0:-2, 1:-1]) / 4.0
        Z_xi_eta = (Z_old[2:, 2:, 1:-1] - Z_old[0:-2, 2:, 1:-1] - Z_old[2:, 0:-2, 1:-1] + Z_old[0:-2, 0:-2, 1:-1]) / 4.0

        X_xi_zeta = (X_old[1:-1, 2:, 2:] - X_old[1:-1, 0:-2, 2:] - X_old[1:-1, 2:, 0:-2] + X_old[1:-1, 0:-2, 0:-2]) / 4.0
        Y_xi_zeta = (Y_old[1:-1, 2:, 2:] - Y_old[1:-1, 0:-2, 2:] - Y_old[1:-1, 2:, 0:-2] + Y_old[1:-1, 0:-2, 0:-2]) / 4.0
        Z_xi_zeta = (Z_old[1:-1, 2:, 2:] - Z_old[1:-1, 0:-2, 2:] - Z_old[1:-1, 2:, 0:-2] + Z_old[1:-1, 0:-2, 0:-2]) / 4.0

        X_eta_zeta = (X_old[2:, 1:-1, 2:] - X_old[0:-2, 1:-1, 2:] - X_old[2:, 1:-1, 0:-2] + X_old[0:-2, 1:-1, 0:-2]) / 4.0
        Y_eta_zeta = (Y_old[2:, 1:-1, 2:] - Y_old[0:-2, 1:-1, 2:] - Y_old[2:, 1:-1, 0:-2] + Y_old[0:-2, 1:-1, 0:-2]) / 4.0
        Z_eta_zeta = (Z_old[2:, 1:-1, 2:] - Z_old[0:-2, 1:-1, 2:] - Z_old[2:, 1:-1, 0:-2] + Z_old[0:-2, 1:-1, 0:-2]) / 4.0
        
        # === 5. Apply Jacobi iteration to update interior points ===
        # The equation comes from discretizing the 3D TTM PDE and solving for the central point.
        denom = 2 * (A11 + A22 + A33 + epsilon)
        
        # Orthogonality control: apply weight to mixed derivative terms
        C12 = ortho_weight * A12
        C13 = ortho_weight * A13
        C23 = ortho_weight * A23
        
        # -- Update X --
        num_X = (A22 * (X_old[2:, 1:-1, 1:-1] + X_old[0:-2, 1:-1, 1:-1]) +
                 A11 * (X_old[1:-1, 2:, 1:-1] + X_old[1:-1, 0:-2, 1:-1]) +
                 A33 * (X_old[1:-1, 1:-1, 2:] + X_old[1:-1, 1:-1, 0:-2]) -
                 2 * (C12 * X_xi_eta + C13 * X_xi_zeta + C23 * X_eta_zeta))
        X_new[1:-1, 1:-1, 1:-1] = num_X / denom

        # -- Update Y --
        num_Y = (A22 * (Y_old[2:, 1:-1, 1:-1] + Y_old[0:-2, 1:-1, 1:-1]) +
                 A11 * (Y_old[1:-1, 2:, 1:-1] + Y_old[1:-1, 0:-2, 1:-1]) +
                 A33 * (Y_old[1:-1, 1:-1, 2:] + Y_old[1:-1, 1:-1, 0:-2]) -
                 2 * (C12 * Y_xi_eta + C13 * Y_xi_zeta + C23 * Y_eta_zeta))
        Y_new[1:-1, 1:-1, 1:-1] = num_Y / denom

        # -- Update Z --
        num_Z = (A22 * (Z_old[2:, 1:-1, 1:-1] + Z_old[0:-2, 1:-1, 1:-1]) +
                 A11 * (Z_old[1:-1, 2:, 1:-1] + Z_old[1:-1, 0:-2, 1:-1]) +
                 A33 * (Z_old[1:-1, 1:-1, 2:] + Z_old[1:-1, 1:-1, 0:-2]) -
                 2 * (C12 * Z_xi_eta + C13 * Z_xi_zeta + C23 * Z_eta_zeta))
        Z_new[1:-1, 1:-1, 1:-1] = num_Z / denom

        # === 6. Check for convergence ===
        error_x = np.abs(X_new - X_old).max()
        error_y = np.abs(Y_new - Y_old).max()
        error_z = np.abs(Z_new - Z_old).max()

        if error_x < tol and error_y < tol and error_z < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return X_new, Y_new, Z_new

    print(f"Warning: Did not converge after {max_iter} iterations.")
    return X_new, Y_new, Z_new

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
    point_order = np.arange(nnode * nnode).reshape((nnode, nnode))
    t = gen_tensor_topology(shape)

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

def gen_spherical_mesh(nnode: int, r: float) -> Mesh:
    """
    Generates a high-quality, structured hexahedral mesh within a sphere.

    This function starts with a Cartesian cubic grid and transforms it into a
    spherical domain using a low-distortion mapping analogous to the 2D
    "squircle" method. This avoids the severe element skewing and singularity
    issues common at the poles with other mapping methods.

    Args:
        nnode (int): The number of nodes along one edge of the grid.
                     Should be odd for a center node.
        r (float): The radius of the outer spherical boundary.

    Returns:
        Mesh: An object containing the mesh's points and topology.
    """
    # 1. Create a standard Cartesian grid within a cube from -r to r
    lin = np.linspace(-r, r, nnode)
    X, Y, Z = np.meshgrid(lin, lin, lin)

    # 2. Apply the superior cube-to-sphere mapping
    # This transformation is fully vectorized.
    
    # Calculate Euclidean distance and "cube" distance for each point
    # Note: np.linalg.norm could also be used but is slower for this direct case.
    r_euclidean = np.sqrt(X**2 + Y**2 + Z**2)
    
    # The L-infinity norm (Chebyshev distance) is the max absolute coordinate
    r_cube = np.max(np.abs(np.array([X, Y, Z])), axis=0)
    
    # Calculate the scaling factor to map the cube to a sphere
    # Use np.divide to safely handle the origin (0/0 -> 0, handled by `where`)
    # We initialize the output `out` array with ones, so the scale at the origin
    # is 1, leaving it unchanged.
    scale = np.divide(r_cube, r_euclidean, out=np.ones_like(r_euclidean), where=(r_euclidean != 0))

    # Apply the transformation to get the new node positions
    X_new = X * scale
    Y_new = Y * scale
    Z_new = Z * scale
    
    X, Y, Z, = X_new, Y_new, Z_new
    X_new, Y_new, Z_new = solve_elliptic_grid_generator3D(X, Y, Z)
    # 3. Assemble the final mesh data structure
    points = np.vstack((X_new.flatten(), Y_new.flatten(), Z_new.flatten())).T
    shape = (nnode, nnode, nnode)
    topology = gen_tensor_topology(shape)

    # Facets (boundary faces) could be defined here if needed, but are omitted for brevity
    # as they are more complex than in 2D (6 faces of a cube).

    return Mesh(p=points, t=topology, element='hexahedron', facet=[])

def gen_torus_mesh(): return
