from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
import numpy as np

@dataclass
class Mesh:
    """
    A mesh definition for manipulating FEA meshes.

    p: points, the locations of the element nodes,
    in form of [[x_1, y_1, z_1], [x_2, y_2, z_2], ...]

    t: topology, the connectivity of the elements/cells,
    in form of [[n_1, n_2, n_3, n_4, ...], [n_1, n_2, n_3, n_4, ...], ...]

    element: the meshio type of the element

    Cell structure:

        In 2D cell, vertices should match the order:

          3---2
          |   |
          0---1

        In 3D cell, order of vertices should match the numbering:

            7---6
           /   /|
          4---5 2
          |   |/
          0---1

    """
    p: List[List[float]] = field(default_factory=list) # Changed to List[List[float]] for clarity
    t: List[List[int]] = field(default_factory=list)    # Changed to List[List[int]] for clarity
    element: Optional[str] = None
    facet: Optional[List[int]] = None
    dim: int = 0

    def __post_init__(self):
        if self.p: # Check if points exist before accessing self.p[0]
            self.dim = len(self.p[0])

    def plot_mesh(self):
        if self.dim == 2:
            self.plot_mesh2d()
        if self.dim == 3:
            self.plot_mesh3d()

    def plot_mesh2d(self):
        print("Plotting 2D mesh (placeholder)")
        return

    def plot_mesh3d(self):
        print("Plotting 3D mesh (placeholder)")
        return

    def convert_t(self, old_t=None):
        '''
            rearrange the hexahedron mesh connectivity order for use in
            scikit-fem
        '''
        if old_t is None:
            old_t = self.t.copy()
        new_t = []
        # This conversion typically applies to hexahedrons (8 nodes)
        if len(old_t) > 0 and len(old_t[0]) == 8:
            for t_elem in old_t:
                # The given new_order is for scikit-fem's hexahedron ordering from a common (e.g., meshio) order
                # Original: 0, 1, 2, 3, 4, 5, 6, 7
                # scikit-fem: 0, 4, 3, 1, 7, 5, 2, 6 (or similar, depending on scikit-fem version)
                # The provided order [0, 4, 3, 1, 7, 5, 2, 6] seems unusual for a standard reordering.
                # Common reordering for hex: 0, 1, 2, 3, 4, 5, 6, 7 -> 0, 1, 5, 4, 3, 2, 6, 7 (reordering bottom/top faces)
                # Or other variations. For now, use the one given.
                new_order = [0, 4, 3, 1, 7, 5, 2, 6]
                reordered_list = [t_elem[i] for i in new_order]
                new_t.append(reordered_list)
        else:
            print("Warning: convert_t is typically for 8-node hexahedrons, input t does not match.")
            new_t = old_t.copy() # If not 8-node, return original
        return new_t

    def get_hexahedron_faces(self, element_nodes: List[int]) -> List[List[int]]:
        """
        Returns the indices of the nodes for each of the 6 faces of a hexahedron,
        assuming the standard numbering.
        """
        # Node order: (0,1,2,3) for bottom, (4,5,6,7) for top
        #  7---6
        # /   /|
        # 4---5 2
        # |   |/
        # 0---1
        faces = [
            [element_nodes[0], element_nodes[1], element_nodes[2], element_nodes[3]],  # Bottom (0-1-2-3)
            [element_nodes[4], element_nodes[5], element_nodes[6], element_nodes[7]],  # Top (4-5-6-7)
            [element_nodes[0], element_nodes[1], element_nodes[5], element_nodes[4]],  # Front (0-1-5-4)
            [element_nodes[3], element_nodes[2], element_nodes[6], element_nodes[7]],  # Back (3-2-6-7)
            [element_nodes[0], element_nodes[3], element_nodes[7], element_nodes[4]],  # Left (0-3-7-4)
            [element_nodes[1], element_nodes[2], element_nodes[6], element_nodes[5]]   # Right (1-2-6-5)
        ]
        return faces

    def quad_from_points(self, query_points: List[List[float]]) -> List[List[int]]:
        """
        Identifies quadrilateral faces (topology indices) within the mesh that
        are defined by the given 'query_points'.

        Assumes 'query_points' are the physical coordinates of the four corners
        of a quadrilateral face, the order does not strictly matter for matching
        since we check permutations.

        Args:
            query_points: A list of 4 points, each being a list of [x, y, z] coordinates.

        Returns:
            A list of lists. Each inner list represents the global node indices
            of a quad cell found that matches the query points. An empty list
            if no match is found. This represents `quad_t` in the problem statement.
        """
        quad_t = []
        if self.dim != 3:
            print("Error: quad_from_points expects a 3D mesh.")
            return quad_t
        if len(query_points) != 4:
            print("Error: query_points must contain exactly 4 points for a quad.")
            return quad_t

        # Convert query_points to a set of tuples for efficient lookup, accounting for permutations
        # We need the global node indices corresponding to these query_points.
        # First, find the indices of the query_points in self.p
        global_indices_of_query_points = []
        p_np = np.array(self.p) # Convert to numpy array for faster lookup
        
        # Determine a small tolerance for floating point comparisons
        tolerance = 1e-9 
        
        for q_pt in query_points:
            # Find the index of the point in self.p that matches q_pt
            # np.isclose is better for float comparisons than direct equality
            match_indices = np.where(np.all(np.isclose(p_np, q_pt, atol=tolerance), axis=1))[0]
            if len(match_indices) == 0:
                print(f"Warning: Query point {q_pt} not found in mesh points.")
                return [] # A query point is not in the mesh, no quad can be formed.
            # Assuming uniqueness, take the first match
            global_indices_of_query_points.append(match_indices[0])

        # Create a frozenset for permutation-invariant comparison
        query_indices_set = frozenset(global_indices_of_query_points)

        # Iterate through each element in the mesh
        for i, element_nodes in enumerate(self.t):
            # We assume hexahedron elements for quad faces in 3D
            if len(element_nodes) == 8: # Check if it's a hexahedron
                faces_of_element = self.get_hexahedron_faces(element_nodes)
                
                for face_nodes in faces_of_element:
                    # Convert face_nodes to a frozenset for permutation-invariant comparison
                    face_set = frozenset(face_nodes)
                    
                    if face_set == query_indices_set:
                        # Found a match! The face_nodes are the global indices of the quad.
                        # The order in face_nodes will be consistent with the hexahedron's face definition.
                        quad_t.append(face_nodes)
                        # Optionally, if you expect only one match, you can break here.
                        # For boundary facets, it's possible for multiple elements to share a face.
                        # However, for a user-specified set of points defining *one* face geometry,
                        # it's usually intended to find that specific face.
                        # If a specific element's face is desired, additional info would be needed.
            else:
                # Handle other 3D elements if necessary (e.g., tetrahedrons have triangular faces)
                # For this problem, we assumed quad faces imply hexahedrons.
                pass

        return quad_t

    def facet_from_points(self, points: List[List[float]]):
        """
        Placeholder for general facet generation.
        Currently, it specifically calls quad_from_points if 3D.
        """
        if self.dim == 3:
            # Assuming 'points' here are the 4 points defining a quad face.
            self.facet = self.quad_from_points(points)
            if not self.facet:
                print("No quad facet found matching the given points.")
        else:
            print("facet_from_points current implementation is for 3D meshes (quads).")


    def save(self,
             file: str, # Added type hint for file path
             point_data: Optional[Dict[str, Union[List, np.ndarray]]] = None,
             cell_data: Optional[Dict[str, Union[List, np.ndarray]]] = None,
             **kwargs
             ) -> None:
        try:
            import meshio
            # In meshio, cells are a list of (cell_type, connectivity_array) tuples
            # We need to structure self.t and self.element accordingly.
            if self.element and self.t:
                cells = [(self.element, np.array(self.t))]
            else:
                cells = []

            tmp = meshio.Mesh(np.array(self.p), cells, point_data, cell_data, **kwargs)
            meshio.write(file, tmp)
        except ImportError:
            print('meshio is not available, not saving mesh')
        except Exception as e:
            print(f"An error occurred while saving the mesh: {e}")

if __name__ == '__main__':
    # Example Usage:

    # 1. Create a simple 3D mesh (a single hexahedron)
    # Define points for a unit cube
    points_cube = [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0]   # 7
    ]
    # Define connectivity for a single hexahedron (using standard meshio/abaqus order)
    # 0,1,2,3,4,5,6,7
    connectivity_cube = [[0, 1, 2, 3, 4, 5, 6, 7]]

    mesh_cube = Mesh(p=points_cube, t=connectivity_cube, element="hexahedron")

    print(f"Mesh dimension: {mesh_cube.dim}")

    # 2. Test quad_from_points
    # Example 1: Bottom face (0-1-2-3) of the cube
    # Points defining the bottom face (order doesn't matter for the query set)
    query_points_bottom = [
        [0.0, 0.0, 0.0], # node 0
        [1.0, 0.0, 0.0], # node 1
        [1.0, 1.0, 0.0], # node 2
        [0.0, 1.0, 0.0]  # node 3
    ]
    found_quads_bottom = mesh_cube.quad_from_points(query_points_bottom)
    print("\nQuads found for bottom face:")
    print(found_quads_bottom) # Expected: [[0, 1, 2, 3]] (or a reordered permutation of these indices)

    # Example 2: Top face (4-5-6-7) of the cube
    query_points_top = [
        [0.0, 0.0, 1.0], # node 4
        [1.0, 0.0, 1.0], # node 5
        [1.0, 1.0, 1.0], # node 6
        [0.0, 1.0, 1.0]  # node 7
    ]
    found_quads_top = mesh_cube.quad_from_points(query_points_top)
    print("\nQuads found for top face:")
    print(found_quads_top) # Expected: [[4, 5, 6, 7]]

    # Example 3: A face with different point order (should still work due to frozenset)
    query_points_front_亂序 = [
        [1.0, 0.0, 0.0], # node 1
        [0.0, 0.0, 0.0], # node 0
        [0.0, 0.0, 1.0], # node 4
        [1.0, 0.0, 1.0]  # node 5
    ]
    found_quads_front = mesh_cube.quad_from_points(query_points_front_亂序)
    print("\nQuads found for front face (unordered query):")
    print(found_quads_front) # Expected: [[0, 1, 5, 4]]

    # Example 4: A point slightly off (should not be found)
    query_points_off = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.001] # Slightly off z-coordinate
    ]
    found_quads_off = mesh_cube.quad_from_points(query_points_off)
    print("\nQuads found for slightly off points:")
    print(found_quads_off) # Expected: [] and a warning

    # Example 5: Using facet_from_points with the same query
    mesh_cube.facet_from_points(query_points_bottom)
    print(f"\nFacet attribute after calling facet_from_points (bottom face): {mesh_cube.facet}")
    
    # Test convert_t (will reorder the single hexahedron)
    converted_t = mesh_cube.convert_t()
    print("\nConverted connectivity (scikit-fem order):")
    print(converted_t)
    # For element 0-1-2-3-4-5-6-7, new order [0, 4, 3, 1, 7, 5, 2, 6] results in:
    # [0, 4, 3, 1, 7, 5, 2, 6]

    # Test saving (requires meshio to be installed)
    try:
        import meshio
        mesh_cube.save("test_mesh.vtu")
        print("\nMesh saved to test_mesh.vtu")
    except ImportError:
        print("\nSkipping save test: meshio not installed.")

