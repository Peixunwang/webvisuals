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
    p: List[float] = field(default_factory=list)
    t: List[float] = field(default_factory=list)
    element: Optional[str] = None
    facet: Optional[List[int]] = None
    dim: int = 0
    
    def __post_init__(self):
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

    def convert_t(self, old_t=None) -> List[List[int]]:
        '''
            rearrange the hexahedron mesh connectivity order for use in 
            scikit-fem
        '''
        if old_t is None:
            old_t = self.t.copy()
        new_t = []
        for t in old_t:
            new_order = [0, 4, 3, 1, 7, 5, 2, 6]
            reordered_list = [t[i] for i in new_order]
            new_t.append(reordered_list)
        return new_t



    def facet_from_nodes(self, nodes):
        if self.dim == 2:
            return
        if self.dim == 3:
            return self.quad_from_nodes(nodes)


    def quad_from_nodes(self, nodes: List[int]) -> List[List[int]]:
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
        nodes = set(nodes)
        quad_t = []
        for cell in self.t:
            for face in self.get_hexahedron_faces(cell):
                if set(face).issubset(nodes):
                    quad_t.append(face)
        return quad_t
    
    def get_hexahedron_faces(self, cell: List[int]) -> List[List[int]]:
        """
        Returns the indices of the nodes for each of the 6 faces of a hexahedron,
        Node order: (0,1,2,3) for bottom, (4,5,6,7) for top:

                7---6
               /   /|
              4---5 2
              |   |/
              0---1
        """
        faces = [
            [cell[0], cell[1], cell[2], cell[3]],  # Bottom (0-1-2-3)
            [cell[4], cell[5], cell[6], cell[7]],  # Top (4-5-6-7)
            [cell[0], cell[1], cell[5], cell[4]],  # Front (0-1-5-4)
            [cell[3], cell[2], cell[6], cell[7]],  # Back (3-2-6-7)
            [cell[0], cell[3], cell[7], cell[4]],  # Left (0-3-7-4)
            [cell[1], cell[2], cell[6], cell[5]]   # Right (1-2-6-5)
        ]
        return faces
    
    def nodes_satisfy(self, expression) -> List[int]:
        '''
        return node id where the expression condition is satisfied
        '''

        iter = filter(lambda x: expression(x[1]), enumerate(self.p))
        indices = [index for index, value in iter]
        return indices

    def save(self, 
             file,
             point_data: Optional[Dict[str, Union[List, np.ndarray]]] = None,
             cell_data: Optional[Dict[str, Union[List, np.ndarray]]] = None,
             **kwargs
             ) -> None:
        try:
            import meshio
            tmp = meshio.Mesh(self.p, {self.element : self.t}, point_data, cell_data, **kwargs)
            meshio.write(file, tmp)
        except ImportError:
            print('meshio is not available, not saving mesh')

if __name__ == '__main__':
    mesh = Mesh()
