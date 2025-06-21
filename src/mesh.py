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
    in form of [[n_1, n_2, n_3, n_4, ...], [n_1, n_2, n_3, n_4, ...], ...]:

    
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
         
    element: the meshio type of the element
    """
    p: List[float] = field(default_factory=list)
    t: List[float] = field(default_factory=list)
    element: Optional[str] = None
    facet: Optional[List[int]] = None
    dim: int = 0
    
    def __post_init__(self):
        self.dim = len(self.p[0])


    def facet_from_points(self, points):
        pass
    
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
            print('meshio is not available, cannot save mesh')

if __name__ == '__main__':
    mesh = Mesh()