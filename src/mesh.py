from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import numpy as np

@dataclass
class Mesh:
    """
    A mesh definition for manipulating FEA meshes
    p: points, the locations of the element nodes, 
       in form of [[x_1, y_1, z_1], [x_2, y_2, z_2], ...]
    t: topology, the connectivity of the elements/cells,
       in form of [[n_1, n_2, n_3, n_4, ...], [n_1, n_2, n_3, n_4, ...], ...]
    element: the meshio type of the element
    """
    p: np.ndarray = []
    t: np.ndarray = []
    element: str = ''
    facet = None
    


    @property
    def p(self):
        return self.p
    
    @property
    def t(self):
        return self.t
    
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