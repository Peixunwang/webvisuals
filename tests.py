import unittest
from src.mesh_generators import gen_block_mesh
from src.mesh import Mesh


class TestMesh(unittest.TestCase):
    
    def test_gen_block_mesh(self):
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
                        [7, 8, 11, 10],
                        [9, 10, 13, 12],
                        [10, 11, 14, 13]],
                        'quad',
                        [[0, 1], [1, 2], 
                         [2, 5], [5, 8], [8, 11], [11, 14], 
                         [14, 13], [13, 12], 
                         [12, 9], [9, 6], [6, 3], [3, 0]])
        test_mesh_2d = gen_block_mesh([(0, 1), (0, 2)], (3, 5))
        self.assertEqual(mesh_2d, test_mesh_2d)


if __name__ == '__main__':
    unittest.main()