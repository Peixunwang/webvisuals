from .mesh import Mesh, Face, Edge
from typing import List
import numpy as np
import itertools

def face_to_meshgrid(mesh: Mesh, faces: List[Face]) -> np.ndarray:
    faces_graph = get_faces_graph(faces)
    # print('faces_graph: ', faces_graph)
    nnode_x = len(faces_graph) + 1
    nnode_y = len(faces_graph[0]) + 1
    X = np.zeros((nnode_x, nnode_y))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    points = mesh.p
    for y, row in enumerate(faces_graph):
        for x, face in enumerate(row):
            X[y][x] = points[face[0]][0]
            Y[y][x] = points[face[0]][1]
            Z[y][x] = points[face[0]][2]

            X[y][x+1] = points[face[1]][0]
            Y[y][x+1] = points[face[1]][1]
            Z[y][x+1] = points[face[1]][2]

            X[y+1][x+1] = points[face[2]][0]
            Y[y+1][x+1] = points[face[2]][1]
            Z[y+1][x+1] = points[face[2]][2]

            X[y+1][x] = points[face[3]][0]
            Y[y+1][x] = points[face[3]][1]
            Z[y+1][x] = points[face[3]][2]
    return X, Y, Z

def get_nodes_graph(elem: List) -> List:
    '''
        elem is one of [edges, faces, cells]
    '''

    print('get_nodes_graph place holder')

def get_edges_graph(cells: List) -> List: print('get_edges_graph place holder')

def get_faces_graph(faces: List[Face]) -> List:
    """
    Assembles faces to a numpy meshgrid, returning the [X, Y, Z] values of the 2D meshgrid.

    Args:
        faces: A list of lists, where each inner list represents a 2D cell 
               and contains four int indices referencing points in the mesh.

    Returns:
        A numpy array of shape (3, num_rows, num_cols) representing the X, Y, and Z
        coordinates of the meshgrid.
    """

    def fix_orient(this_face: List, that_face: List):
        if not fix_orientation:
            return
        edge = list(set(this_face) & set(that_face))
        for i, node in enumerate(this_face):
            if edge[0] == node:
                n_1 = i
                break
        for i, node in enumerate(this_face):
            if edge[1] == node:
                n_2 = i
                break
        this_edge = [min([n_1, n_2]), max([n_1, n_2])]
        if this_edge == [0, 3]:
            this_edge = [3, 0]
        this_edge_indices = [this_face[this_edge[0]], this_face[this_edge[1]]]

        correct_rot = False
        this_ext = this_face.copy()
        this_ext.append(this_face[0])
        that_ext = that_face.copy()
        that_ext.append(that_face[0])
        for i in range(4):
            correct_rot = correct_rot or this_edge_indices == [that_ext[i+1], that_ext[i]]
        if not correct_rot:
            get_correct_rot(that_face)

        for i, node in enumerate(that_face):
            if edge[0] == node:
                n_1 = i
                break
        for i, node in enumerate(that_face):
            if edge[1] == node:
                n_2 = i
                break
        that_edge = [min([n_1, n_2]), max([n_1, n_2])]
        if that_edge == [0, 3]:
            that_edge = [3, 0]
        that_ext = that_face.copy()
        that_ext.append(that_face[0])
        get_edge_indices = lambda x, edge: [edge[x[0]], 
                                      edge[x[1]]]
        for i in range(4):
            if get_edge_indices([0, 1], this_face) == get_edge_indices([i+1, i], that_ext):
                shift_nodes(i-2, that_face)
            if get_edge_indices([1, 2], this_face) == get_edge_indices([i+1, i], that_ext):
                shift_nodes(i-3, that_face)
            if get_edge_indices([2, 3], this_face) == get_edge_indices([i+1, i], that_ext):
                shift_nodes(i, that_face)
            if get_edge_indices([3, 0], this_face) == get_edge_indices([i+1, i], that_ext):
                shift_nodes(i-1, that_face)

    def get_correct_rot(face: Face):
        face_nodes = [face[0], face[3], face[2], face[1]]
        face[:] = face_nodes

    def shift_nodes(start, face: Face):
        rotated_part = face[start:] 
        remaining_part = face[:start] 
        rotated_face = rotated_part + remaining_part
        face[:] = rotated_face

    def next_face(this_edge: set, this_face: Face) -> tuple[Edge, Face]:
        for that_face in faces:
            if that_face == this_face: 
                continue

            if this_edge.issubset(set(that_face)):
                for f in checked_faces:
                    if f == that_face:
                        return
                checked_faces.append(that_face)
                that_edge = set(that_face).difference(this_edge)
                fix_orient(this_face, that_face)
                return that_edge, that_face
        return None
    
    def last_face(edge: Edge, face: Face):
        test = next_face(edge, face)
        if test:
            return last_face(test[0], test[1])
        else:
            return edge, face

    def make_row(face: Face):
        face_row = []
        face_row.append(face)
        test = next_face({face[1], face[2]}, face)
        while test:
            face_row.append(test[1])
            test = next_face(test[0], test[1])
        return face_row

    if not faces:
        return []

    fix_orientation = True
    faces_graph = []
    checked_faces = [faces[0]]
    left_edge, left_face = last_face({faces[0][3], faces[0][0]}, faces[0])
    checked_faces = [left_face]
    bot_left_edge, bot_left_face = last_face({left_face[0], left_face[1]}, left_face)
    checked_faces = [bot_left_face]
    faces_graph.append(make_row(bot_left_face))
    test = next_face({bot_left_face[2], bot_left_face[3]}, bot_left_face)
    while test:
        faces_graph.append(make_row(test[1]))
        test = next_face(test[0], test[1])
    return faces_graph

def get_cells_graph(cells: List) -> List: print('get_cells_graph place holder')