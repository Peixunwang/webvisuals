from .mesh import Mesh
from typing import List
import numpy as np

def face_to_meshgrid(mesh: Mesh, faces: List) -> np.ndarray:
    """
    Assembles faces to a numpy meshgrid, returning the [X, Y, Z] values of the 2D meshgrid.

    Args:
        mesh: A Mesh object
        faces: A list of lists, where each inner list represents a 2D cell 
               and contains four int indices referencing points in the mesh.

    Returns:
        A numpy array of shape (3, num_rows, num_cols) representing the X, Y, and Z
        coordinates of the meshgrid.
    """

    def left_to_face(face, faces):
        node = face[0]
        for obj in faces:
            if obj[1] == node:
                return obj
        return None
    
    def right_to_face(face, faces):
        node = face[1]
        for obj in faces:
            if obj[0] == node:
                return obj
        return None
    
    def top_to_face(face, faces):
        node = face[3]
        for obj in faces:
            if obj[0] == node:
                return obj
        return None

    def bottom_to_face(face, faces):
        node = face[0]
        for obj in faces:
            if obj[3] == node:
                return obj
        return None
    
    def left_most_face(face, faces):
        left = left_to_face(face, faces)
        if left:
            return left_most_face(left, faces)
        else:
            return face
        
    def bottom_most_face(face, faces):
        bottom = bottom_to_face(face, faces)
        if bottom:
            return bottom_most_face(bottom, faces)
        else:
            return face

    def bottom_left_face(face, faces):
        left = left_most_face(face, faces)
        bottom_left = bottom_most_face(left, faces)
        return bottom_left
    
    def make_row(face, faces):
        face_row = [face]
        while right_to_face(face, faces):
            face = right_to_face(face, faces)
            face_row.append(face)
        return face_row
    
    def make_grid(face, faces):
        faces_grid = [make_row(face, faces)]
        while top_to_face(face, faces):
            face = top_to_face(face, faces)
            faces_grid.append(make_row(face, faces))
        return faces_grid
    
    if not faces:
        return np.array([])
    
    face = bottom_left_face(faces[0], faces)
    faces_grid = make_grid(face, faces)

    
    print(faces_grid)
