from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple
import numpy as np

'''
To do list:
1. to_tri() / to_tet()
'''

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
    t: List[int] = field(default_factory=list)
    element: Optional[str] = None
    facet: Optional[List[int]] = None
    dim: int = 0
    boundaries: Optional[Dict[str, List]] = None
    cell_data: Dict[str, List] = None

    def __post_init__(self):
        self.dim = len(self.p[0])

    def plot_mesh(self):
        if self.dim == 2:
            self.plot_mesh2d()
        if self.dim == 3:
            self.plot_mesh3d()

    def plot_mesh2d(
        self,
        surfaces: Optional[List] = None,
        meshgrid: Optional[List[tuple]] = None,
        colors: Optional[List[str]] = None,
        projection: str = "auto",
        figsize: tuple = (10, 8),
        show: bool = True,
        **plot_kwargs,
    ):
        """
        Plot boundary surfaces projected onto a 2D plane.

        Takes the same ``surfaces`` / ``meshgrid`` / ``colors`` interface as
        :meth:`plot_mesh3d`, but projects each 3-D meshgrid down to two axes
        and draws the result with :func:`matplotlib.pyplot.pcolormesh` (for
        the filled cells) and overlaid quad edges.

        Projection
        ----------
        For a planar surface the third coordinate is (nearly) constant.
        ``projection`` controls which two axes are kept:

        ========== ========================================
        Value      Kept axes
        ========== ========================================
        ``"auto"`` Automatically drops the axis with the
                   smallest range (i.e. the "flat" axis).
        ``"xy"``   Keep $x$ and $y$  (drop $z$).
        ``"xz"``   Keep $x$ and $z$  (drop $y$).
        ``"yz"``   Keep $y$ and $z$  (drop $x$).
        ========== ========================================

        Args:
            surfaces: Boundary name groups – same format as
                :meth:`plot_mesh3d`.
            meshgrid: Pre-computed ``(X, Y, Z)`` tuples.
            colors: One colour string per group.
            projection: Which 2-D plane to project onto.
            figsize: Matplotlib figure size.
            show: Call ``plt.show()`` automatically.
            **plot_kwargs: Extra keyword arguments forwarded to every
                ``ax.plot`` call for edges (e.g. ``linewidth``, ``linestyle``).

        Returns:
            Tuple[Figure, Axes]: The matplotlib figure and axes so callers can
            further customise the plot.

        Example::

            # By boundary name
            fig, ax = mesh.plot_mesh2d(
                surfaces=[["top", "bottom"], "left"],
                colors=["lightblue", "lightgreen"],
                projection="xz",
            )

            # By raw meshgrid
            X, Y, Z = mesh.meshgrid_from_faces("front")
            fig, ax = mesh.plot_mesh2d(
                meshgrid=[(X, Y, Z)],
                colors=["salmon"],
            )
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        # -- Build ordered list of (X, Y, Z) groups ---------------------------
        grids: List[tuple] = []

        # 1. Named surface groups
        if surfaces is not None:
            if self.boundaries is None:
                raise ValueError(
                    "Cannot look up surface names: `self.boundaries` is None."
                )
            for entry in surfaces:
                keys = [entry] if isinstance(entry, str) else list(entry)
                for key in keys:
                    if key not in self.boundaries:
                        raise KeyError(
                            f"'{key}' not found in boundaries. "
                            f"Available: {list(self.boundaries.keys())}"
                        )
                X, Y, Z = self.meshgrid_from_faces(*keys)
                grids.append((X, Y, Z))

        # 2. Explicit meshgrid tuples
        if meshgrid is not None:
            for grid in meshgrid:
                if len(grid) != 3:
                    raise ValueError(
                        "Each meshgrid entry must be a (X, Y, Z) tuple."
                    )
                grids.append(tuple(grid))

        # 3. Fallback: plot every boundary as its own group
        if not grids and self.boundaries:
            for key in self.boundaries:
                X, Y, Z = self.meshgrid_from_faces(key)
                grids.append((X, Y, Z))

        # -- Resolve colours ---------------------------------------------------
        _DEFAULT_CYCLE = [
            "cyan", "lightgreen", "salmon", "khaki",
            "plum", "lightskyblue", "peachpuff", "thistle",
        ]

        if colors is not None:
            if len(colors) != len(grids):
                raise ValueError(
                    f"Length of `colors` ({len(colors)}) must match the "
                    f"number of surface groups ({len(grids)})."
                )
            resolved_colors = list(colors)
        else:
            resolved_colors = [
                _DEFAULT_CYCLE[i % len(_DEFAULT_CYCLE)]
                for i in range(len(grids))
            ]

        # -- Determine projection axes ----------------------------------------
        def _detect_projection(grids: List[tuple]) -> Tuple[int, int, str, str]:
            """Return (axis_h, axis_v, label_h, label_v) for the two kept axes."""
            all_X = np.concatenate([np.asarray(g[0]).ravel() for g in grids])
            all_Y = np.concatenate([np.asarray(g[1]).ravel() for g in grids])
            all_Z = np.concatenate([np.asarray(g[2]).ravel() for g in grids])

            ranges = [
                np.ptp(all_X),  # axis 0 – X
                np.ptp(all_Y),  # axis 1 – Y
                np.ptp(all_Z),  # axis 2 – Z
            ]
            drop = int(np.argmin(ranges))
            axes_labels = ["X", "Y", "Z"]
            kept = [i for i in range(3) if i != drop]
            return kept[0], kept[1], axes_labels[kept[0]], axes_labels[kept[1]]

        _PROJ_MAP = {
            "xy": (0, 1, "X", "Y"),
            "xz": (0, 2, "X", "Z"),
            "yz": (1, 2, "Y", "Z"),
        }

        if projection == "auto":
            ax_h, ax_v, lbl_h, lbl_v = _detect_projection(grids)
        elif projection in _PROJ_MAP:
            ax_h, ax_v, lbl_h, lbl_v = _PROJ_MAP[projection]
        else:
            raise ValueError(
                f"Unknown projection '{projection}'. "
                f"Use 'auto', 'xy', 'xz', or 'yz'."
            )

        # -- Helper: extract the two kept coordinates from a grid --------------
        def _get_uv(grid: tuple, ah: int, av: int):
            """Return (U, V) arrays for the two kept axes."""
            components = [np.asarray(grid[0]), np.asarray(grid[1]), np.asarray(grid[2])]
            return components[ah], components[av]

        # -- Plot defaults -----------------------------------------------------
        default_edge_kwargs = dict(
            linewidth=0.5,
            linestyle="-",
        )
        default_edge_kwargs.update(plot_kwargs)

        edge_color = default_edge_kwargs.pop("edgecolor", "black")
        edge_color = default_edge_kwargs.pop("edgecolors", edge_color)

        # -- Create figure & axes ----------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        for (X, Y, Z), facecolor in zip(grids, resolved_colors):
            U, V = _get_uv((X, Y, Z), ax_h, ax_v)
            nrows, ncols = U.shape

            # Build quad patches for each cell in the meshgrid
            patches = []
            for i in range(nrows - 1):
                for j in range(ncols - 1):
                    # Four corners of the quad, counter-clockwise
                    quad = np.array([
                        [U[i,     j],     V[i,     j]],
                        [U[i,     j + 1], V[i,     j + 1]],
                        [U[i + 1, j + 1], V[i + 1, j + 1]],
                        [U[i + 1, j],     V[i + 1, j]],
                    ])
                    patches.append(mpatches.Polygon(quad, closed=True))

            pc = PatchCollection(
                patches,
                facecolor=facecolor,
                edgecolor=edge_color,
                **default_edge_kwargs,
            )
            ax.add_collection(pc)

        ax.set_xlabel(f"{lbl_h}-axis")
        ax.set_ylabel(f"{lbl_v}-axis")
        ax.set_aspect("equal")
        ax.autoscale_view()

        if show:
            plt.show()

        return fig, ax


    def plot_mesh3d(
        self,
        surfaces: Optional[List[str]] = None,
        meshgrid: Optional[List[tuple]] = None,
        colors: Optional[List[str]] = None,
        figsize: tuple = (10, 8),
        show: bool = True,
        **surf_kwargs,
    ):
        """
        Plot 3D boundary surfaces using the integrated Viewer.

        Each entry in ``surfaces`` or ``meshgrid`` is treated as one *group*.
        A corresponding entry in ``colors`` sets the colour for that group.

        Args:
            surfaces: A list of boundary name strings (or lists of names that
                are merged into one group).  Each element can be:

                - ``str``  – a single boundary name, e.g. ``"top"``
                - ``List[str]`` – several names merged into one group,
                  e.g. ``["top", "front", "bottom", "back"]``

                Each group is assigned the colour at the same index in
                ``colors``.
            meshgrid: A list of ``(X, Y, Z)`` meshgrid tuples.  Each tuple is
                one group and gets the colour at the matching index in
                ``colors``.
            colors: Colours for every group.  The length must equal the total
                number of groups (``len(surfaces or []) + len(meshgrid or
                [])``).  If ``None``, a default colour cycle is used.
            figsize: Figure size forwarded to :class:`Viewer`.
            show: Whether to call ``plt.show()`` at the end.
            **surf_kwargs: Extra keyword arguments forwarded to **every**
                ``ax.plot_surface`` call (e.g. ``edgecolors``, ``linewidth``,
                ``alpha``, ``shade``).

        Returns:
            Viewer: The :class:`Viewer` instance for further customisation.

        Example::

            # Named boundaries with per-group colours
            viewer = mesh.plot_mesh3d(
                surfaces=[
                    ["top", "front", "bottom", "back"],
                    "left",
                    "right",
                ],
                colors=["lightblue", "lightgreen", "lightgreen"],
                edgecolors="black",
                linewidth=0.5,
            )

            # Raw meshgrids with per-group colours
            viewer = mesh.plot_mesh3d(
                meshgrid=[grid1, grid2],
                colors=["red", "blue"],
            )

            # Mix named + raw
            viewer = mesh.plot_mesh3d(
                surfaces=[["top", "bottom"]],
                meshgrid=[external_grid],
                colors=["lightblue", "salmon"],
            )
        """
        from .viewer import Viewer

        viewer = Viewer(figsize=figsize)

        # -- Build ordered list of (X, Y, Z) groups ---------------------------
        grids: List[tuple] = []

        # 1. Named surface groups
        if surfaces is not None:
            if self.boundaries is None:
                raise ValueError(
                    "Cannot look up surface names: `self.boundaries` is None."
                )
            for entry in surfaces:
                # Normalise to a list of keys
                keys = [entry] if isinstance(entry, str) else list(entry)
                for key in keys:
                    if key not in self.boundaries:
                        raise KeyError(
                            f"'{key}' not found in boundaries. "
                            f"Available: {list(self.boundaries.keys())}"
                        )
                X, Y, Z = self.meshgrid_from_faces(*keys)
                grids.append((X, Y, Z))

        # 2. Explicit meshgrid tuples
        if meshgrid is not None:
            for grid in meshgrid:
                if len(grid) != 3:
                    raise ValueError(
                        "Each meshgrid entry must be a (X, Y, Z) tuple."
                    )
                grids.append(tuple(grid))

        # 3. Fallback: plot every boundary as its own group
        if not grids and self.boundaries:
            for key in self.boundaries:
                X, Y, Z = self.meshgrid_from_faces(key)
                grids.append((X, Y, Z))

        # -- Resolve colours ---------------------------------------------------
        _DEFAULT_CYCLE = [
            "cyan", "lightgreen", "salmon", "khaki",
            "plum", "lightskyblue", "peachpuff", "thistle",
        ]

        if colors is not None:
            if len(colors) != len(grids):
                raise ValueError(
                    f"Length of `colors` ({len(colors)}) must match the "
                    f"number of surface groups ({len(grids)})."
                )
            resolved_colors = list(colors)
        else:
            resolved_colors = [
                _DEFAULT_CYCLE[i % len(_DEFAULT_CYCLE)]
                for i in range(len(grids))
            ]

        # -- Surface-kwarg defaults --------------------------------------------
        default_kwargs = dict(
            edgecolors="k",
            linewidth=0.3,
            shade=False,
        )
        default_kwargs.update(surf_kwargs)

        # -- Plot each group ---------------------------------------------------
        for (X, Y, Z), color in zip(grids, resolved_colors):
            viewer.plot_surf(X, Y, Z, color=color, **default_kwargs)

        viewer.set_labels("X-axis", "Y-axis", "Z-axis")
        viewer.set_aspect("equal")

        if show:
            viewer.show()

        return viewer



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

    def facet_from_nodes(self, nodes) -> List[List[int]]:
        if self.dim == 2:
            return
        if self.dim == 3:
            return self.quad_from_nodes(nodes)

    def quad_from_nodes(self, nodes: List[int]) -> List[List[int]]:
        """
        Identifies quadrilateral faces (topology indices) within the mesh that
        are defined by the given 'query_points (nodes)'.

        Assumes 'query_points' are the physical coordinates of the four corners
        of a quadrilateral face, the order does not strictly matter for matching
        since we check permutations.

        Args:
            nodes: A list of 4 points, each being a list of [x, y, z] coordinates.

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
    
    def meshgrid_from_faces(self, *keys):
        from .mesh_helpers import face_to_meshgrid
        facets = []
        for key in keys:
            indices = self.boundaries[key]
            facet = self.facet_from_nodes(indices)
            facets.extend(facet)
        X, Y, Z = face_to_meshgrid(self, facets)
        return X, Y, Z

    def get_hexahedron_faces(self, cell: List[int]) -> List[List[int]]:
        """
        Returns the indices of the nodes for each of the 6 faces of a hexahedron,
        Node order: (0,1,2,3) for bottom, (4,5,6,7) for top:

                7---6
               /   /|
              4---5 2
              |   |/
              0---1

        note: the actual cell connectivity ordering is:
              [0, 1, 3, 2, 4, 5, 7, 6]
        """
        faces = [
            [cell[3], cell[2], cell[1], cell[0]],  # Bottom (3-2-1-0)
            [cell[4], cell[5], cell[6], cell[7]],  # Top (4-5-6-7)
            [cell[0], cell[1], cell[5], cell[4]],  # Front (0-1-5-4)
            [cell[7], cell[6], cell[2], cell[3]],  # Back (7-6-2-3)
            [cell[3], cell[0], cell[4], cell[7]],  # Left (3-0-4-7)
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
    
    def translate(self, direction: Union[List[float], np.ndarray]):
        """
        Translates the mesh by a given vector.

        Args:
            direction (Union[List[float], np.ndarray]): The translation vector,
                e.g., [dx, dy, dz].
        """
        if not self.p:
            return  # Nothing to translate

        p_arr = np.array(self.p, dtype=float)
        direction_arr = np.array(direction, dtype=float)

        if p_arr.shape[1] != direction_arr.shape[0]:
            raise ValueError(
                f"Point dimension {p_arr.shape[1]} does not match "
                f"direction vector dimension {direction_arr.shape[0]}."
            )

        self.p = (p_arr + direction_arr).tolist()

    def rotate(self, direction):
        print('rotate mesh (place holder)')

    def remove_point(self, node):
        print('remove_point (place holder)')

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

class continuous_surface:
    def __init__(self, surf):
        self.surf = surf

    def is_periodic(self):
        print('place holder')

@dataclass
class Face:
    '''
        The four vertices of a face
    '''
    nodes = []

@dataclass
class Edge:
    '''
        The two ends of an edge
    '''
    nodes = []

if __name__ == '__main__':
    mesh = Mesh()
