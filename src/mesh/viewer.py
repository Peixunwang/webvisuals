import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_interactions import zoom_factory

class Viewer:
    """
    A wrapper class for Matplotlib's 3D plotting to simplify surface plots.
    """
    def __init__(self, figsize=(10, 8)):
        """
        Initializes the Viewer, creating a Matplotlib figure and a 3D axes object.
        
        Args:
            figsize (tuple): The size of the figure (width, height) in inches.
        """
        # with plt.ioff():
        self.fig = plt.figure(figsize=figsize)
            # self.pan_handler = panhandler(self.fig)
        self.ax: Axes3D = self.fig.add_subplot(projection='3d')
        self.disconnect_zoom = self.zoom_factory_3d(self.ax)

    def plot_surf(self, X, Y, Z, **kwargs):
        """
        Plots a 3D surface on the viewer's axes.
        
        This method is a wrapper around `ax.plot_surface`.
        
        Args:
            X (array-like): 2D array of x-coordinates.
            Y (array-like): 2D array of y-coordinates.
            Z (array-like): 2D array of z-coordinates.
            **kwargs: Arbitrary keyword arguments to pass to `ax.plot_surface`.
                      Examples: `color`, `edgecolors`, `linewidth`, `shade`.
        """
        self.ax.plot_surface(X, Y, Z, **kwargs)

    def set_labels(self, xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis"):
        """Sets the labels for the x, y, and z axes."""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_zlabel(zlabel)

    def set_aspect(self, aspect='equal'):
        """Sets the aspect ratio for the plot axes."""
        self.ax.set_aspect(aspect)

    def zoom_factory_3d(self, ax: Axes3D, base_scale=1.1):
        """
        Creates a zoom factory for a 3D axes.

        Args:
            ax (Axes3D): The 3D axes to attach the zoom functionality to.
            base_scale (float): The zoom factor. A value > 1 will zoom out, < 1 will zoom in.
        """
        def zoom_fun(event):
            # Check if the scroll event occurred inside the axes
            if event.inaxes != ax:
                return

            # Determine zoom direction
            if event.button == 'up':
                # Zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # Zoom out
                scale_factor = base_scale
            else:
                # Not a recognized zoom event
                return

            # Get current limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_zlim = ax.get_zlim()

            # Calculate the center of the current view
            x_center = (cur_xlim[1] + cur_xlim[0]) / 2
            y_center = (cur_ylim[1] + cur_ylim[0]) / 2
            z_center = (cur_zlim[1] + cur_zlim[0]) / 2

            # Get the current range of the axes
            x_range = (cur_xlim[1] - cur_xlim[0])
            y_range = (cur_ylim[1] - cur_ylim[0])
            z_range = (cur_zlim[1] - cur_zlim[0])
            
            # Apply the scale factor to the range
            new_x_range = x_range * scale_factor
            new_y_range = y_range * scale_factor
            new_z_range = z_range * scale_factor

            # Set new limits, centered on the old center
            ax.set_xlim([x_center - new_x_range / 2, x_center + new_x_range / 2])
            ax.set_ylim([y_center - new_y_range / 2, y_center + new_y_range / 2])
            ax.set_zlim([z_center - new_z_range / 2, z_center + new_z_range / 2])

            # Redraw the plot
            ax.figure.canvas.draw_idle()


        # Get the figure canvas and connect the scroll event
        fig = ax.get_figure()
        cid = fig.canvas.mpl_connect('scroll_event', zoom_fun)

        # Return a disconnect function
        def disconnect():
            fig.canvas.mpl_disconnect(cid)

        return disconnect


    def show(self):
        """Displays the final plot."""
        plt.show()
