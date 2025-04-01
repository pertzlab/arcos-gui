"""Module containing plotting classes."""

from re import findall
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from arcos4py.tools import calculate_statistics
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.notifications import show_info
from qtpy import QtWidgets
from scipy.stats import gaussian_kde

from ._config import ARCOS_LAYERS, COLOR_CYCLE
from ._shape_functions import fix_3d_convex_hull, get_bbox, get_bbox_3d

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer


class CollevPlotter(QtWidgets.QWidget):
    """
    QWidget for plotting a scatterplot of Collective events.
    Make a matplotlib figure canvas and add it to a Qwidget.
    Canvas, figure and axis objects can be acessed by self.canvas,
    self.fig and self.ax. This plots duration of Collective events over their size as
    returned by arcos.

    Attributes:
        viewer (napari.viewer.Viewer): Napari viewer instance
        parent (qtpy.QtWidgets.QWidget): Parent widget, optional
    """

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        """Class constructor.

        Parameters:
            viewer (napari.viewer.Viewer): Napari viewer instance
            parent (qtpy.QtWidgets.QWidget): Parent widget, optional
        """
        super().__init__(parent)
        self.viewer = viewer
        self.collid_name: str = "collid"
        self.nbr_collev: int = 0
        self.stats = pd.DataFrame(
            data={"total_size": [], "duration": [], self.collid_name: []}
        )
        self._callbacks: list = []
        self._init_mpl_widgets()
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.arcos = pd.DataFrame(data={"t": [], "id": [], "x": [], "y": [], "z": []})
        self.point_size = 10
        self.frame_column = "time"
        self.trackid_col = "id"
        self.posx = "x"
        self.posy = "y"
        self.posz = "z"
        self.output_order = "tzyx"
        self.update_plot(
            self.frame_column,
            self.trackid_col,
            self.posx,
            self.posy,
            self.posz,
            self.arcos,
            self.output_order,
        )

    def _init_mpl_widgets(self):
        """
        Method to initialise a matplotlib figure canvas, to generate,
        set plot style and axis, and populate it with a matplotlib.figure.Figure.
        """
        # construct layout
        self.layout_collevplot = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout_collevplot)

        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(2.5, 1.5))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("Total Size")
            self.ax.set_ylabel("Event Duration")

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_collevplot.addWidget(self.toolbar)
        self.layout_collevplot.addWidget(self.canvas)
        self.setWindowTitle("Collective Events")
        self._update_layout()

    def _update_layout(self):
        self.fig.tight_layout(pad=0.1, w_pad=0.001, h_pad=0.05)

    def clear_plot(self):
        """Method to clear the plot."""
        self.ax.cla()
        self.ax.set_xlabel("Total Size")
        self.ax.set_ylabel("Event Duration")
        self.fig.canvas.draw_idle()

    def update_plot(
        self,
        frame_column,
        trackid_col,
        posx,
        posy,
        posz,
        arcos_data,
        arcos_stats,
        point_size=10,
        output_order="tzyx",
    ):
        """
        Method to update the matplotlibl axis object self.ax with new values from
        the stored_variables object
        """
        collev_stats = arcos_stats
        self.arcos = arcos_data
        self.point_size = point_size
        self.frame_column = frame_column
        self.trackid_col = trackid_col
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.output_order = output_order
        # if no calculation was run so far (i.e. when the widget is initialized)
        # populate it with no data else calculate stats for collective events,
        # generate plot with data
        if not self.arcos.empty and self.trackid_col:
            self.stats = collev_stats

            self.ax.cla()
            self.ax.spines["bottom"].set_color("white")
            self.ax.spines["top"].set_color("white")
            self.ax.spines["right"].set_color("white")
            self.ax.spines["left"].set_color("white")
            self.ax.xaxis.label.set_color("white")
            self.ax.yaxis.label.set_color("white")
            self.ax.tick_params(colors="white", which="both")
            self.ax.axis("on")
            self.ax.scatter(
                self.stats[["total_size"]],
                self.stats[["duration"]],
                alpha=0.8,
                picker=True,
            )
            self.ax.set_xlabel("Total Size")
            self.ax.set_ylabel("Event Duration")
            self.fig.canvas.draw_idle()
            self.nbr_collev = self.stats[self.collid_name].nunique()
        # generate empty annotation, set it to invisible
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            bbox=dict(boxstyle="round", fc="#252932", ec="white", linewidth=0.3),
            fontsize=7,
            color="white",
            clip_on=True,
        )
        self.annot.set_visible(False)
        # instantiate blitmanager and add annotation to it.
        # Used to improve performance of annotations
        # rendering annotation label.

        self.blit_manager = BlitManager(self.canvas, [self.annot])

    def update_annot(self, ind):
        """Update the annotation.

        Updates hover annotation showing collective event id.

        Parameters:
            ind (dict): Index of plotted data corresponding to the
            current mouse location.
        """
        pos = self.ax.collections[0].get_offsets()[ind["ind"][0]]
        pos_text = pos.copy()
        text = f"id: {int(self.stats[self.collid_name][ind['ind'][0]])}"
        self.annot.set_text(text)
        self.fig.canvas.draw_idle()
        renderer = self.fig.canvas.get_renderer()
        bbox = self.annot.get_window_extent(renderer)
        bbox_data = self.ax.transData.inverted().transform(bbox)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        size_h = bbox_data[1][0] - bbox_data[0][0]
        size_v = bbox_data[1][1] - bbox_data[0][1]

        # moves lable by the size of the bounding box
        # if it would be cut off by the plot boundry
        if pos_text[0] > (xlim[1] - size_h):
            pos_text[0] -= size_h
        if pos_text[1] > (ylim[1] - size_v):
            pos_text[1] -= size_v
        self.annot.xy = pos
        self.annot.set_position(pos_text)
        self.annot.get_bbox_patch().set_alpha(1)

    def hover(self, event):
        """Display annotation of collective even on hover.

        Checks if current mouse position is over a datapoint. If yes,
        displays corresponding collective event id.
        """
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            try:
                cont, ind = self.ax.collections[0].contains(event)
                if cont:
                    self.update_annot(ind)
                    self.annot.set_visible(True)
                    # blitting to improve performance.
                    self.blit_manager.update()
                else:
                    if vis:
                        self.annot.set_visible(False)
                        # blitting to improve performance.
                        self.blit_manager.update()
            except IndexError:
                pass

    def on_pick(self, event):
        """Displays the selected collective event in the napari viewer.

        On pick gets collective event id from picked datapoint, gets
        the correspondig starting frame, moves to this and
        draws a bounding box arround the extends of the collective event.

        Parameters:
            event (matplotlib_pick_event): event generated from selecting a datapoint.
        """
        ind = event.ind
        clid = int(self.stats.iloc[ind[0]][self.collid_name])
        current_colev = self.arcos[self.arcos["collid"] == clid]
        edge_size = self.point_size / 5
        frame = self.stats.iloc[ind[0]].iloc[2]
        if ARCOS_LAYERS["event_boundingbox"] in self.viewer.layers:
            self.viewer.layers.remove(ARCOS_LAYERS["event_boundingbox"])
        if self.posz is None:
            bbox, bbox_param = get_bbox(
                current_colev,
                clid,
                self.frame_column,
                self.posx,
                self.posy,
                edge_size,
                self.output_order,
            )
            self.viewer.add_shapes(bbox, **bbox_param)
        else:
            timepoints = list(range(0, int(self.viewer.dims.range[0][1])))
            df_tp = pd.DataFrame(timepoints, columns=[self.frame_column])
            bbox_tuple = get_bbox_3d(
                current_colev,
                self.frame_column,
                self.posx,
                self.posy,
                self.posz,
                self.output_order,
            )
            bbox_tuple = fix_3d_convex_hull(
                df_tp, bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], self.frame_column
            )
            self.viewer.add_surface(
                bbox_tuple,
                colormap="red",
                opacity=0.15,
                name=ARCOS_LAYERS["event_boundingbox"],
                shading="none",
            )

        current_step = list(self.viewer.dims.current_step)
        t_index = self.output_order.index("t")
        current_step[t_index] = frame
        self.viewer.dims.current_step = current_step


class NoodlePlot(QtWidgets.QWidget):
    """
    QWidget for plotting.
    Class to make a matplotlib figure canvas and add it to a Qwidget.
    Canvas, figure and axis objects can be acessed by self.canvas,
    self.fig and self.ax. This plots tracks of Collective events over time.
    """

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        """
        QWidget for plotting a Noodleplot of Collective events.
        Tracks of objects are plotted and colored by collective event id.
        Make a matplotlib figure canvas and add it to a Qwidget.
        Canvas, figure and axis objects can be acessed by self.canvas,
        self.fig and self.ax.

        Attributes:
            viewer (napari.viewer.Viewer): Napari viewer instance
            parent (qtpy.QtWidgets.QWidget): Parent widget, optional
        """
        super().__init__(parent)
        self.viewer = viewer
        self.collid_name: str = "collid"
        self.nbr_collev: int = 0
        self.stats = pd.DataFrame(
            data={"total_size": [], "duration": [], self.collid_name: []}
        )
        self._callbacks: list = []
        self._init_mpl_widgets()
        self.arcos = pd.DataFrame(data={"t": [], "id": [], "x": [], "y": [], "z": []})
        self.point_size = 10
        self.frame_column = "frame"
        self.trackid_col = "id"
        self.posx = "x"
        self.posy = "y"
        self.posz = "z"
        self.projection_index = 3
        self.output_order = "tzyx"
        self.update_plot(
            self.frame_column,
            self.trackid_col,
            self.posx,
            self.posy,
            self.posz,
            self.arcos,
            self.output_order,
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.combo_box.currentIndexChanged.connect(self.update_plot_data)

    def _init_mpl_widgets(self):
        """
        Method to initialise a matplotlib figure canvas, to generate,
        set plot style and axis, and populate it with a matplotlib.figure.Figure.
        """
        # construct layout
        self.combo_box = QtWidgets.QComboBox(self)
        combobox_label = QtWidgets.QLabel(self)
        combobox_label.setText("Projection Axis")
        layout_noodle_plot = QtWidgets.QVBoxLayout()
        layout_combobox = QtWidgets.QHBoxLayout()
        layout_combobox.addWidget(combobox_label)
        layout_combobox.addWidget(self.combo_box)
        layout_noodle_plot.addLayout(layout_combobox)

        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(2.5, 1.5))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Position (px)")

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout_noodle_plot.addWidget(self.toolbar)
        layout_noodle_plot.addWidget(self.canvas)
        self.setLayout(layout_noodle_plot)
        self.setWindowTitle("Noodle Plot")
        self._update_layout()

    def _update_layout(self):
        self.fig.tight_layout(pad=0.1, w_pad=0.001, h_pad=0.05)

    def prepare_data(
        self,
        df: pd.DataFrame,
        colev: str,
        trackid: str,
        frame: str,
        posx: str,
        posy: str,
        posz: str,
    ):
        """From arcos collective event data,
        generates a list of numpy arrays, one for each event.

        Parameters:
            df (pd.DataFrame): DataFrame containing collective events from arcos.
            colev (str): Name of the collective event column in df.
            trackid (str): Name of the track column in df.
            frame: (str): Name of the frame column in df.
            posx (str): Name of the X coordinate column in df.
            posy (str): Name of the Y coordinate column in df.
            posz (str): Name of the Z coordinate column in df,
            or "None" (str) if no z column.

        Returns (list[np.ndarray], np.ndarray): List of collective events data,
        colors for each collective event.
        """
        col_fact = f"{trackid}_factorized"

        # factorize column in order to prevent numpy grouping error in detrending
        value, _ = df[trackid].factorize()
        df[col_fact] = value

        # values need to be sorted to group with numpy
        df.sort_values([colev, col_fact], inplace=True)
        if posz:
            array = df[[colev, col_fact, frame, posx, posy, posz]].to_numpy()
        else:
            array = df[[colev, col_fact, frame, posx, posy]].to_numpy()
        # generate goroups for each unique value
        grouped_array = np.split(
            array, np.unique(array[:, 0], axis=0, return_index=True)[1][1:]
        )
        # make collids sequential
        seq_colids = np.concatenate(
            [np.repeat(i, value.shape[0]) for i, value in enumerate(grouped_array)],
            axis=0,
        )
        array_seq_colids = np.column_stack((array, seq_colids))
        # split sequential collids array by trackid and collid
        grouped_array = np.split(
            array_seq_colids,
            np.unique(array_seq_colids[:, :2], axis=0, return_index=True)[1][1:],
        )
        # generate colors for each collective event, wrap arround the color cycle
        colors = np.take(
            np.array(COLOR_CYCLE), [i + 1 for i in np.unique(seq_colids)], mode="wrap"
        )
        return grouped_array, colors

    def calc_stats(self, frame_column, trackid_col):
        """Calculates stats for collective events."""
        # if no calculation was run so far (i.e. when the widget is initialized)
        # populate it with no data
        if not self.arcos.empty:
            self.stats = calculate_statistics(
                self.arcos, frame_column, self.collid_name, trackid_col
            )

    def clear_plot(self):
        """Method to clear the plot."""
        self.ax.cla()
        self.ax.set_xlabel("Total Size")
        self.ax.set_ylabel("Event Duration")
        self.fig.canvas.draw_idle()

    def update_plot(
        self,
        frame_column,
        trackid_col,
        posx,
        posy,
        posz,
        arcos_data,
        point_size=10,
        output_order="tzyx",
    ):
        """
        Method to update the matplotlibl axis object self.ax with new values from
        the stored_variables object and update the projection choices.

        Parameters:
            frame_column (str): The name of the frame column.
            trackid_col (str): The name of the trackid column.
            posx (str): The name of the x coordinate column.
            posy (str): The name of the y coordinate column.
            posz (str): The name of the z coordinate column.
            arcos_data (pd.DataFrame): DataFrame containing arcos output data.
            point_size (int): The size of the points drawn in napari.
        """
        self.arcos = arcos_data
        self.point_size = point_size
        self.frame_column = frame_column
        self.trackid_col = trackid_col
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.output_order = output_order
        if self.posz:
            projection_list = [self.posx, self.posy, self.posz]
        else:
            projection_list = [self.posx, self.posy]
        self.combo_box.clear()
        self.combo_box.addItems(projection_list)
        self.update_plot_data()

    def update_plot_data(self):
        """Update plot data."""
        # if no calculation was run so far (i.e. when the widget is initialized)
        # populate it with no data
        projection_type = self.combo_box.currentText()
        if projection_type == self.posx:
            self.projection_index = 3
        elif projection_type == self.posy:
            self.projection_index = 4
        elif projection_type == self.posz:
            self.projection_index = 5
        if not self.arcos.empty and self.trackid_col:
            self.dat_grpd, self.colors = self.prepare_data(
                self.arcos,
                "collid",
                self.trackid_col,
                self.frame_column,
                self.posx,
                self.posy,
                self.posz,
            )
            self.calc_stats(self.frame_column, self.trackid_col)
            self.ax.cla()
            self.ax.spines["bottom"].set_color("white")
            self.ax.spines["top"].set_color("white")
            self.ax.spines["right"].set_color("white")
            self.ax.spines["left"].set_color("white")
            self.ax.xaxis.label.set_color("white")
            self.ax.yaxis.label.set_color("white")
            self.ax.tick_params(colors="white", which="both")
            self.ax.axis("on")
            self.ax.set_xlabel("Time Point")
            self.ax.set_ylabel("Position")
            self.fig.canvas.draw_idle()
            for dat in self.dat_grpd:
                self.ax.plot(
                    dat[:, 2],
                    dat[:, self.projection_index],
                    c=self.colors[int(dat[0, -1])],
                    picker=1,
                )
                self.ax.set_xlim(0, int(self.viewer.dims.range[0][1]))

            self.nbr_collev = self.stats[self.collid_name].nunique()
        # generate empty annotation
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            bbox=dict(boxstyle="round", fc="#252932", ec="white", linewidth=0.3),
            fontsize=7,
            color="white",
            clip_on=True,
            animated=True,
        )
        self.annot.set_visible(False)
        # instantiate the BlitManager for faster rendering of the hover annotation.
        self.blint_manager = BlitManager(self.canvas, [self.annot])

    def update_annot(self, ind, line):
        """Update the annotation.

        Updates hover annotation showing collective event id.

        Parameters:
            ind (dict): Index of plotted data corresponding
            to the current mouse location.
            line (matplotlib.Artist.artist): Line artist where hover detected the event.
        """
        x_vals, y_vals = line.get_data()
        pos_text = [x_vals[ind["ind"][0]], y_vals[ind["ind"][0]]]
        clid_index = int(findall(r"\d+", line.get_label())[0])
        clid = int(self.dat_grpd[clid_index][0, 0])
        self.annot.xy = (x_vals[ind["ind"][0]], y_vals[ind["ind"][0]])
        text = f"id:{clid}"
        self.annot.set_text(text)
        # get size of the annotation bbox
        canvas_renderer = self.fig.canvas.get_renderer()
        bbox = self.annot.get_window_extent(renderer=canvas_renderer)
        bbox_data = self.ax.transData.inverted().transform(bbox)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        size_h = bbox_data[1][0] - bbox_data[0][0]
        size_v = bbox_data[1][1] - bbox_data[0][1]

        # fix for annotation drawn at the border of the plot (moves annotation)
        if pos_text[0] > (xlim[1] - size_h):
            pos_text[0] -= size_h
        if pos_text[1] > (ylim[1] - size_v):
            pos_text[1] -= size_v

        self.annot.xy = (x_vals[ind["ind"][0]], y_vals[ind["ind"][0]])
        self.annot.set_position(pos_text)
        self.annot.get_bbox_patch().set_alpha(1)

    def hover(self, event):
        """Display annotation of collective even on hover.

        Checks if current mouse position is over a datapoint. If yes,
        displays corresponding collective event id for the correct Line2D artist.
        """
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            # get line, or liens that was hovered over
            selected_line = [line for line in self.ax.lines if line.contains(event)[0]]
            if selected_line:
                # loop over all lines that are within hover radius
                for line in selected_line:
                    _, ind = line.contains(event)
                    self.update_annot(ind, line)
                    self.annot.set_visible(True)
                    # blitting for faster rendering of annotations.
                    self.blint_manager.update()
                    break  # break on first line with content
            else:
                if vis:
                    self.annot.set_visible(False)
                    # blitting for faster rendering of annotations.
                    self.blint_manager.update()

    def on_pick(self, event):
        """Displays the selected collective event in the napari viewer.

        On pick gets collective event id from picked datapoint, gets
        the correspondig starting frame, moves to this and
        draws a bounding box arround the extends of the collective event.

        Parameters:
            event (matplotlib_pick_event): event generated from selecting a datapoint.
        """
        clid_index = int(findall(r"\d+", event.artist.get_label())[0])
        clid = int(self.dat_grpd[clid_index][0, 0])
        current_colev = self.arcos[self.arcos["collid"] == clid]
        edge_size = self.point_size / 5
        frame = int(
            self.stats[self.stats[self.collid_name] == clid]["first_timepoint"].iloc[0]
        )
        if ARCOS_LAYERS["event_boundingbox"] in self.viewer.layers:
            self.viewer.layers.remove(ARCOS_LAYERS["event_boundingbox"])
        if self.posz is None:
            bbox, bbox_param = get_bbox(
                current_colev,
                clid,
                self.frame_column,
                self.posx,
                self.posy,
                edge_size,
                self.output_order,
            )
            self.viewer.add_shapes(bbox, **bbox_param)
        else:
            timepoints = list(range(0, int(self.viewer.dims.range[0][1])))
            df_tp = pd.DataFrame(timepoints, columns=[self.frame_column])
            bbox_tuple = get_bbox_3d(
                current_colev,
                self.frame_column,
                self.posx,
                self.posy,
                self.posz,
                self.output_order,
            )
            bbox_tuple = fix_3d_convex_hull(
                df_tp, bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], self.frame_column
            )
            self.viewer.add_surface(
                bbox_tuple,
                colormap="red",
                opacity=0.15,
                name=ARCOS_LAYERS["event_boundingbox"],
                shading="none",
            )

        current_step = list(self.viewer.dims.current_step)
        t_index = self.output_order.index("t")
        current_step[t_index] = frame
        self.viewer.dims.current_step = current_step


class BlitManager:
    """Class to manage blitting of the annotation. Copied from matplotlib docs."""

    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas (FigureCanvasAgg): The canvas to work with, this only works for
        sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists (Iterable[Artist]): List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for _artist in animated_artists:
            self.add_artist(_artist)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        _canvas = self.canvas
        if event is not None:
            if event.canvas != _canvas:
                raise RuntimeError
        self._bg = _canvas.copy_from_bbox(_canvas.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters:
            art (Artist): The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class TimeSeriesPlots(QtWidgets.QWidget):
    """
    QWidget for plotting.
    Class to make a matplotlib figure canvas and add it to a Qwidget.
    Canvas, figure and axis objects can be acessed by self.canvas, self.fig and self.ax.
    This plots several different Timeseries plots such as Position/t plots,
    tracklength histogram and a measurment density plot.

    Attributes:
        parent (qtpy.QtWidgets.QWidget): Parent widget, optional
    """

    def __init__(self, parent=None):
        """Constructs class with given arguments
        Parameters:
            parent (qtpy.QtWidgets.QWidget): Parent widget, optional
        """
        super().__init__(parent=parent)

        # available plots
        self.plot_list = [
            "tracklength histogram",
            "measurment density plot",
            "measurment density plot rescaled",
            "original vs detreded",
            "x/t-plot",
            "y/t-plot",
        ]
        # will be set later in methods
        self._dataframe = pd.DataFrame()
        self._dataframe_resc = pd.DataFrame()
        self._frame_col = None
        self._track_id_col = None
        self._x_coord_col = None
        self._y_coord_col = None
        self._measurement = None
        self._measurement_resc_col = None
        self._object_id_number = None

        self._init_widgets()

    def _init_widgets(self):
        """
        Method to initialise a matplotlib figure canvas as well as a spinbox,
        Button and label widgets. Additionally, generates a
        matplotlib.backends.backend_qt5agg.FigureCanvas, a set plot style and axis,
        and populates it with a matplotlib.figure.Figure.
        These are the added to a QVboxlayout.
        """
        # creating spinbox widget
        self.sample_number = QtWidgets.QSpinBox()
        self.sample_number.setMinimum(1)
        self.sample_number.setMaximum(200)
        self.sample_number.setValue(20)

        self.button = QtWidgets.QPushButton("Update Plot")
        self.resc_check = QtWidgets.QCheckBox("Show Rescaled")
        self.resc_check.setVisible(False)
        self.resc_check.setChecked(True)
        self.orig_check = QtWidgets.QCheckBox("Show Original")
        self.orig_check.setVisible(False)
        self.orig_check.setChecked(True)

        # label
        self.spinbox_title = QtWidgets.QLabel("Sample Size")
        self.spinbox_title.setVisible(False)

        # creating a combo box widget
        self.combo_box = QtWidgets.QComboBox(self)
        self.combo_box.addItems(self.plot_list)

        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(2.5, 1.5))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, self)

        # construct layout
        layout = QtWidgets.QVBoxLayout()
        layout_combobox = QtWidgets.QVBoxLayout()
        layout_spinbox = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        # add widgets to sub_layouts
        layout_combobox.addWidget(self.button)
        layout_combobox.addWidget(self.combo_box)

        layout_spinbox.addWidget(self.spinbox_title)
        layout_spinbox.addWidget(self.sample_number)
        layout_spinbox.addWidget(self.resc_check)
        layout_spinbox.addWidget(self.orig_check)

        layout.addLayout(layout_combobox)
        layout.addLayout(layout_spinbox)

        # add sublayouts together
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setWindowTitle("Collective Events")
        self.combo_box.currentIndexChanged.connect(self._update)
        self._update_layout()
        self.button.clicked.connect(self._update_from_button)
        self.orig_check.stateChanged.connect(self._update)
        self.resc_check.stateChanged.connect(self._update)

    def _update_layout(self):
        self.fig.tight_layout(pad=0.1, w_pad=0.001, h_pad=0.05)

    def data_clear(self):
        """
        Method to clear the data from the plot.
        """
        self.ax.clear()
        self._dataframe = pd.DataFrame()
        self._dataframe_resc = pd.DataFrame()
        self._frame_col = None
        self._track_id_col = None
        self._x_coord_col = None
        self._y_coord_col = None
        self._measurement = None
        self._measurement_resc_col = None
        self._object_id_number = None

    def update_plot(
        self,
        dataframe: pd.DataFrame,
        dataframe_resc: pd.DataFrame,
        frame_column,
        track_id_col,
        x_coord_col,
        y_coord_col,
        measurement_col,
        measurement_resc_col,
        object_id_number=None,
    ):
        """
        Method to update the from the dropdown menu chosen
        matplotlibl plot with values from
        the stored_variables object dataframe.
        """
        self._dataframe = dataframe
        self._dataframe_resc = dataframe_resc
        self._frame_col = frame_column
        self._track_id_col = track_id_col
        self._x_coord_col = x_coord_col
        self._y_coord_col = y_coord_col
        self._measurement = measurement_col
        self._measurement_resc_col = measurement_resc_col
        self._object_id_number = object_id_number
        self._update()

    def _update_from_button(self):
        self._object_id_number = None
        self._update()

    def _update(self):
        # return plottype that should be plotted
        plottype = self.combo_box.currentText()
        # sample number for position/t-plots
        n_samples = self.sample_number.value()

        # check if some data was loaded already, otherwise do nothing
        if not self._dataframe.empty:
            self._init_empty_plot()

            # tracklength histogram
            if plottype == "tracklength histogram":
                self._tracklenght_histogram()

            # measurment density plot, kde
            elif plottype == "measurment density plot":
                self._meas_density_plot()

            elif plottype == "measurment density plot rescaled":
                self._meas_density_plot_rescaled()

            # xy/t plots
            elif plottype == "x/t-plot":
                self._xt_plot(n_samples)

            elif plottype == "y/t-plot":
                self._yt_plot(n_samples)

            elif plottype == "original vs detreded":
                self._orig_vs_detrended_plot()

            self.fig.canvas.draw_idle()
        else:
            show_info("No Data to plot")

    def _orig_vs_detrended_plot(self):
        self.sample_number.setVisible(False)
        self.spinbox_title.setVisible(False)

        self.resc_check.setVisible(True)
        self.orig_check.setVisible(True)

        if not self._dataframe_resc.empty and self._track_id_col:
            if self._object_id_number:
                vals = self._object_id_number
            else:
                vals = np.random.choice(
                    self._dataframe_resc[self._track_id_col].unique(),
                    1,
                    replace=False,
                )
                self._object_id_number = vals
            dataframe_resc_cp = (
                self._dataframe_resc.set_index(self._track_id_col)
                .loc[vals]
                .reset_index()
                .copy(deep=True)
            )
            grouped = dataframe_resc_cp.groupby(self._track_id_col)
            plot_data_types = []
            if self.resc_check.isChecked():
                plot_data_types.append(self._measurement_resc_col)
            if self.orig_check.isChecked():
                plot_data_types.append(self._measurement)
            for val in vals:
                df_g = grouped.get_group(val)
                if not plot_data_types:
                    self.ax.plot()
                else:
                    df_g.plot(
                        x=self._frame_col,
                        y=plot_data_types,
                        ax=self.ax,
                    )
                    x_val = df_g[df_g[f"{self._measurement}.bin"] != 0][self._frame_col]
                    y_val = np.repeat(self.ax.get_ylim()[0], x_val.size)
                    indices = np.where(np.diff(x_val) != 1)[0] + 1
                    x_split = np.split(x_val, indices)
                    y_split = np.split(y_val, indices)
                    for idx, (x_val, y_val) in enumerate(zip(x_split, y_split)):
                        if idx == 0:
                            self.ax.plot(x_val, y_val, color="red", lw=2, label="bin")
                        else:
                            self.ax.plot(x_val, y_val, color="red", lw=2)
            if plot_data_types:
                self.ax.legend(loc=2, prop={"size": 6})
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("Mes Value")

    def _yt_plot(self, n_samples):
        self.sample_number.setVisible(True)
        self.spinbox_title.setVisible(True)
        if self._track_id_col:
            sample = pd.Series(self._dataframe[self._track_id_col].unique()).sample(
                n_samples, replace=True
            )
            pd_from_r_df = self._dataframe.loc[
                self._dataframe[self._track_id_col].isin(sample)
            ]
            df_grp = pd_from_r_df.groupby(self._track_id_col)
            for _, df in df_grp:
                self.ax.plot(df[self._frame_col], df[self._y_coord_col])
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Position Y")

    def _xt_plot(self, n_samples):
        self.sample_number.setVisible(True)
        self.spinbox_title.setVisible(True)
        self.resc_check.setVisible(False)
        self.orig_check.setVisible(False)
        if self._track_id_col:
            sample = pd.Series(self._dataframe[self._track_id_col].unique()).sample(
                n_samples, replace=True
            )
            pd_from_r_df = self._dataframe.loc[
                self._dataframe[self._track_id_col].isin(sample)
            ]
            df_grp = pd_from_r_df.groupby(self._track_id_col)
            for _, df in df_grp:
                self.ax.plot(df[self._frame_col], df[self._x_coord_col])
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Position X")

    def _meas_density_plot_rescaled(self):
        if not self._dataframe_resc.empty:
            measurement_resc_values = self._dataframe_resc[
                self._measurement_resc_col
            ].interpolate()
            if measurement_resc_values.size != 0:
                self.sample_number.setVisible(False)
                self.spinbox_title.setVisible(False)
                self.resc_check.setVisible(False)
                self.orig_check.setVisible(False)
                density = gaussian_kde(measurement_resc_values)
                x_val = np.linspace(
                    min(measurement_resc_values),
                    max(measurement_resc_values),
                    100,
                )
                y_val = density(x_val)
                self.ax.plot(x_val, y_val)
                self.ax.set_xlabel("measurement values")
                self.ax.set_ylabel("density")

    def _meas_density_plot(self):
        self.sample_number.setVisible(False)
        self.spinbox_title.setVisible(False)
        self.resc_check.setVisible(False)
        self.orig_check.setVisible(False)
        density = gaussian_kde(self._dataframe[self._measurement].interpolate())
        x_val = np.linspace(
            min(self._dataframe[self._measurement]),
            max(self._dataframe[self._measurement]),
            100,
        )
        y_val = density(x_val)
        self.ax.plot(x_val, y_val)
        self.ax.set_xlabel("measurement values")
        self.ax.set_ylabel("density")

    def _tracklenght_histogram(self):
        self.sample_number.setVisible(False)
        self.spinbox_title.setVisible(False)
        self.resc_check.setVisible(False)
        self.orig_check.setVisible(False)
        if self._track_id_col:
            track_length = self._dataframe.groupby(self._track_id_col).size()
            self.ax.hist(track_length)
        self.ax.set_xlabel("tracklength")
        self.ax.set_ylabel("counts")

    def _init_empty_plot(self):
        self.ax.cla()
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.tick_params(colors="white", which="both")
        self.ax.axis("on")
