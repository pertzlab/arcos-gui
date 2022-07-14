from re import findall
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from arcos4py.tools import calcCollevStats
from arcos_gui.shape_functions import (
    COLOR_CYCLE,
    fix_3d_convex_hull,
    get_bbox,
    get_bbox_3d,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.notifications import show_info
from qtpy import QtWidgets
from scipy.stats import gaussian_kde

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
        self.frame_col = "time"
        self.trackid_col = "id"
        self.posx = "x"
        self.posy = "y"
        self.posz = "z"
        self.update_plot(
            self.frame_col,
            self.trackid_col,
            self.posx,
            self.posy,
            self.posz,
            self.arcos,
        )

    def _init_mpl_widgets(self):
        """
        Method to initialise a matplotlib figure canvas, to generate,
        set plot style and axis, and populate it with a matplotlib.figure.Figure.
        """
        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["figure.dpi"] = 110
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(3, 1.5))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.scatter([], [])
            self.ax.set_xlabel("Total Size")
            self.ax.set_ylabel("Event Duration")
            self.canvas.figure.tight_layout()

        self.toolbar = NavigationToolbar(self.canvas, self)

        # construct layout
        self.layout_collevplot = QtWidgets.QVBoxLayout()
        self.layout_collevplot.addWidget(self.toolbar)
        self.layout_collevplot.addWidget(self.canvas)
        self.setLayout(self.layout_collevplot)
        self.setWindowTitle("Collective Events")

    def update_plot(
        self, frame_col, trackid_col, posx, posy, posz, arcos_data, point_size=10
    ):
        """
        Method to update the matplotlibl axis object self.ax with new values from
        the stored_variables object
        """
        collev_stats = calcCollevStats()
        self.arcos = arcos_data
        self.point_size = point_size
        self.frame_col = frame_col
        self.trackid_col = trackid_col
        self.posx = posx
        self.posy = posy
        self.posz = posz
        # if no calculation was run so far (i.e. when the widget is initialized)
        # populate it with no data else calculate stats for collective events,
        # generate plot with data
        if not self.arcos.empty:
            self.stats = collev_stats.calculate(
                self.arcos,
                self.frame_col,
                self.collid_name,
                self.trackid_col,
            )

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
                cmap=COLOR_CYCLE,
                picker=True,
            )
            self.ax.set_xlabel("Total Size")
            self.ax.set_ylabel("Event Duration")
            self.fig.canvas.draw_idle()
            self.nbr_collev = self.stats.shape[0]
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
        self.bm = BlitManager(self.canvas, [self.annot])

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
            cont, ind = self.ax.collections[0].contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                # blitting to improve performance.
                self.bm.update()
            else:
                if vis:
                    self.annot.set_visible(False)
                    # blitting to improve performance.
                    self.bm.update()

    def on_pick(self, event):
        """Displays the selected collective event in the napari viewer.

        On pick gets collective event id from picked datapoint, gets
        the correspondig starting frame, moves to this and
        draws a bounding box arround the extends of the collective event.

        Parameters:
            event (matplotlib_pick_event): event generated from selecting a datapoint.
        """
        ind = event.ind
        clid = int(self.stats.iloc[ind[0]][0])
        current_colev = self.arcos[self.arcos["collid"] == clid]
        edge_size = self.point_size.value() / 5
        frame = self.stats.iloc[ind[0]][5]
        if "event_boundingbox" in self.viewer.layers:
            self.viewer.layers.remove("event_boundingbox")
        if self.posz == "None":
            bbox, bbox_param = get_bbox(
                current_colev, clid, self.frame_col, self.posx, self.posy, edge_size
            )
            self.viewer.add_shapes(bbox, **bbox_param)
        else:
            timepoints = [i for i in range(0, int(self.viewer.dims.range[0][1]))]
            df_tp = pd.DataFrame(timepoints, columns=[self.frame_col])
            bbox_tuple = get_bbox_3d(
                current_colev, self.frame_col, self.posx, self.posy, self.posz
            )
            bbox_tuple = fix_3d_convex_hull(
                df_tp, bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], self.frame_col
            )
            self.viewer.add_surface(
                bbox_tuple,
                colormap="red",
                opacity=0.15,
                name="event_boundingbox",
                shading="none",
            )

        if len(self.viewer.dims.current_step) == 3:
            t, y, x = self.viewer.dims.current_step
            self.viewer.dims.current_step = (frame, y, x)
        elif len(self.viewer.dims.current_step) == 4:
            t, y, x, z = self.viewer.dims.current_step
            self.viewer.dims.current_step = (frame, y, x, z)


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
        self.frame_col = "time"
        self.trackid_col = "id"
        self.posx = "x"
        self.posy = "y"
        self.posz = "z"
        self.projection_index = 3
        self.update_plot(
            self.frame_col,
            self.trackid_col,
            self.posx,
            self.posy,
            self.posz,
            self.arcos,
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.combo_box.currentIndexChanged.connect(self.update_plot_data)

    def _init_mpl_widgets(self):
        """
        Method to initialise a matplotlib figure canvas, to generate,
        set plot style and axis, and populate it with a matplotlib.figure.Figure.
        """
        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["figure.dpi"] = 110
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(3, 1.5))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.plot([], [])
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Position coordinate")
            self.canvas.figure.tight_layout()

        self.toolbar = NavigationToolbar(self.canvas, self)
        # construct layout
        self.combo_box = QtWidgets.QComboBox(self)
        combobox_label = QtWidgets.QLabel(self)
        combobox_label.setText("Projection Axis")
        layout_noodle_plot = QtWidgets.QVBoxLayout()
        layout_combobox = QtWidgets.QHBoxLayout()
        layout_combobox.addWidget(combobox_label)
        layout_combobox.addWidget(self.combo_box)
        layout_noodle_plot.addWidget(self.toolbar)
        layout_noodle_plot.addLayout(layout_combobox)
        layout_noodle_plot.addWidget(self.canvas)
        self.setLayout(layout_noodle_plot)
        self.setWindowTitle("Noodle Plot")

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
        value, label = df[trackid].factorize()
        df[col_fact] = value

        # values need to be sorted to group with numpy
        df.sort_values([colev, col_fact], inplace=True)
        if posz != "None":
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

    def calc_stats(self, frame_col, trackid_col):
        """Calculates stats for collective events."""
        collev_stats = calcCollevStats()
        # if no calculation was run so far (i.e. when the widget is initialized)
        # populate it with no data
        if not self.arcos.empty:
            self.stats = collev_stats.calculate(
                self.arcos,
                frame_col,
                self.collid_name,
                trackid_col,
            )

    def update_plot(
        self, frame_col, trackid_col, posx, posy, posz, arcos_data, point_size=10
    ):
        """
        Method to update the matplotlibl axis object self.ax with new values from
        the stored_variables object and update the projection choices.

        Parameters:
            frame_col (str): The name of the frame column.
            trackid_col (str): The name of the trackid column.
            posx (str): The name of the x coordinate column.
            posy (str): The name of the y coordinate column.
            posz (str): The name of the z coordinate column.
            arcos_data (pd.DataFrame): DataFrame containing arcos output data.
            point_size (int): The size of the points drawn in napari.
        """
        self.arcos = arcos_data
        self.point_size = point_size
        self.frame_col = frame_col
        self.trackid_col = trackid_col
        self.posx = posx
        self.posy = posy
        self.posz = posz
        if self.posz != "None":
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
        if not self.arcos.empty:
            self.dat_grpd, self.colors = self.prepare_data(
                self.arcos,
                "collid",
                self.trackid_col,
                self.frame_col,
                self.posx,
                self.posy,
                self.posz,
            )
            self.calc_stats(self.frame_col, self.trackid_col)
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

            self.nbr_collev = self.stats.shape[0]
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
        self.bm = BlitManager(self.canvas, [self.annot])

    def update_annot(self, ind, line):
        """Update the annotation.

        Updates hover annotation showing collective event id.

        Parameters:
            ind (dict): Index of plotted data corresponding
            to the current mouse location.
            line (matplotlib.Artist.artist): Line artist where hover detected the event.
        """
        x, y = line.get_data()
        pos_text = [x[ind["ind"][0]], y[ind["ind"][0]]]
        clid_index = int(findall(r"\d+", line.get_label())[0])
        clid = int(self.dat_grpd[clid_index][0, 0])
        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = f"id:{clid}"
        self.annot.set_text(text)
        # get size of the annotation bbox
        r = self.fig.canvas.get_renderer()
        bbox = self.annot.get_window_extent(renderer=r)
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

        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
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
                    cont, ind = line.contains(event)
                    self.update_annot(ind, line)
                    self.annot.set_visible(True)
                    # blitting for faster rendering of annotations.
                    self.bm.update()
                    break  # break on first line with content
            else:
                if vis:
                    self.annot.set_visible(False)
                    # blitting for faster rendering of annotations.
                    self.bm.update()

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
        edge_size = self.point_size.value() / 5
        frame = int(self.stats[self.stats.iloc[:, 0] == clid].iloc[:, 5])
        if "event_boundingbox" in self.viewer.layers:
            self.viewer.layers.remove("event_boundingbox")
        if self.posz == "None":
            bbox, bbox_param = get_bbox(
                current_colev, clid, self.frame_col, self.posx, self.posy, edge_size
            )
            self.viewer.add_shapes(bbox, **bbox_param)
        else:
            timepoints = [i for i in range(0, int(self.viewer.dims.range[0][1]))]
            df_tp = pd.DataFrame(timepoints, columns=[self.frame_col])
            bbox_tuple = get_bbox_3d(
                current_colev, self.frame_col, self.posx, self.posy, self.posz
            )
            bbox_tuple = fix_3d_convex_hull(
                df_tp, bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], self.frame_col
            )
            self.viewer.add_surface(
                bbox_tuple,
                colormap="red",
                opacity=0.15,
                name="event_boundingbox",
                shading="none",
            )

        if len(self.viewer.dims.current_step) == 3:
            t, y, x = self.viewer.dims.current_step
            self.viewer.dims.current_step = (frame, y, x)
        elif len(self.viewer.dims.current_step) == 4:
            t, y, x, z = self.viewer.dims.current_step
            self.viewer.dims.current_step = (frame, y, x, z)


class BlitManager:
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

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
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
            "x/t-plot",
            "y/t-plot",
        ]
        self._init_widgets()
        self.parent_gui = self.parent()

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

        # label
        self.spinbox_title = QtWidgets.QLabel("Sample Size")
        self.spinbox_title.setVisible(False)

        # creating a combo box widget
        self.combo_box = QtWidgets.QComboBox(self)
        self.combo_box.addItems(self.plot_list)
        # self.combo_box.currentIndexChanged.connect(self.update_plot)

        # set up figure and axe objects
        with plt.style.context("dark_background"):
            plt.rcParams["figure.dpi"] = 110
            plt.rcParams["axes.edgecolor"] = "#ffffff"
            self.fig = Figure(figsize=(3, 2))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.scatter([], [])
            self.ax.set_xlabel("X Axis")
            self.ax.set_ylabel("Y Axis")
            self.canvas.figure.tight_layout()

        self.toolbar = NavigationToolbar(self.canvas, self)

        # construct layout
        layout = QtWidgets.QVBoxLayout()
        layout_combobox = QtWidgets.QVBoxLayout()
        layout_spinbox = QtWidgets.QHBoxLayout()

        # add widgets to sub_layouts
        layout_combobox.addWidget(self.button)
        layout_combobox.addWidget(self.combo_box)
        layout_spinbox.addWidget(self.spinbox_title)
        layout_spinbox.addWidget(self.sample_number)

        # add sublayouts together
        layout.addWidget(self.toolbar)
        layout.addLayout(layout_combobox)
        layout.addLayout(layout_spinbox)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setWindowTitle("Collective Events")

    def update_plot(self, columnpicker_widget, dataframe: pd.DataFrame):
        """
        Method to update the from the dropdown menu chosen
        matplotlibl plot with values from
        the stored_variables object dataframe.
        """
        self.measurement = self.parent_gui.measurement

        # return plottype that should be plotted
        plottype = self.combo_box.currentText()
        # sample number for position/t-plots
        n = self.sample_number.value()

        # get column values and dataframe
        frame = columnpicker_widget.frame.value
        track_id = columnpicker_widget.track_id.value
        x_coordinates = columnpicker_widget.x_coordinates.value
        y_coordinates = columnpicker_widget.y_coordinates.value
        # measurement = columnpicker_widget.measurment.value

        # check if some data was loaded already, otherwise do nothing
        if not dataframe.empty:
            self.ax.cla()
            self.ax.spines["bottom"].set_color("white")
            self.ax.spines["top"].set_color("white")
            self.ax.spines["right"].set_color("white")
            self.ax.spines["left"].set_color("white")
            self.ax.xaxis.label.set_color("white")
            self.ax.yaxis.label.set_color("white")
            self.ax.tick_params(colors="white", which="both")
            self.ax.axis("on")

            # tracklength histogram
            if plottype == "tracklength histogram":
                self.sample_number.setVisible(False)
                self.spinbox_title.setVisible(False)
                track_length = dataframe.groupby(track_id).size()
                self.ax.hist(track_length)
                self.ax.set_xlabel("tracklength")
                self.ax.set_ylabel("counts")

            # measurment density plot, kde
            elif plottype == "measurment density plot":
                self.sample_number.setVisible(False)
                self.spinbox_title.setVisible(False)
                density = gaussian_kde(dataframe[self.measurement].interpolate())
                x = np.linspace(
                    min(dataframe[self.measurement]),
                    max(dataframe[self.measurement]),
                    100,
                )
                y = density(x)
                self.ax.plot(x, y)
                self.ax.set_xlabel("measurement values")
                self.ax.set_ylabel("density")

            elif plottype == "measurment density plot rescaled":
                try:
                    self.measurement_resc = self.parent_gui.start_arcos.resc_col
                    self.measurement_resc_values = self.parent_gui.start_arcos.data[
                        self.measurement_resc
                    ]
                    self.sample_number.setVisible(False)
                    self.spinbox_title.setVisible(False)
                    density = gaussian_kde(self.measurement_resc_values.interpolate())
                    x = np.linspace(
                        min(self.measurement_resc_values),
                        max(self.measurement_resc_values),
                        100,
                    )
                    y = density(x)
                    self.ax.plot(x, y)
                    self.ax.set_xlabel("measurement values")
                    self.ax.set_ylabel("density")
                except AttributeError:
                    pass

            # xy/t plots
            elif plottype == "x/t-plot":
                self.sample_number.setVisible(True)
                self.spinbox_title.setVisible(True)
                sample = pd.Series(dataframe[track_id].unique()).sample(n, replace=True)
                pd_from_r_df = dataframe.loc[dataframe[track_id].isin(sample)]
                df_grp = pd_from_r_df.groupby(track_id)
                for label, df in df_grp:
                    self.ax.plot(df[frame], df[x_coordinates])
                self.ax.set_xlabel("Frame")
                self.ax.set_ylabel("Position X")

            elif plottype == "y/t-plot":
                self.sample_number.setVisible(True)
                self.spinbox_title.setVisible(True)
                sample = pd.Series(dataframe[track_id].unique()).sample(n, replace=True)
                pd_from_r_df = dataframe.loc[dataframe[track_id].isin(sample)]
                df_grp = pd_from_r_df.groupby(track_id)
                for label, df in df_grp:
                    self.ax.plot(df[frame], df[y_coordinates])
                self.ax.set_xlabel("Frame")
                self.ax.set_ylabel("Position Y")
            self.fig.canvas.draw_idle()
        else:
            show_info("No Data to plot")
