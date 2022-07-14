from copy import deepcopy
from os import sep
from pathlib import Path
from typing import TYPE_CHECKING

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory, magicgui
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info
from qtpy import QtWidgets, uic
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from superqt import QDoubleRangeSlider, QRangeSlider

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer

# local imports
from arcos4py import ARCOS
from arcos4py.tools import filterCollev
from arcos_gui._plots import CollevPlotter, NoodlePlot, TimeSeriesPlots
from arcos_gui.data_module import process_input, read_data_header
from arcos_gui.export_movie import iterate_over_frames, resize_napari
from arcos_gui.magic_guis import (
    OPERATOR_DICTIONARY,
    columnpicker,
    show_timestamp_options,
    timestamp_options,
    toggle_visible_second_measurment,
)
from arcos_gui.plot_dialog import CollevPlotDialog, DataPlot
from arcos_gui.shape_functions import (
    COLOR_CYCLE,
    fix_3d_convex_hull,
    get_verticesHull,
    make_surface_3d,
    make_timestamp,
)
from arcos_gui.temp_data_storage import data_storage
from napari.utils import Colormap

# icons
ICONS = Path(__file__).parent / "_icons"
browse_file_icon = QIcon(str(ICONS / "folder-open-line.svg"))
expand_plot_icon = QIcon(str(ICONS / "enlarge_window.png"))

# initalize class
stored_variables = data_storage()
tab20 = Colormap(COLOR_CYCLE, "tab20", interpolation="zero")


class _MainUI:

    UI_FILE = str(Path(__file__).parent / "_ui" / "ARCOS_widget.ui")

    # The UI_FILE above contains these objects:
    frame_interval_label: QtWidgets.QLabel
    min_tracklength_label: QtWidgets.QLabel
    max_tracklength_label: QtWidgets.QLabel
    rescale_meas_label: QtWidgets.QLabel
    position_label: QtWidgets.QLabel
    file_LineEdit: QtWidgets.QLineEdit
    open_file_button: QtWidgets.QPushButton
    browse_file: QtWidgets.QPushButton
    position: QtWidgets.QComboBox
    additional_filter_combobox: QtWidgets.QComboBox
    additional_filter_combobox_label: QtWidgets.QLabel
    frame_interval: QtWidgets.QSpinBox
    rescale_measurment: QtWidgets.QSpinBox
    min_tracklength: QtWidgets.QSlider
    min_tracklength_spinbox: QtWidgets.QDoubleSpinBox
    max_tracklength: QtWidgets.QSlider
    max_tracklength_spinbox: QtWidgets.QDoubleSpinBox
    filter_groupBox: QtWidgets.QGroupBox
    filter_input_data: QtWidgets.QPushButton
    horizontalLayout_tracklength: QtWidgets.QHBoxLayout

    clip_low_label: QtWidgets.QLabel
    clip_high_label: QtWidgets.QLabel
    intervale_type_label: QtWidgets.QLabel
    bias_method_label: QtWidgets.QLabel
    smooth_k_label: QtWidgets.QLabel
    bias_k_label: QtWidgets.QLabel
    polyDeg_label: QtWidgets.QLabel
    bin_peak_threshold_label: QtWidgets.QLabel
    bin_threshold_label: QtWidgets.QLabel
    neighbourhood_label: QtWidgets.QLabel
    min_clustersize_label: QtWidgets.QLabel
    nprev_spinbox: QtWidgets.QSpinBox
    nprev_spinbox_label: QtWidgets.QLabel
    min_dur_label: QtWidgets.QLabel
    tot_size_label: QtWidgets.QLabel

    interpolate_meas: QtWidgets.QCheckBox
    clip_meas: QtWidgets.QCheckBox
    clip_low: QtWidgets.QDoubleSpinBox
    clip_high: QtWidgets.QDoubleSpinBox
    bias_method: QtWidgets.QComboBox
    smooth_k: QtWidgets.QComboBox
    bias_k: QtWidgets.QSpinBox
    polyDeg: QtWidgets.QSpinBox
    bin_peak_threshold: QtWidgets.QDoubleSpinBox
    bin_threshold: QtWidgets.QDoubleSpinBox
    neighbourhood_size: QtWidgets.QSpinBox
    min_clustersize: QtWidgets.QSpinBox
    min_dur: QtWidgets.QSpinBox
    total_event_size: QtWidgets.QSpinBox
    Progress: QtWidgets.QProgressBar
    update_arcos: QtWidgets.QPushButton
    arcos_group: QtWidgets.QGroupBox
    clip_frame: QtWidgets.QFrame
    add_convex_hull_checkbox: QtWidgets.QCheckBox

    point_size_label: QtWidgets.QLabel
    select_lut_label: QtWidgets.QLabel
    max_lut_mapping_label: QtWidgets.QLabel
    min_lut_mapping_label: QtWidgets.QLabel
    point_size: QtWidgets.QDoubleSpinBox
    LUT: QtWidgets.QComboBox
    max_lut: QtWidgets.QSlider
    max_lut_spinbox: QtWidgets.QDoubleSpinBox
    min_lut: QtWidgets.QSlider
    min_lut_spinbox: QtWidgets.QDoubleSpinBox
    reset_lut: QtWidgets.QPushButton
    layer_properties: QtWidgets.QGroupBox
    horizontalLayout_lut: QtWidgets.QHBoxLayout

    collevplot_goupbox: QtWidgets.QGroupBox
    timeseriesplot_groupbox: QtWidgets.QGroupBox
    evplot_layout: QtWidgets.QVBoxLayout
    evplot_layout_2: QtWidgets.QVBoxLayout
    tsplot_layout: QtWidgets.QVBoxLayout
    nbr_collev_display: QtWidgets.QLCDNumber
    expand_plot: QtWidgets.QPushButton
    expand_plot_2: QtWidgets.QPushButton

    def setup_ui(self):
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file

        self.LUT.addItems(stored_variables.colormaps)
        self.LUT.setCurrentText("RdYlBu_r")

        # set text of Line edit
        self.file_LineEdit.setText(".")


class MainWindow(QtWidgets.QWidget, _MainUI):
    """
    Widget allowing a user to import a csv file, filter this file,
    choose arcos parameters, choose LUT mappings aswell as shape sizes.
    """

    def __init__(self, viewer: napari.viewer.Viewer, remote=True):
        """Constructs class with provided arguments."""
        super().__init__()
        self.viewer: napari.viewer.Viewer = viewer
        self.setup_ui()
        self._filename: str = self.file_LineEdit.text()
        self.layers_to_create: list = []
        self.what_to_run: set = set()
        self.data: pd.DataFrame = pd.DataFrame()
        self.filtered_data: pd.DataFrame = pd.DataFrame()
        self.arcos_filtered: pd.DataFrame = pd.DataFrame()
        self.measurement = "None"
        self.timeseriesplot = TimeSeriesPlots(parent=self)
        self.noodle_plot = NoodlePlot(parent=self, viewer=self.viewer)
        self.collevplot = CollevPlotter(parent=self, viewer=self.viewer)
        self.plot_dialog_collev = CollevPlotDialog(parent=self)
        self.plot_dialog_data = DataPlot(parent=self)
        self._add_plot_widgets()
        self._init_ranged_sliderts()
        self.browse_file.setIcon(browse_file_icon)
        self.expand_plot.setIcon(expand_plot_icon)
        self.expand_plot_2.setIcon(expand_plot_icon)
        self._set_default_visible()

        self._init_callbacks_for_whattorun()
        self._init_columnpicker_callbacks()
        self._init_callback_for_sample_data()
        self._init_size_contrast_callbacks()
        self._connect_ranged_sliders_to_spinboxes()
        self._connect_pushbutton_callbacks()
        self._init_callbacks_visible_arcosparameters()
        self._init_plot_callbacks()
        self._init_columns()
        self.expand_plot.clicked.connect(self._open_plot)
        self.expand_plot_2.clicked.connect(self._open_data_plot)

    def _add_plot_widgets(self):
        """Add the plot widgets to the main window."""
        self.evplot_layout.addWidget(self.collevplot)
        self.evplot_layout_2.addWidget(self.noodle_plot)
        self.tsplot_layout.addWidget(self.timeseriesplot)

    def _open_plot(self):
        self.plot_dialog_collev.closeEvent = self._close_plot
        self.plot_dialog_collev.evplot_layout.addWidget(self.collevplot)
        self.plot_dialog_collev.evplot_layout_2.addWidget(self.noodle_plot)
        self.plot_dialog_collev.show()
        self.collevplot_goupbox.hide()

    def _open_data_plot(self):
        self.plot_dialog_data.closeEvent = self._close_plot_2
        self.plot_dialog_data.tsplot_layout.addWidget(self.timeseriesplot)
        self.plot_dialog_data.show()
        self.timeseriesplot_groupbox.hide()

    def _close_plot(self, event):
        self.evplot_layout.addWidget(self.collevplot)
        self.evplot_layout_2.addWidget(self.noodle_plot)
        self.collevplot_goupbox.show()

    def _close_plot_2(self, event):
        self.tsplot_layout.addWidget(self.timeseriesplot)
        self.timeseriesplot_groupbox.show()

    def _ts_plot_update(self):
        """Updates the ts-plot with new data."""
        self.timeseriesplot.update_plot(columnpicker, self.filtered_data)

    def _init_plot_callbacks(self):
        """Initializes the plot callbacks."""
        self.timeseriesplot.combo_box.currentIndexChanged.connect(self._ts_plot_update)
        self.timeseriesplot.button.clicked.connect(self._ts_plot_update)

    def collev_plot_update(self):
        """Updates the collective event plots and the collecitve
        events counter with new data."""
        self.collevplot.update_plot(
            self.frame,
            self.track_id,
            self.x_coordinates,
            self.y_coordinates,
            self.z_coordinates,
            self.arcos_filtered,
            point_size=self.point_size,
        )
        self.noodle_plot.update_plot(
            self.frame,
            self.track_id,
            self.x_coordinates,
            self.y_coordinates,
            self.z_coordinates,
            self.arcos_filtered,
            point_size=self.point_size,
        )
        self.nbr_collev_display.display(self.collevplot.nbr_collev)

    def _init_callbacks_visible_arcosparameters(self):
        """Initialize the callbacks for visible parameters in bias method groupbox."""
        self.bias_method.currentIndexChanged.connect(
            self.toggle_bias_method_parameter_visibility
        )

    def _connect_pushbutton_callbacks(self):
        """Connect push button callbacks in the widget to corresponding methods."""
        # callback to open file dialog
        self.browse_file.clicked.connect(self.browse_files)
        # callback for updating what to run in arcos_widget
        self.open_file_button.clicked.connect(self.open_columnpicker)
        # reset what to run
        self.filter_input_data.clicked.connect(self.after_filter_input_data)
        self.open_file_button.clicked.connect(self.what_to_run.clear)
        self.update_arcos.clicked.connect(self.reset_contrast)
        self.update_arcos.clicked.connect(self.run)

    def after_filter_input_data(self):
        """Collection of methods that run after filtering input data."""
        self.update_what_to_run_all()
        self.filter_data()
        self.reset_contrast()
        self.set_point_size()

    def _connect_ranged_sliders_to_spinboxes(self):
        """Method to connect ranged sliders to spinboxes to sync values."""
        self.tracklenght_slider.valueChanged.connect(
            self.handleSlider_tracklength_ValueChange
        )
        self.min_tracklength_spinbox.valueChanged.connect(
            self.handle_min_tracklenght_box_ValueChange
        )
        self.max_tracklength_spinbox.valueChanged.connect(
            self.handle_max_tracklength_box_ValueChange
        )
        self.lut_slider.valueChanged.connect(self.handleSlider_lut_ValueChange)
        self.min_lut_spinbox.valueChanged.connect(self.handle_min_lut_box_ValueChange)
        self.max_lut_spinbox.valueChanged.connect(self.handle_max_lut_box_ValueChange)

    def _init_size_contrast_callbacks(self):
        """Connects various callbacks that correspond to size,
        contrast and lut changes."""
        # execute LUT and point functions
        self.reset_lut.clicked.connect(self.reset_contrast)
        # update size and LUT
        self.LUT.currentIndexChanged.connect(self.update_lut)
        self.lut_slider.valueChanged.connect(self.change_cell_colors)
        self.LUT.currentIndexChanged.connect(self.change_cell_colors)
        self.point_size.valueChanged.connect(self.change_cell_size)

    def _init_callback_for_sample_data(self):
        """Callback to load sample data via the napari sample data interface."""
        stored_variables.register_callback(self.callback_file_Linedit_text)

    def _init_columnpicker_callbacks(self):
        """Connect callbacks suppoed to run after OK press in columnpicker widget."""
        columnpicker.measurement_math.changed.connect(toggle_visible_second_measurment)
        columnpicker.Ok.changed.connect(self.close_columnpicker)
        columnpicker.Ok.changed.connect(self.set_positions)
        columnpicker.Ok.changed.connect(self.get_tracklengths)
        columnpicker.Ok.changed.connect(self.remove_layers_after_columnpicker)
        columnpicker.Ok.changed.connect(self.after_filter_input_data)

    def _init_callbacks_for_whattorun(self):
        """Connect callbacks for updating 'what to run'.

        This is used to run only the appropriate parts of the
        run_arcos method depending on the parameters changed.
        """
        self.clip_low.valueChanged.connect(self.update_what_to_run_all)
        self.clip_high.valueChanged.connect(self.update_what_to_run_all)
        self.smooth_k.valueChanged.connect(self.update_what_to_run_all)
        self.bias_k.valueChanged.connect(self.update_what_to_run_all)
        self.bin_threshold.valueChanged.connect(self.update_what_to_run_all)
        self.neighbourhood_size.valueChanged.connect(self.update_what_to_run_tracking)
        self.bin_peak_threshold.valueChanged.connect(self.update_what_to_run_all)
        self.min_clustersize.valueChanged.connect(self.update_what_to_run_tracking)
        self.nprev_spinbox.valueChanged.connect(self.update_what_to_run_tracking)
        self.min_dur.valueChanged.connect(self.update_what_to_run_filtering)
        self.total_event_size.valueChanged.connect(self.update_what_to_run_filtering)
        self.add_convex_hull_checkbox.stateChanged.connect(
            self.update_what_to_run_filtering
        )

    def _init_ranged_sliderts(self):
        """Initialize ranged sliders from superqt."""
        self.lut_slider = QDoubleRangeSlider(Qt.Horizontal)
        self.tracklenght_slider = QRangeSlider(Qt.Horizontal)
        self.horizontalLayout_lut.addWidget(self.lut_slider)
        self.horizontalLayout_tracklength.addWidget(self.tracklenght_slider)

        # set starting values
        self.lut_slider.setRange(0, 10)
        self.tracklenght_slider.setRange(0, 10)
        self.tracklenght_slider.setValue((0, 10))
        self.lut_slider.setValue((0, 10))

    def _init_columns(self):
        """Method that sets default values for columnpicker."""
        self.frame = "None"
        self.track_id = "None"
        self.x_coordinates = "None"
        self.y_coordinates = "None"
        self.z_coordinates = "None"
        self.first_measurement = "None"
        self.second_measurement = "None"
        self.field_of_view_id = "None"
        self.additional_filter_column_name = "None"

    def handleSlider_tracklength_ValueChange(self):
        """Method to handle trancklenght value changes."""
        slider_vals = self.tracklenght_slider.value()
        self.min_tracklength_spinbox.setValue(slider_vals[0])
        self.max_tracklength_spinbox.setValue(slider_vals[1])

    def handle_min_tracklenght_box_ValueChange(self, value):
        """Method to handle min tracklenght spinbox."""
        slider_vals = self.tracklenght_slider.value()
        self.tracklenght_slider.setValue((value, slider_vals[1]))

    def handle_max_tracklength_box_ValueChange(self, value):
        """Method to handle max tracklength spinbox."""
        slider_vals = self.tracklenght_slider.value()
        self.tracklenght_slider.setValue((slider_vals[0], value))

    def handleSlider_lut_ValueChange(self):
        """Method to handle lut value change."""
        slider_vals = self.lut_slider.value()
        self.min_lut_spinbox.setValue(slider_vals[0])
        self.max_lut_spinbox.setValue(slider_vals[1])

    def handle_min_lut_box_ValueChange(self, value):
        """Method to handle lut min spinbox value change."""
        slider_vals = self.lut_slider.value()
        self.lut_slider.setValue((value, slider_vals[1]))

    def handle_max_lut_box_ValueChange(self, value):
        """Method to handle lut max spinbox value change."""
        slider_vals = self.lut_slider.value()
        self.lut_slider.setValue((slider_vals[0], value))

    def _set_default_visible(self):
        """Method that sets the default visible widgets in the main window."""
        self.clip_meas.setChecked(False)
        self.position.setVisible(False)
        self.position_label.setVisible(False)
        self.additional_filter_combobox.setVisible(False)
        self.additional_filter_combobox_label.setVisible(False)
        self.polyDeg.setVisible(False)
        self.polyDeg_label.setVisible(False)

    def toggle_bias_method_parameter_visibility(self):
        """
        based on the seleciton of bias method:
        shows or hides the appropriate options in the main window.
        """
        if self.bias_method.currentText() == "runmed":
            self.smooth_k.setVisible(True)
            self.smooth_k_label.setVisible(True)
            self.polyDeg.setVisible(False)
            self.polyDeg_label.setVisible(False)
            self.bias_k.setVisible(True)
            self.bias_k_label.setVisible(True)
            self.bin_peak_threshold.setVisible(True)
            self.bin_peak_threshold_label.setVisible(True)
            self.bin_threshold.setVisible(True)
            self.bin_threshold_label.setVisible(True)

        if self.bias_method.currentText() == "lm":
            self.smooth_k.setVisible(True)
            self.smooth_k_label.setVisible(True)
            self.polyDeg.setVisible(True)
            self.polyDeg_label.setVisible(True)
            self.bias_k.setVisible(False)
            self.bias_k_label.setVisible(False)
            self.bin_peak_threshold.setVisible(True)
            self.bin_peak_threshold_label.setVisible(True)
            self.bin_threshold.setVisible(True)
            self.bin_threshold_label.setVisible(True)

        if self.bias_method.currentText() == "none":
            self.smooth_k.setVisible(True)
            self.smooth_k_label.setVisible(True)
            self.polyDeg.setVisible(False)
            self.polyDeg_label.setVisible(False)
            self.bias_k.setVisible(False)
            self.bias_k_label.setVisible(False)
            self.bin_peak_threshold.setVisible(False)
            self.bin_peak_threshold_label.setVisible(False)
            self.bin_threshold.setVisible(True)
            self.bin_threshold_label.setVisible(True)

    def callback_file_Linedit_text(self, value):
        """Used to load sample data from test_data.
        and set columnpicker to indicated strings."""
        self.file_LineEdit.setText(value)
        self.open_file_button.click()
        columnpicker.frame.value = "t"
        columnpicker.track_id.value = "id"
        columnpicker.x_coordinates.value = "x"
        columnpicker.y_coordinates.value = "y"
        columnpicker.z_coordinates.value = "None"
        columnpicker.measurment.value = "m"
        columnpicker.field_of_view_id.value = "Position"
        columnpicker.additional_filter.value = "None"
        columnpicker.measurement_math.value = "None"

    def update_what_to_run_all(self):
        """
        sets 'what to run' attribute to 'all' in the what_to_run attirbute,
        that is used in the main function to check if what to run
        when certain field have updated values.
        """
        self.what_to_run.add("all")

    def update_what_to_run_tracking(self):
        """sets 'what to run' attribute to 'from_tracking' in the what_to_run attirbute,
        that is used in the main function to check if what to run
        when certain field have updated values."""
        self.what_to_run.add("from_tracking")

    def update_what_to_run_filtering(self):
        """sets 'what to run' attribute to 'from_filtering'
        in the what_to_run attirbute,
        that is used in the main function to check if what to run
        when certain field have updated values."""
        self.what_to_run.add("from_filtering")

    def browse_files(self):
        """Opens a filedialog and saves path as a string in self.filename"""
        self.filename = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load CSV file", str(Path.home()), "csv(*.csv);; csv.gz(*.csv.gz);;"
        )
        self.file_LineEdit.setText(self.filename[0])

    def subtract_timeoffset(self):
        """Method to subtract the timeoffset in the frame column of data"""
        data = self.data
        data[self.frame] -= min(data[self.frame])
        self.data = data

    def set_point_size(self):
        """
        updates values in lut mapping sliders
        """
        data = self.data
        if not data.empty:
            minx = min(data[self.x_coordinates])
            maxx = max(data[self.x_coordinates])
            miny = min(data[self.y_coordinates])
            maxy = max(data[self.y_coordinates])

            max_coord_diff = max(maxx - minx, maxy - miny)
            self.point_size.setValue(
                0.75482
                + 0.00523857 * max_coord_diff
                + 9.0618311e-6 * max_coord_diff**2
            )

    def reset_contrast(self):
        """updates values in lut mapping slider."""
        min_max = stored_variables.min_max
        # change slider values
        self.max_lut_spinbox.setMaximum(min_max[1])
        self.max_lut_spinbox.setMinimum(min_max[0])
        self.min_lut_spinbox.setMaximum(min_max[1])
        self.min_lut_spinbox.setMinimum(min_max[0])
        self.lut_slider.setRange(*min_max)
        self.max_lut_spinbox.setValue(min_max[1])
        self.min_lut_spinbox.setValue(min_max[0])

    def update_lut(self):
        """updates LUT choice in stored_variables."""
        stored_variables.lut = self.LUT.currentText()

    def change_cell_colors(self):
        """Method to update lut and corresponding lut mappings."""
        layer_list = self.get_layer_list()
        min_value = self.min_lut_spinbox.value()
        max_value = self.max_lut_spinbox.value()
        if "all_cells" in layer_list:
            self.viewer.layers["all_cells"].face_colormap = self.LUT.currentText()
            self.viewer.layers["all_cells"].face_contrast_limits = (
                min_value,
                max_value,
            )
            self.viewer.layers["all_cells"].refresh_colors()

    def change_cell_size(self):
        """Method to update size of points and shapes layers:
        "all_cells, "active cells", "coll cells", "coll events"
        and if created "event_boundingbox".
        """
        layer_list = self.get_layer_list()
        size = self.point_size.value()
        if "all_cells" in layer_list:
            self.viewer.layers["all_cells"].size = size
            self.viewer.layers["active cells"].size = round(size / 2.5, 2)

        if "coll cells" in layer_list:
            self.viewer.layers["coll cells"].size = round(size / 1.7, 2)

        if "event_boundingbox" in self.viewer.layers:
            self.viewer.layers["event_boundingbox"].edge_width = size / 5

    def open_columnpicker(self):
        """Open Columnpicker dialog window.
        Take a filename and if it is a csv file,
        opens it and stores it in the stored_variables_object.
        Shows columnpicker dialog.
        """
        extension = [".csv", ".csv.gz"]
        csv_file = self.file_LineEdit.text()
        if not csv_file.endswith(tuple(extension)):
            show_info("Not a csv file")
        else:
            csv_file = self.file_LineEdit.text()
            self.layers_to_create.clear()
            columns, delimiter_value = read_data_header(csv_file)
            columns_w_none = columns + ["None"]
            columnpicker.frame.choices = columns
            columnpicker.track_id.choices = columns
            columnpicker.x_coordinates.choices = columns
            columnpicker.y_coordinates.choices = columns
            columnpicker.z_coordinates.choices = columns_w_none
            columnpicker.measurment.choices = columns
            columnpicker.second_measurment.choices = columns
            columnpicker.field_of_view_id.choices = columns_w_none
            columnpicker.additional_filter.choices = columns_w_none
            # columnpicker.field_of_view_id.set_choice("None", "None")
            # columnpicker.additional_filter.set_choice("None", "None")
            # columnpicker.z_coordinates.set_choice("None", "None")
            columnpicker.show()
            self.data = pd.read_csv(csv_file, delimiter=delimiter_value)

    def close_columnpicker(self):
        """Stores the chosen columns inside of
        class attributes and closes the columnpicker dialog.
        Additionally subtracts the frame-offset from the frame
        column in data.
        """
        # populate column dictionnary
        self.frame = columnpicker.frame.value
        self.track_id = columnpicker.track_id.value
        self.x_coordinates = columnpicker.x_coordinates.value
        self.y_coordinates = columnpicker.y_coordinates.value
        self.z_coordinates = columnpicker.z_coordinates.value
        self.first_measurement = columnpicker.measurment.value
        self.second_measurement = columnpicker.second_measurment.value
        self.field_of_view_id = columnpicker.field_of_view_id.value
        self.additional_filter_column_name = columnpicker.additional_filter.value
        columnpicker.close()
        self.measurement, self.data = self.calculate_measurment(
            self.data, self.first_measurement, self.second_measurement
        )
        self.subtract_timeoffset()

    def calculate_measurment(self, data, in_meas_1_name, in_meas_2_name):
        """Perform operation on the two measurement columns.

        Calcualates new column that will be used to detect collective events.
        Operation is determined in the columnpicker dialog and loaded from
        the OPERATOR_DICTIONARY.
        """
        data_in = data
        operation = columnpicker.measurement_math.value
        if operation in OPERATOR_DICTIONARY.keys():
            out_meas_name = OPERATOR_DICTIONARY[operation][1]
            data_in[out_meas_name] = OPERATOR_DICTIONARY[operation][0](
                data[in_meas_1_name], data[in_meas_2_name]
            )

        else:
            out_meas_name = in_meas_1_name
        return out_meas_name, data_in

    def remove_layers_after_columnpicker(self):
        """Method to remove existing arcos layers before loading new data"""
        layer_list = self.get_layer_list()
        for layer in [
            "coll cells",
            "coll events",
            "active cells",
            "all_cells",
            "Timestamp",
            "event_boundingbox",
        ]:
            if layer in layer_list:
                self.viewer.layers.remove(layer)

    def get_layer_list(self):
        """Get list of open layers."""
        layer_list = [layer.name for layer in self.viewer.layers]
        return layer_list

    def set_positions(self):
        """Get unique positions from data, empty positions dialog
        for preveious data and append new positions."""
        if self.field_of_view_id != "None":
            positions = list(self.data[self.field_of_view_id].unique())
        else:
            positions = ["None"]

        if self.additional_filter_column_name != "None":
            additional_filter_choices = (
                self.data[self.additional_filter_column_name].unique().tolist()
            )
        else:
            additional_filter_choices = ["None"]

        # delete position values is position dialog self.positions
        self.position.clear()
        self.additional_filter_combobox.clear()
        for i in additional_filter_choices:
            self.additional_filter_combobox.addItem(str(i), i)
        # add new positions
        for i in positions:
            self.position.addItem(str(i), i)

        # hides position choice if no position column exists in the raw data
        # i.e. during columnpicker position was set to None.
        # Also hides it if there is only one position available.
        # is updated everytime when new data is read in
        if self.position.count() <= 1:
            self.position.setVisible(False)
            self.position_label.setVisible(False)
        else:
            self.position.setVisible(True)
            self.position_label.setVisible(True)

        if self.additional_filter_combobox.count() <= 1:
            self.additional_filter_combobox.setVisible(False)
            self.additional_filter_combobox_label.setVisible(False)
        else:
            self.additional_filter_combobox.setVisible(True)
            self.additional_filter_combobox_label.setVisible(True)

    def set_posCol(self) -> list:
        """Generates the posCol list containing all position names."""
        if self.z_coordinates != "None":
            posCols = [self.x_coordinates, self.y_coordinates, self.z_coordinates]
            return posCols

        posCols = [self.x_coordinates, self.y_coordinates]
        return posCols

    def get_tracklengths(self):
        """
        Groups filtered data by track_id and
        returns minimum and maximum tracklenght.
        Updates min and max tracklenght in
        the widget spinbox and sliders.
        """
        data = self.data
        if not data.empty:
            if self.field_of_view_id != "None":
                track_lenths = self.data.groupby(
                    [
                        self.field_of_view_id,
                        self.track_id,
                    ]
                ).size()
            else:
                track_lenths = self.data.groupby([self.track_id]).size()
            minmax = (min(track_lenths), max(track_lenths))

            if minmax[1] - minmax[0] > 1:
                self.tracklenght_slider.setMinimum(minmax[0])
                self.tracklenght_slider.setMaximum(minmax[1])

            self.min_tracklength_spinbox.setMinimum(minmax[0])
            self.max_tracklength_spinbox.setMinimum(minmax[0])

            self.min_tracklength_spinbox.setMaximum(minmax[1])
            self.max_tracklength_spinbox.setMaximum(minmax[1])

            self.min_tracklength_spinbox.setValue(minmax[0])
            self.max_tracklength_spinbox.setValue(minmax[1])

    def filter_data(self):
        """
        Used to filter the input data to contain a single position.
        If selected in the columpicker dialog, an additional filter option
        is displayed.
        Filter options also include minimum and maximum tracklength.
        Allows for rescaling of measurment variable.

        """
        # gets raw data read in by arcos_widget from stored_variables object
        # and columns from columnpicker value
        posCols = self.set_posCol()
        in_data = process_input(
            df=self.data,
            field_of_view_column=self.field_of_view_id,
            frame_column=self.frame,
            pos_columns=posCols,
            track_id_column=self.track_id,
            measurement_column=self.measurement,
        )
        if self.data.empty or self.field_of_view_id == self.measurement:
            show_info("No data loaded, or not loaded correctly")
        else:
            # if the position column was not chosen in columnpicker,
            # dont filter by position
            if self.field_of_view_id != "None":
                # hast to be done before .filter_tracklenght otherwise code could break
                # if track ids are not unique to positions
                in_data.filter_position(self.position.currentData())

            if self.additional_filter_column_name != "None":
                in_data.filter_second_column(
                    self.additional_filter_column_name,
                    self.additional_filter_combobox.currentData(),
                )
            # filter by tracklenght
            in_data.filter_tracklength(
                self.min_tracklength_spinbox.value(),
                self.max_tracklength_spinbox.value(),
            )
            # option to rescale the measurment column
            in_data.rescale_measurment(rescale_factor=self.rescale_measurment.value())
            # option to set frame interval
            in_data.frame_interval(self.frame_interval.value())

            filtered_data = in_data.return_pd_df()

            # get min and max values
            if not filtered_data.empty:
                max_meas = max(filtered_data[self.measurement])
                min_meas = min(filtered_data[self.measurement])
                stored_variables.min_max = (min_meas, max_meas)
            self.filtered_data = filtered_data
            self._ts_plot_update()
            show_info("Data Filtered!")

    def check_for_collid_column(
        self, data: pd.DataFrame, collid_column="collid", suffix="old"
    ):
        """If collid_column is present in input data,
        add suffix to prevent dataframe merge conflic"""
        if collid_column in data.columns:
            data.rename(
                columns={collid_column: f"{collid_column}_{suffix}"}, inplace=True
            )
        return data

    # @profile
    def run_arcos(self):
        """
        Method to detect collective events.

        When called the loaded data is processed according to the selected parameters,
        in order to detect collective events.

        Updates layers_to_create (list) with:
            all_cells (tuple): returns color coded points
            of all cells filtered by the filter_widget.
            Color code represents measurment active_cells: returns black dots,
            representing cells determined as being active
            by arcos binarise measurment function

            active cells (tuple):points representing cells that have been classified as
            being active by the arcos binarization approach.

            coll cells (tuple): returns points representing cells that are according
            to the calculation done by arcos part of a collective event.
            Colored by collective event id with tab20 colormap.

            coll events (tuple): returns convex hulls of individual collective
            events, in 2D case, color is coded accoring to a color_cycle attribute.
            in 3D case according to a LUT.
        """

        posCols = self.set_posCol()
        measbin_col = f"{self.measurement}.bin"
        collid_name = "collid"
        self.filtered_data = self.check_for_collid_column(
            self.filtered_data, collid_name
        )

        # checks if this part of the function has to be run,
        # depends on the parameters changed in arcos widget

        if self.filtered_data.empty:
            show_info("No Data Loaded, Use arcos_widget to load and filter data first")
        else:
            if "all" in self.what_to_run:

                # get point_size
                size = self.point_size.value()

                # sets Progressbar to 0
                self.Progress.reset()
                # create arcos object, run arcos
                arcos = ARCOS(
                    data=self.filtered_data,
                    posCols=posCols,
                    frame_column=self.frame,
                    id_column=self.track_id,
                    measurement_column=self.measurement,
                    clid_column=collid_name,
                )

                self.Progress.setValue(4)

                # if corresponding checkbox was selected run interpolate measurments
                if self.interpolate_meas.isChecked():
                    arcos.interpolate_measurements()

                # if corresponding checkbock was selected run clip_measuremnts
                if self.clip_meas.isChecked():
                    arcos.clip_meas(
                        clip_low=self.clip_low.value(),
                        clip_high=self.clip_high.value(),
                    )

                self.Progress.setValue(8)

                # binarize data and update self.ts variable
                # update from where to run
                self.ts = arcos.bin_measurements(
                    smoothK=self.smooth_k.value(),
                    biasK=self.bias_k.value(),
                    peakThr=self.bin_peak_threshold.value(),
                    binThr=self.bin_threshold.value(),
                    polyDeg=self.polyDeg.value(),
                    biasMet=self.bias_method.currentText(),
                )
                self.start_arcos = deepcopy(arcos)
                self.what_to_run.add("from_tracking")

                self.Progress.setValue(12)

            # if statement checks if this part of the function has to be run
            # depends on the parameters changed in arcos widget
            if "from_tracking" in self.what_to_run:
                arcos = deepcopy(self.start_arcos)  # type: ignore
                # if active cells were detected, run this
                if 1 in self.ts[measbin_col].values:
                    # track collective events
                    arcos.trackCollev(
                        self.neighbourhood_size.value(),
                        self.min_clustersize.value(),
                        self.nprev_spinbox.value(),
                    )

                    self.Progress.setValue(16)
                # if no active cells were detected remove previous layer,
                # since this does not correspont to current widget parameters
                else:
                    layer_list = self.get_layer_list()
                    # check if layers exist, if yes, remove them
                    if "active cells" in layer_list:
                        self.viewer.layers.remove("active cells")
                        self.viewer.layers.remove("all_cells")
                    self.Progress.setValue(40)
                    show_info("No active Cells detected, consider adjusting parameters")

                # update attributes with current state of arcos and what to run
                # copy is required in order to facilitate recalculation from specific
                # points without the  need to recalculate everything
                self.tracking_arcos = deepcopy(arcos)
                self.what_to_run.add("from_filtering")

            # depending on the parameters changed in arcos widget
            if "from_filtering" in self.what_to_run:

                # get most recent data from attribute
                arcos = deepcopy(self.tracking_arcos)  # type: ignore

                # if no data show info to run arcos first
                # and set the progressbar to 0
                if arcos is None:
                    show_info("No data available, run arcos first")
                    self.Progress.setValue(0)

                # if cells were classifed as being active (represented by a 1)
                # filter tracked events acording to chosen parameters
                elif 1 in self.ts[measbin_col].values:

                    # set return varaibles to check which layers have to be created
                    return_collev = False
                    return_points = True
                    filterer = filterCollev(
                        arcos.data, self.frame, collid_name, self.track_id
                    )
                    self.arcos_filtered = filterer.filter(
                        self.min_dur.value(),
                        self.total_event_size.value(),
                    ).copy(deep=True)
                    # makes filtered collids sequential
                    clid_np = self.arcos_filtered[collid_name].to_numpy()
                    clids_sorted_i = np.argsort(clid_np)
                    clids_reverse_i = np.argsort(clids_sorted_i)
                    clid_np_sorted = clid_np[(clids_sorted_i)]
                    grouped_array_clids = np.split(
                        clid_np_sorted,
                        np.unique(clid_np_sorted, axis=0, return_index=True)[1][1:],
                    )
                    seq_colids = np.concatenate(
                        [
                            np.repeat(i, value.shape[0])
                            for i, value in enumerate(grouped_array_clids)
                        ],
                        axis=0,
                    )[clids_reverse_i]
                    seq_colids_from_one = np.add(seq_colids, 1)
                    self.arcos_filtered.loc[:, collid_name] = seq_colids_from_one

                    self.Progress.setValue(20)

                    self.Progress.setValue(20)
                    # merge tracked and original data
                    merged_data = pd.merge(
                        self.ts,
                        self.arcos_filtered[
                            [
                                self.frame,
                                self.track_id,
                                "collid",
                            ]
                        ],
                        how="left",
                        on=[
                            self.frame,
                            self.track_id,
                        ],
                    )
                    stored_variables.data_merged = merged_data
                    if self.z_coordinates == "None":
                        # column list
                        vColsCore = [
                            self.frame,
                            self.y_coordinates,
                            self.x_coordinates,
                        ]
                    else:
                        # column list
                        vColsCore = [
                            self.frame,
                            self.y_coordinates,
                            self.x_coordinates,
                            self.z_coordinates,
                        ]

                    # np matrix with all cells
                    datAll = merged_data[vColsCore].to_numpy()

                    # a dictionary with activities;
                    # shown as a color code of all cells

                    datAllProp = {"act": merged_data[self.measurement]}
                    # np matrix with acvtive cells; shown as black dots
                    datAct = merged_data[merged_data[measbin_col] > 0][
                        vColsCore
                    ].to_numpy()

                    # get point_size
                    size = self.point_size.value()

                    active_cells = (
                        datAct,
                        {
                            "size": round(size / 2.5, 2),
                            "edge_width": 0,
                            "face_color": "black",
                            "opacity": 1,
                            "symbol": "disc",
                            "name": "active cells",
                        },
                        "points",
                    )

                    # np matrix with cells in collective events
                    datColl = merged_data[~np.isnan(merged_data["collid"])][
                        vColsCore
                    ].to_numpy()

                    # tuple to return layer as layer.data.tuple
                    all_cells = (
                        datAll,
                        {
                            "properties": datAllProp,
                            "edge_width": 0,
                            "edge_color": "act",
                            "face_color": "act",
                            "face_colormap": stored_variables.lut,
                            "face_contrast_limits": stored_variables.min_max,
                            "size": size,
                            "edge_width": 0,
                            "opacity": 1,
                            "symbol": "disc",
                            "name": "all_cells",
                        },
                        "points",
                    )
                    self.Progress.setValue(24)
                    # check if collective events were detected and add layer,
                    # if yes calculate convex hulls for collective events
                    if datColl.size != 0:
                        # convex hulls only if user selects checkbox
                        if self.add_convex_hull_checkbox.isChecked():
                            if self.z_coordinates == "None":
                                datChull, color_ids = get_verticesHull(
                                    merged_data[~np.isnan(merged_data["collid"])],
                                    frame=self.frame,
                                    colid=collid_name,
                                    col_x=self.x_coordinates,
                                    col_y=self.y_coordinates,
                                )

                                self.Progress.setValue(28)

                                self.Progress.setValue(32)

                                coll_events = (
                                    datChull,
                                    {
                                        "face_color": color_ids,
                                        "shape_type": "polygon",
                                        "text": None,
                                        "opacity": 0.5,
                                        "edge_color": "white",
                                        "edge_width": 0,
                                        "name": "coll events",
                                    },
                                    "shapes",
                                )

                            else:
                                event_surfaces = make_surface_3d(
                                    merged_data[~np.isnan(merged_data["collid"])],
                                    self.frame,
                                    self.x_coordinates,
                                    self.y_coordinates,
                                    self.z_coordinates,
                                    "collid",
                                )

                                event_surfaces = fix_3d_convex_hull(
                                    merged_data[vColsCore],
                                    event_surfaces[0],
                                    event_surfaces[1],
                                    event_surfaces[2],
                                    self.frame,
                                )
                                coll_events = (
                                    event_surfaces,
                                    {
                                        "colormap": tab20,
                                        "name": "coll events",
                                        "opacity": 0.5,
                                    },
                                    "surface",
                                )

                        # get point_size
                        size = self.point_size.value()
                        # create remaining layer.data.tuples
                        np_clids = merged_data[~np.isnan(merged_data["collid"])][
                            "collid"
                        ].to_numpy()

                        color_ids = np.take(
                            np.array(COLOR_CYCLE), [i for i in np_clids], mode="wrap"
                        )
                        coll_cells = (
                            datColl,
                            {
                                "face_color": color_ids,
                                "size": round(size / 1.2, 2),
                                "edge_width": 0,
                                "opacity": 1,
                                "name": "coll cells",
                            },
                            "points",
                        )

                        return_collev = True

                        self.Progress.setValue(36)
                    else:
                        layer_list = self.get_layer_list()
                        # check if layers exit, if yes remove them
                        if "coll cells" in layer_list:
                            self.viewer.layers.remove("coll cells")
                        if "coll events" in layer_list:
                            self.viewer.layers.remove("coll events")
                        show_info(
                            "No collective events detected, consider adjusting parameters"  # NOQA
                        )

                    self.Progress.setValue(40)
                    self.layers_to_create.clear()

                    # update layers
                    # check which layers need to be added, add these layers to the list
                    if (
                        return_collev
                        and return_points
                        and self.add_convex_hull_checkbox.isChecked()
                    ):
                        self.layers_to_create = [
                            all_cells,
                            active_cells,
                            coll_cells,
                            coll_events,
                        ]
                    if (
                        return_collev
                        and return_points
                        and not self.add_convex_hull_checkbox.isChecked()
                    ):
                        self.layers_to_create = [
                            all_cells,
                            active_cells,
                            coll_cells,
                        ]
                    elif not return_collev and return_points:
                        self.layers_to_create = [all_cells, active_cells]
                    self.what_to_run.clear()

    # @profile
    def make_layers(self):
        """adds layers from self.layers_to_create,
        whitch itself is upated from run_arcos method"""
        layers_names = [layer.name for layer in self.viewer.layers]
        if self.layers_to_create:
            for layer in [
                "coll cells",
                "coll events",
                "active cells",
                "all_cells",
                "event_boundingbox",
            ]:
                if layer in layers_names:
                    self.viewer.layers.remove(layer)
            for result in self.layers_to_create:
                self.viewer.add_layer(napari.layers.Layer.create(*result))

    def run(self):
        """main function to run arcos, add layers and change cell size"""
        self.run_arcos()
        self.make_layers()
        self.collev_plot_update()
        self.change_cell_size()
        if self.z_coordinates != "None":
            self.viewer.dims.ndisplay = 3
        else:
            self.viewer.dims.ndisplay = 2


def export_csv(merged_data):
    """
    Function to export the arcos data stored in the
    stored_varaibles object and save it to the
    filepath set with the arcos_widget widget
    """
    if merged_data.empty:
        show_info("No data to export, run arcos first")
    else:
        output_csv_folder.close()
        path = str(output_csv_folder.filename.value)
        output_path = f"{path}{sep}{output_csv_folder.Name.value}.csv"
        merged_data.to_csv(output_path)
        show_info(f"wrote csv file to {output_path}")


def movie_export(viewer, automatic_viewer_size):
    """
    Function to export image sequence from the viewer.
    """
    # closes magicgui filepicker
    output_movie_folder.close()
    # if no layers are present, dont export data
    if len(viewer.layers) == 0:
        show_info("No data to export, run arcos first")
    else:
        path = str(output_movie_folder.filename.value)
        output_path = f"{path}{sep}{output_movie_folder.Name.value}"
        # resize viewer if chosen
        if automatic_viewer_size:
            # hides dock widgets, sets path, gets viwer dimensions
            hide_dock_widgets(viewer)
            resize_napari([np.float64(1012), np.float64(1012)], viewer)
        # iterate over frames to export data to chosen output path
        iterate_over_frames.show()
        iterate_over_frames(viewer, output_path)
        iterate_over_frames.close()
        show_dock_widgets(viewer)


@magicgui(
    call_button="Ok",
    filename={"label": "Choose Folder:", "mode": "d"},
    Automaic_viewer_size={
        "widget_type": "CheckBox",
        "label": "Automatically determine correct \n viewer size for export",
    },
)
def output_movie_folder(
    filename=Path(),
    Name="arcos",
    Automaic_viewer_size=True,
):
    viewer = napari.current_viewer()
    """
    FileDialog with magicgui to choose movie path
    """
    output_movie_folder.close()
    movie_export(viewer, Automaic_viewer_size)


# show folder selector for movie export
def show_output_movie_folder():
    output_movie_folder.show()


@magicgui(call_button="Ok", filename={"label": "Choose Folder:", "mode": "d"})
def output_csv_folder(filename=Path(), Name="arcos_data"):
    """
    FileDialog with magicgui for writing csv file
    """
    merged_data = stored_variables.data_merged
    output_csv_folder.close()
    export_csv(merged_data)


# show folder selector for csv epxort
def show_output_csv_folder():
    output_csv_folder.show()


# function to hide and show dockwidgets -> maybe risky method?
def hide_dock_widgets(viewer):
    for key, value in viewer.window._dock_widgets.items():
        value.setVisible(False)


def show_dock_widgets(viewer):
    for key, value in viewer.window._dock_widgets.items():
        value.show()


# init function for magic_factory add_timestamp
def on_ts_init(new_widget):
    new_widget.Set_Timestamp_Options.changed.connect(show_timestamp_options)


@magic_factory(
    widget_init=on_ts_init,
    call_button="Add Timestamp",
    Set_Timestamp_Options={"widget_type": "PushButton"},
)
def add_timestamp(
    viewer: "napari.viewer.Viewer", Set_Timestamp_Options=False
) -> LayerDataTuple:
    """Button to add timestamp"""
    if list(viewer.layers) and viewer.layers.ndim > 2:
        layer_list = [layer.name for layer in viewer.layers]
        if "Timestamp" in layer_list:
            viewer.layers.remove("Timestamp")
        kw_timestamp = make_timestamp(
            viewer,
            start_time=timestamp_options.start_time.value,
            step_time=timestamp_options.step_time.value,
            prefix=timestamp_options.prefix.value,
            suffix=timestamp_options.suffix.value,
            position=timestamp_options.position.value,
            size=timestamp_options.size.value,
            x_shift=timestamp_options.x_shift.value,
            y_shift=timestamp_options.y_shift.value,
        )
        time_stamp = (
            kw_timestamp["data"],
            {
                "properties": kw_timestamp["properties"],
                "face_color": kw_timestamp["face_color"],
                "edge_color": kw_timestamp["edge_color"],
                "shape_type": kw_timestamp["shape_type"],
                "text": kw_timestamp["text"],
                "opacity": kw_timestamp["opacity"],
                "name": "Timestamp",
            },
            "shapes",
        )
        return [time_stamp]
    else:
        show_info("No Layers to add Timestamp to")


def on_export_data_init(new_widget):
    """init function for the export data widget"""
    new_widget.Export_ARCOS_as_csv.changed.connect(show_output_csv_folder)
    new_widget.Export_movie.changed.connect(show_output_movie_folder)


@magic_factory(
    widget_init=on_export_data_init,
    call_button=False,
    Export_ARCOS_as_csv={"widget_type": "PushButton"},
    Export_movie={"widget_type": "PushButton"},
)
def export_data(Export_ARCOS_as_csv=False, Export_movie=False):
    """Widget to export csv and movie data"""
