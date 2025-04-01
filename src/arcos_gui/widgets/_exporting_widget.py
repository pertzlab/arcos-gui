"""Exporting widget for arcos-gui."""

from __future__ import annotations

import os
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from arcos_gui.processing import BatchProcessor
from arcos_gui.tools import (
    ALLOWED_SETTINGS,
    AVAILABLE_OPTIONS_FOR_BATCH,
    BatchFileDialog,
    MovieExporter,
    ParameterFileDialog,
)
from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy import QtWidgets, uic
from qtpy.QtCore import QTimer, Signal
from qtpy.QtGui import QIcon

if TYPE_CHECKING:
    import napari.viewer
    from arcos_gui.processing import DataStorage

# icons
ICONS = Path(__file__).parent.parent / "_icons"


class _exportwidget(QtWidgets.QWidget):
    UI_FILE = str(Path(__file__).parent.parent / "_ui" / "export_widget.ui")

    # The UI_FILE above contains these objects:

    browse_file_data: QtWidgets.QPushButton
    file_LineEdit_data: QtWidgets.QLineEdit
    base_name_LineEdit_data: QtWidgets.QLineEdit
    data_export_button: QtWidgets.QPushButton
    param_export_button: QtWidgets.QPushButton
    param_import_button: QtWidgets.QPushButton
    stats_export_button: QtWidgets.QPushButton
    batch_processing_button: QtWidgets.QPushButton
    abort_batch_button: QtWidgets.QPushButton

    browse_file_img: QtWidgets.QPushButton
    file_LineEdit_img: QtWidgets.QLineEdit
    base_name_LineEdit_img: QtWidgets.QLineEdit
    img_seq_export_button: QtWidgets.QPushButton
    format_combobox: QtWidgets.QComboBox
    upsample_spinbox: QtWidgets.QSpinBox
    fps_spinbox: QtWidgets.QSpinBox

    closing = Signal()

    def __init__(self, data_storage: DataStorage, parent=None):
        super().__init__(parent)
        self._data_storage_instance = data_storage
        self._batch_process_path: str | None = None
        self._parameters_path: str | None = None
        self.setup_ui()

    def _browse_file_data(self):
        base_path = self._data_storage_instance.file_name.value
        base_path = str(Path(base_path).parent)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", base_path
        )
        self.file_LineEdit_data.setText(path)

    def _browse_file_img(self):
        base_path = self._data_storage_instance.file_name.value
        base_path = str(Path(base_path).parent)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", base_path
        )
        self.file_LineEdit_img.setText(path)

    def _browse_parmeters_export(self):
        if self._parameters_path is None:
            base_path = self._data_storage_instance.file_name.value
        else:
            base_path = self._parameters_path
        base_path = str(Path(base_path).parent)
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save as Yaml", base_path, filter="*.yaml"
        )[0]
        self._parameters_path = path
        if path:
            if not path.endswith(".yaml"):
                path = os.path.join(path, ".yaml")
            return path
        else:
            return None

    def _browse_parmeters_import(self):
        if self._parameters_path is None:
            base_path = self._data_storage_instance.file_name.value
        else:
            base_path = self._parameters_path
        base_path = str(Path(base_path).parent)

        dialog = ParameterFileDialog(
            selection_values=ALLOWED_SETTINGS,
            directory=base_path,
            parent=self,
            caption="Select Parameters to Import",
        )

        if dialog.exec_():
            # Get the selected directory
            path = dialog.selectedFiles()[0]
            self._parameters_path = path

            # Get the values corresponding to the checkboxes that are checked
            options_selected = dialog.get_selected_options()
            dialog.close()
            dialog.deleteLater()
            return path, options_selected
        else:
            dialog.close()
            dialog.deleteLater()
            return None, None

    def _browse_batch_output(self):
        if self._batch_process_path is None:
            base_path = self._data_storage_instance.file_name.value
        else:
            base_path = self._batch_process_path
        base_path = str(Path(base_path).parent)

        dialog = BatchFileDialog(
            selection_values=AVAILABLE_OPTIONS_FOR_BATCH,
            directory=base_path,
            parent=self,
        )
        # toggle of 'timelapse_frames' checkbox
        for checkbox in dialog.checkboxes.items():
            if checkbox[0] == "timelapse_frames":
                checkbox[1].setChecked(False)
                # checkbox[1].setEnabled(False)
        dialog.setWindowTitle("Select Directory")

        if dialog.exec_():
            # Get the values corresponding to the checkboxes that are checked
            options_selected = dialog.get_selected_options()

            # Get the selected directory
            path = dialog.selectedFiles()[0]
            self._batch_process_path = path
            dialog.close()
            dialog.deleteLater()
            return path, options_selected
        else:
            dialog.close()
            dialog.deleteLater()
            return None, None

    def _update_base_name_data(self):
        base_name = self._data_storage_instance.file_name.value
        self.base_name_LineEdit_data.setText(Path(base_name).stem)
        self.base_name_LineEdit_img.setText(Path(base_name).stem)

    def _connect_signals(self):
        self.browse_file_data.clicked.connect(self._browse_file_data)
        self.browse_file_img.clicked.connect(self._browse_file_img)
        self._data_storage_instance.file_name.value_changed.connect(
            self._update_base_name_data
        )

    def setup_ui(self):
        """Load the .ui file and set icons."""
        uic.loadUi(self.UI_FILE, self)  # load QtDesigner .ui file
        self._connect_signals()
        self.browse_file_icon = QIcon(str(ICONS / "folder-open-line.svg"))
        self.browse_file_data.setIcon(self.browse_file_icon)
        self.browse_file_img.setIcon(self.browse_file_icon)
        self.abort_batch_button.setStyleSheet(
            "background-color : #7C0A02; color : white"
        )
        self.format_combobox.addItems(["png", "mp4", "gif", "jpeg", "tif"])
        self._hide_abort_batch_button()

    def _hide_abort_batch_button(self):
        self.abort_batch_button.setVisible(False)
        self.abort_batch_button.setEnabled(False)
        self.batch_processing_button.setVisible(True)
        self.batch_processing_button.setEnabled(True)

    def _show_abort_batch_button(self):
        self.abort_batch_button.setVisible(True)
        self.abort_batch_button.setEnabled(True)
        self.batch_processing_button.setVisible(False)
        self.batch_processing_button.setEnabled(False)

    def closeEvent(self, event):
        self.closing.emit()
        event.accept()


class ExportController:
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        data_storage_instance: DataStorage,
        parent=None,
    ):
        self.viewer = viewer
        self._data_storage_instance = data_storage_instance
        self.widget = _exportwidget(self._data_storage_instance, parent)
        self.abort_timer = QTimer(parent)
        self.abort_timer.timeout.connect(self.abort_timer_timeout)

        self.current_date = self._get_current_date()
        self._connect_callbacks()

    def _get_current_date(self):
        now = datetime.now()
        return now.strftime("%Y%m%d")

    def _export_arcos_data(self):
        if self._data_storage_instance.arcos_output.value.empty:
            show_info("No data to export, run arcos first")
        else:
            path = Path(self.widget.file_LineEdit_data.text())
            output_name = f"{self.current_date}_{self.widget.base_name_LineEdit_data.text()}_arcos_output.csv"
            outpath = os.path.join(path, output_name)
            self._data_storage_instance.arcos_output.value.to_csv(outpath, index=False)
            show_info(f"wrote csv file to {outpath}")

    def _export_arcos_stats(self):
        if self._data_storage_instance.columns.value.object_id is None:
            show_info("No Stats are calculated without a track id column selected")

        elif self._data_storage_instance.arcos_stats.value.empty:
            show_info("No data to export, run arcos first")

        else:
            path = Path(self.widget.file_LineEdit_data.text())
            output_name = f"{self.current_date}_{self.widget.base_name_LineEdit_data.text()}_arcos_stats.csv"
            outpath = os.path.join(path, output_name)
            self._data_storage_instance.arcos_stats.value.to_csv(outpath, index=False)
            show_info(f"wrote csv file to {outpath}")

    def _export_arcos_params(self):
        path = self.widget._browse_parmeters_export()
        if path is None:
            return
        self._data_storage_instance.export_to_yaml(path)
        show_info(f"wrote yaml file to {path}")

    def _import_arcos_params(self):
        path, what_to_import = self.widget._browse_parmeters_import()
        if path is None:
            return
        if not what_to_import:
            show_info("No settings selected to import")
            return
        self._data_storage_instance.import_from_yaml(path, what_to_import)
        show_info(f"imported yaml file from {path}")

    def _export_image_series(self):
        if self.viewer.layers == []:
            show_info("No layers to export")
        else:
            path = Path(self.widget.file_LineEdit_img.text())
            output_name = f"{self.current_date}_{self.widget.base_name_LineEdit_img.text()}_arcos_output"
            MovieExporter(
                self.viewer,
                path,
            ).run(
                output_format=self.widget.format_combobox.currentText(),
                output_name=output_name,
                fps=self.widget.fps_spinbox.value(),
                scale_factor=self.widget.upsample_spinbox.value(),
            )

    def _connect_callbacks(self):
        self.widget.closing.connect(self.closeEvent)
        self.widget.data_export_button.clicked.connect(self._export_arcos_data)
        self.widget.stats_export_button.clicked.connect(self._export_arcos_stats)
        self.widget.param_export_button.clicked.connect(self._export_arcos_params)
        self.widget.param_import_button.clicked.connect(self._import_arcos_params)
        self.widget.img_seq_export_button.clicked.connect(self._export_image_series)
        self.widget.batch_processing_button.clicked.connect(self.batch_processing)
        self.widget.abort_batch_button.clicked.connect(self._abort_batch)

    def batch_processing(self):
        inpath, what_to_export = self.widget._browse_batch_output()
        if inpath is None:
            return
        self.batch_worker = BatchProcessor(
            inpath,
            self._data_storage_instance,
            what_to_export,
        )

        self.show_activity_dock(True)
        # Connect signals and slots - assuming batch_worker has finished, aborted, and run methods, and signals
        self.batch_worker.finished.connect(self._on_batch_finish)
        self.batch_worker.errored.connect(self._batch_error)

        self.batch_worker.new_total_files.connect(self.update_progress_files)
        self.batch_worker.new_total_filters.connect(self.update_progress_filters)
        self.batch_worker.finished.connect(self.abort_timer_stop)
        self.batch_worker.aborted.connect(self.abort_timer_stop)

        self.batch_worker.start()
        self.widget._show_abort_batch_button()

    def update_progress_files(self, value):
        if value < 2:
            self.pbar_files = progress(desc="Files")
            return
        self.pbar_files = progress(total=value, desc="Files")
        self.batch_worker.progress_update_files.connect(
            lambda: self.pbar_files.update(1)
        )
        self.batch_worker.aborted.connect(self._close_progress)

    def update_progress_filters(self, value):
        if value < 2:
            return
        self.pbar_filters = progress(
            total=value, nest_under=self.pbar_files, desc="Positions / Filters"
        )
        self.batch_worker.progress_update_filters.connect(
            lambda: self.pbar_filters.update(1)
        )

    def abort_worker(self):
        self.batch_worker.quit()

    def abort_timer_start(self):
        self.abort_timer.start(1000)

    def abort_timer_timeout(self):
        self.update_abort_button()

    def abort_timer_stop(self):
        self.abort_timer.stop()
        self._close_progress()
        self.widget.abort_batch_button.setText("Abort Batch Processing")

    def update_abort_button(self):
        self.dots = self.widget.abort_batch_button.text().count(".")
        if self.dots < 3:
            self.widget.abort_batch_button.setText(
                self.widget.abort_batch_button.text() + "."
            )
        else:
            self.widget.abort_batch_button.setText("Aborting")

    def _on_batch_finish(self):
        show_info("Batch processing finished")
        self.widget._hide_abort_batch_button()
        self._close_progress()

    def _close_progress(self):
        try:
            self.pbar_filters.close()
        except AttributeError:
            pass
        try:
            self.pbar_files.close()
        except AttributeError:
            pass
        self.show_activity_dock(False)

    def _batch_error(self, error_message: Exception):
        show_info(f"{error_message}")
        traceback.print_exception(None, error_message, error_message.__traceback__)

    def show_activity_dock(self, state=True):
        # show/hide activity dock if there is actual progress to see
        try:
            with warnings.catch_warnings():
                # suppress FutureWarning for now: https://github.com/napari/napari/issues/4598
                warnings.simplefilter(action="ignore", category=FutureWarning)
                self.viewer.window._status_bar._toggle_activity_dock(state)
        except AttributeError:
            print("show_activity_dock failed")

    def closeEvent(self):
        self._abort_batch()

    def _abort_batch(self):
        try:
            self.batch_worker.quit()
            self.widget.abort_batch_button.setEnabled(False)
            self.widget.abort_batch_button.setText("Aborting")
            self.abort_timer_start()
        except (AttributeError, RuntimeError):
            pass
        show_info("Aborting Batch Processing")
