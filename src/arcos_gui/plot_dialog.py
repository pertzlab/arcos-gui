import os
from pathlib import Path

from qtpy import uic
from qtpy.QtCore import QFile
from qtpy.QtWidgets import QDialog, QVBoxLayout


class CollevPlotDialog(QDialog):

    evplot_layout: QVBoxLayout
    evplot_layout_2: QVBoxLayout

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

    def load_ui(self):
        loader = uic
        path = os.fspath(Path(__file__).parent / "_ui" / "plot_dialog.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.loadUi(ui_file, self)
        ui_file.close()


class DataPlot(QDialog):

    tsplot_layout: QVBoxLayout

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

    def load_ui(self):
        loader = uic
        path = os.fspath(Path(__file__).parent / "_ui" / "data_plot_dialog.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.loadUi(ui_file, self)
        ui_file.close()
