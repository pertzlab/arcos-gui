"""Widgets."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = "0.1.5"

from arcos_gui.widgets._arcos_widget import ArcosController
from arcos_gui.widgets._bottom_bar_widget import BottombarController
from arcos_gui.widgets._dialog_widgets import columnpicker
from arcos_gui.widgets._exporting_widget import ExportController
from arcos_gui.widgets._filter_widget import FilterController
from arcos_gui.widgets._input_data_widget import InputdataController
from arcos_gui.widgets._plot_widgets import collevPlotWidget, tsPlotWidget
from arcos_gui.widgets._visualization_settings_widget import LayerpropertiesController

__all__ = [
    "ArcosController",
    "ExportController",
    "FilterController",
    "InputdataController",
    "LayerpropertiesController",
    "tsPlotWidget",
    "collevPlotWidget",
    "columnpicker",
    "BottombarController",
]
