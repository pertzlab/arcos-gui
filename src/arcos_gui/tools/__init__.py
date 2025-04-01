"""Utility functoins to handle data processing."""

__author__ = """Benjamin Graedel"""
__email__ = "benjamin.graedel@unibe.ch"
__version__ = "0.1.5"


from arcos_gui.tools._config import (
    ALLOWED_SETTINGS,
    ARCOS_LAYERS,
    ARCOSPARAMETERS_DEFAULTS,
    AVAILABLE_OPTIONS_FOR_BATCH,
    COLOR_CYCLE,
    OPERATOR_DICTIONARY,
    TAB20,
)
from arcos_gui.tools._image_sequence_export import MovieExporter
from arcos_gui.tools._plots import CollevPlotter, NoodlePlot, TimeSeriesPlots
from arcos_gui.tools._shape_functions import (
    fix_3d_convex_hull,
    get_bbox,
    get_bbox_3d,
    get_verticesHull,
    make_surface_3d,
    reshape_by_input_string,
)

from ._ui_util_func import (
    BatchFileDialog,
    OutputOrderValidator,
    ParameterFileDialog,
    ThrottledCallback,
    get_layer_list,
    remove_layers_after_columnpicker,
    set_track_lenths,
)

__all__ = [
    "ALLOWED_SETTINGS",
    "ARCOSPARAMETERS_DEFAULTS",
    "AVAILABLE_OPTIONS_FOR_BATCH",
    "OPERATOR_DICTIONARY",
    "COLOR_CYCLE",
    "TAB20",
    "ARCOS_LAYERS",
    "MovieExporter",
    "NoodlePlot",
    "CollevPlotter",
    "TimeSeriesPlots",
    "remove_layers_after_columnpicker",
    "get_layer_list",
    "set_track_lenths",
    "get_bbox",
    "get_bbox_3d",
    "fix_3d_convex_hull",
    "get_verticesHull",
    "make_surface_3d",
    "reshape_by_input_string",
    "ThrottledCallback",
    "BatchFileDialog",
    "OutputOrderValidator",
    "ParameterFileDialog",
]
