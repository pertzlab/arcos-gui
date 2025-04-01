"""Configuration file for the ARCOS GUI."""

import operator

from napari.utils import Colormap

OPERATOR_DICTIONARY = {
    "Divide": (operator.truediv, "Measurement_Ratio"),
    "Multiply": (operator.mul, "Measurement_Product"),
    "Add": (operator.add, "Measurement_Sum"),
    "Subtract": (operator.sub, "Measurement_Difference"),
}

COLOR_CYCLE = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

TAB20 = Colormap(COLOR_CYCLE, "tab20", interpolation="zero")

ARCOS_LAYERS = {
    "all_cells": "All Cells",
    "active_cells": "Active Cells",
    "collective_events_cells": "Collective Events Cells",
    "event_hulls": "Collective Events",
    "event_boundingbox": "Event Bounding Box",
    "timestamp": "Timestamp",
}

AVAILABLE_OPTIONS_FOR_BATCH = [
    "arcos_output",
    "arcos_stats",
    "per_frame_statistics",
    "statsplot",
    "noodleplot",
    "timelapse_frames",
]

ALLOWED_SETTINGS = [
    "file_name",
    "columns",
    "arcos_parameters",
    "min_max_meas",
    "point_size",
    "lut",
    "output_order",
    "min_max_tracklenght",
]


ARCOSPARAMETERS_DEFAULTS = {
    "interpolate_meas": False,
    "clip_measurements": False,
    "clip_low": 0.0,
    "clip_high": 1.0,
    "smooth_k": 1,
    "bias_k": 5,
    "bias_method": "none",
    "polynomial_degree": 1,
    "bin_threshold": 0.5,
    "bin_peak_threshold": 0.5,
    "eps_method": "manual",
    "neighbourhood_size": 20.0,
    "eps_prev": 20.0,
    "min_clustersize": 5,
    "nprev": 1,
    "min_dur": 1,
    "total_event_size": 5,
    "add_convex_hull": True,
    "add_all_cells": True,
    "add_bin_cells": True,
    "bin_advanded_settings": False,
    "detect_advanced_options": False,
}

DEFAULT_BIN_COLOR = "black"
DEFAULT_ALL_CELLS_CMAP = "inferno"
