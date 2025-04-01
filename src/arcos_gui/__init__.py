"""Arcos GUI Plugin for Napari."""

__version__ = "0.1.5"


from arcos_gui._helper_functions import (
    filter_data,
    get_arcos_output,
    get_current_arcos_plugin,
    load_dataframe,
    load_dataframe_with_columnpicker,
    load_sample_data,
    open_plugin,
    run_arcos,
    run_binarization_only,
)
from napari.utils.notifications import show_info

show_info("Loading Plugin")

__all__ = [
    "open_plugin",
    "load_dataframe",
    "load_sample_data",
    "load_dataframe_with_columnpicker",
    "filter_data",
    "run_binarization_only",
    "run_arcos",
    "get_arcos_output",
    "get_current_arcos_plugin",
]
