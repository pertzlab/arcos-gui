"""Module for arcos wrapper functions."""

from __future__ import annotations

import math
import os
import warnings
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arcos4py import ARCOS
from arcos4py.plotting import NoodlePlot, statsPlots
from arcos4py.plotting._plotting import _yield_animation_frames
from arcos4py.tools import (
    calculate_statistics,
    calculate_statistics_per_frame,
    estimate_eps,
    filterCollev,
)
from arcos4py.tools._detect_events import DataFrameTracker, Linker
from arcos_gui.processing._data_storage import ArcosParameters, DataStorage, columnnames
from arcos_gui.processing._preprocessing_utils import (
    calculate_measurement,
    check_for_collid_column,
    create_file_names,
    create_output_folders,
    filter_data,
)
from arcos_gui.tools._config import (
    AVAILABLE_OPTIONS_FOR_BATCH,
    DEFAULT_ALL_CELLS_CMAP,
    DEFAULT_BIN_COLOR,
    OPERATOR_DICTIONARY,
)
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from qtpy.QtCore import Signal
from tqdm.auto import tqdm


class customARCOS(ARCOS):
    """Custom ARCOS class with replaced trackCollev method.

    The trackCollev method is replaced with a custom version that emits a signal
    for progress updates. This signal is connected to the ArcosWidget, which
    updates the progress bar. The custom trackCollev method also checks for the
    abort_requested flag and aborts the tracking if it is set to True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._abort_requested = False
        self.progress_update = None

    def trackCollev(
        self,
        eps: float = 1,
        eps_prev: float | None = None,
        minClsz: int = 1,
        n_prev: int = 1,
        clustering_method: str = "dbscan",
        linking_method: str = "nearest",
        min_samples: int | None = None,
    ) -> pd.DataFrame:
        linker = Linker(
            eps=eps,
            eps_prev=eps_prev,
            min_clustersize=minClsz,
            min_samples=min_samples,
            clustering_method=clustering_method,
            linking_method=linking_method,
            n_prev=n_prev,
            predictor=False,
            n_jobs=1,
        )
        tracker = DataFrameTracker(
            linker=linker,
            position_columns=self.position_columns,
            frame_column=self.frame_column,
            obj_id_column=self.obj_id_column,
            binarized_measurement_column=self.binarized_measurement_column,
            clid_column=self.clid_column,
        )
        df_list = []

        total = self.data[self.frame_column].nunique()

        if self.progress_update:
            self.progress_update.emit("total", total)

        for timepoint in tracker.track(self.data):
            if self.abort_requested:
                self._abort_requested = False
                return pd.DataFrame()
            df_list.append(timepoint)

            if self.progress_update:
                self.progress_update.emit("update", 1)

        if self.progress_update:
            self.progress_update.emit("reset", 0)

        df_out = pd.concat(df_list, axis=0)

        return df_out.query(f"{self.clid_column} != -1").reset_index(drop=True)

    def quit(self):
        self._abort_requested = True

    @property
    def abort_requested(self):
        return self._abort_requested


def empty_std_out(*args, **kwargs):
    pass


def init_arcos_object(
    df_in: pd.DataFrame,
    position_columns: list,
    measurement_name: str,
    frame_col_name: str,
    track_id_col_name: str,
    progress_update_signal: Signal | None = None,
):
    """
    Initialize arcos object from pandas dataframe

    Parameters
    ----------
    df_in : pd.DataFrame
        input dataframe
    position_columns : list
        list of position columns
    measurement_name : str
        name of measurement column
    frame_col_name : str
        name of frame column
    track_id_col_name : str
        name of track id column
    std_out_func : Callable
        function to print to console or gui
    """

    collid_name = "collid"

    df_in = check_for_collid_column(df_in, collid_name)

    # checks if this part of the function has to be run,
    # depends on the parameters changed in arcos widget

    # create arcos object, run arcos
    arcos = customARCOS(
        data=df_in,
        position_columns=position_columns,
        frame_column=frame_col_name,
        obj_id_column=track_id_col_name,
        measurement_column=measurement_name,
        clid_column=collid_name,
    )
    arcos.progress_update = progress_update_signal
    return arcos


def binarization(
    arcos: customARCOS,
    interpolate_meas: bool,
    clip_measurements: bool,
    clip_low: float,
    clip_high: float,
    smooth_k: int,
    bias_k: int,
    bin_peak_threshold: float,
    bin_threshold: float,
    polynomial_degree: int,
    bias_method: str,
) -> customARCOS:
    """
    Binarize measurement data

    Parameters
    ----------
    arcos : customARCOS
        arcos object
    interpolate_meas : bool
        interpolate measurement data
    clip_measurements : bool
        clip measurement data
    clip_low : float
        lower clip value
    clip_high : float
        higher clip value
    smooth_k : int
        smoothing kernel size
    bias_k : int
        bias kernel size
    bin_peak_threshold : float
        peak threshold
    bin_threshold : float
        binarization threshold
    polynomial_degree : int
        polynomial degree
    bias_method : str
        bias method

    Returns
    -------
    arcos_bin : customARCOS
        arcos object with binarized measurement data
    """
    # if corresponding checkbox was selected run interpolate measurments
    if interpolate_meas:
        arcos.interpolate_measurements()

    # if corresponding checkbock was selected run clip_measuremnts
    if clip_measurements:
        arcos.clip_measurements(
            clip_low=clip_low,
            clip_high=clip_high,
        )

    # binarize data and update ts variable
    # update from where to run
    arcos.bin_measurements(
        smooth_k=smooth_k,
        bias_k=bias_k,
        peak_threshold=bin_peak_threshold,
        binarization_threshold=bin_threshold,
        polynomial_degree=polynomial_degree,
        bias_method=bias_method,
    )

    return arcos


def detect_events(
    arcos: customARCOS,
    neighbourhood_size: float,
    eps_prev: float | None,
    min_clustersize: int,
    nPrev_value: int,
):
    """
    Detect collective events with arcos trackCollev method.

    Parameters
    ----------
    arcos : customARCOS
        arcos object
    measbin_col : str
        name of binarized measurement column
    neighbourhood_size : float
        neighbourhood size to consider for event detection
    min_clustersize : int
        minimum cluster size to consider for event detection
    nPrev_value : int
        number of previous frames to consider for event detection

    Returns
    -------
    arcos_events : pd.DataFrame
        dataframe with detected events
    """
    _bin_col = arcos.binarized_measurement_column
    if 1 not in arcos.data[_bin_col].values:
        return None

    arcos_events = arcos.trackCollev(
        eps=neighbourhood_size,
        eps_prev=eps_prev,
        minClsz=min_clustersize,
        n_prev=nPrev_value,
    )
    return arcos_events


def get_eps(arcos: customARCOS, method: str, minClustersize: int, current_eps: float):
    """
    Estimate eps value for arcos trackCollev method.

    Parameters
    ----------
    arcos : customARCOS
        arcos object
    method : str
        method to estimate eps value
    minClustersize : int
        minimum cluster size to consider for event detection
    current_eps : float | None
        current eps value, will be returned if method is manual

    Returns
    -------
    eps : float
        eps value
    """
    methods = ["manual", "kneepoint", "mean"]
    if method not in methods:
        raise ValueError(f"Method must be one of {methods}")

    if method == "kneepoint":
        eps = estimate_eps(
            data=arcos.data[arcos.data[arcos.binarized_measurement_column] == 1],
            method="kneepoint",
            position_columns=arcos.position_columns,
            frame_column=arcos.frame_column,
            n_neighbors=minClustersize,
            plot=False,
        )
        return round(eps, 2)

    if method == "mean":
        eps = estimate_eps(
            arcos.data,
            method="mean",
            position_columns=arcos.position_columns,
            frame_column=arcos.frame_column,
            n_neighbors=minClustersize,
            plot=False,
        )
        return round(eps, 2)
    return round(current_eps, 2)


def filtering_arcos_events(
    detected_events_df: pd.DataFrame,
    frame_col_name: str,
    collid_name: str,
    track_id_col_name: str,
    min_dur: int,
    total_event_size: int,
):
    """
    Filter detected events with arcos filterCollev method.

    Parameters
    ----------
    detected_events_df : pd.DataFrame
        dataframe with detected events
    frame_col_name : str
        name of frame column
    collid_name : str
        name of collid column
    track_id_col_name : str
        name of track id column
    min_dur : int
        minimum duration of events
    total_event_size : int
        minimum size of events

    Returns
    -------
    filtered_events_df : pd.DataFrame
        dataframe with filtered events
    """
    if track_id_col_name:
        filterer = filterCollev(
            data=detected_events_df,
            frame_column=frame_col_name,
            clid_column=collid_name,
            obj_id_column=track_id_col_name,
        )
        arcos_filtered = filterer.filter(
            min_duration=min_dur,
            min_total_size=total_event_size,
        )
    else:
        # filter dataframe by duraiton of events
        detect_events_df = detected_events_df.copy()
        detect_events_df["duration"] = detect_events_df.groupby(
            [frame_col_name, collid_name]
        )[frame_col_name].transform("count")
        arcos_filtered = detect_events_df[detect_events_df["duration"] >= min_dur]
        arcos_filtered = arcos_filtered.drop(columns=["duration"])

    # makes filtered collids sequential
    clid_np = arcos_filtered[collid_name].to_numpy()
    clids_sorted_i = np.argsort(clid_np)
    clids_reverse_i = np.argsort(clids_sorted_i)
    clid_np_sorted = clid_np[(clids_sorted_i)]
    grouped_array_clids = np.split(
        clid_np_sorted,
        np.unique(clid_np_sorted, axis=0, return_index=True)[1][1:],
    )
    seq_colids = np.concatenate(
        [np.repeat(i, value.shape[0]) for i, value in enumerate(grouped_array_clids)],
        axis=0,
    )[clids_reverse_i]
    seq_colids_from_one = np.add(seq_colids, 1)
    arcos_filtered = arcos_filtered.copy()
    arcos_filtered.loc[:, collid_name] = seq_colids_from_one

    return arcos_filtered


def calculate_arcos_stats(
    df_arcos_filtered: pd.DataFrame,
    frame_column: str,
    collid_name: str,
    object_id_name: str,
    position_columns: list,
):
    """Wrapper for calcCollevStats().

    Parameters
    ----------
    df_arcos_filtered : pd.DataFrame
        dataframe with filtered events
    frame_column : str
        name of frame column
    collid_name : str
        name of collid column
    object_id_col_name : str
        name of object id column
    position_columns : list
        list of position columns

    Returns
    -------
    df_arcos_stats : pd.DataFrame
        dataframe with statistics for each event
    """
    df_arcos_stats = calculate_statistics(
        df_arcos_filtered, frame_column, collid_name, object_id_name, position_columns
    )
    return df_arcos_stats


class arcos_worker_base_signals(WorkerBaseSignals):
    binarization_finished = Signal(tuple)
    tracking_finished = Signal()
    new_arcos_output = Signal(tuple)
    new_eps = Signal(float)
    started = Signal()
    finished = Signal()
    aborted = Signal(object)
    arcos_progress_update = Signal(str, int)


class arcos_worker(WorkerBase):
    """Runs arcos with the current parameters defined in the ArcosWidget.

    Updates the data storage with the results. what_to_run is a set of strings
    indicating what to run. The strings correspond to specific steps in the
    arcos pipeline. The steps are:
        - 'binarization': initializes a new customARCOS object and runs the binarization.
        - 'tracking': runs the event detection.
        - 'filtering': runs the event filtering.
    """

    def __init__(
        self,
        what_to_run: set,
        std_out: Callable,
        arcos_parameters: ArcosParameters = ArcosParameters(),
        columns: columnnames = columnnames(),
        filtered_data: pd.DataFrame = pd.DataFrame(),
        arcos_object: customARCOS | None = None,
        arcos_raw_output: pd.DataFrame | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        what_to_run : set
            set of strings indicating what to run
        std_out : Callable
            function to print to the console
        wait_for_parameter_update : bool, optional
            if True, the worker will wait for the parameters to be updated before running
        """
        super().__init__(SignalsClass=arcos_worker_base_signals)
        self.what_to_run = what_to_run
        self.std_out = std_out
        self.arcos_parameters: ArcosParameters = arcos_parameters
        self.columns: columnnames = columns
        self.filtered_data: pd.DataFrame = filtered_data
        if arcos_object is None:
            arcos_object = init_arcos_object(
                df_in=pd.DataFrame(columns=["x", "t", "m", "id"]),
                position_columns=["x"],
                frame_col_name="t",
                track_id_col_name="id",
                measurement_name="m",
                progress_update_signal=self.arcos_progress_update,
            )
        else:
            # attatch progress signal to arcos object if the arcos object is not None
            arcos_object.progress_update = self.arcos_progress_update

        self.arcos_object: customARCOS = arcos_object

        if arcos_raw_output is None:
            arcos_raw_output = pd.DataFrame(
                columns=["t", "id", "collid", "x", "y", "m"], data=[]
            )
        self.arcos_raw_output: pd.DataFrame = arcos_raw_output

    def quit(self) -> None:
        """Quit the worker. Sets the abort_requested flag to True.
        Reimplemented so that it also sets the arcos_abort_requested flag to True.
        """
        self._abort_requested = True
        self.arcos_object.quit()
        super().quit()

    def run_binarization(self):
        if self.filtered_data.empty:
            self.std_out("No data loaded. Load first using the import data tab.")
            return
        self.started.emit()

        if self.abort_requested:
            return
        self.arcos_object = init_arcos_object(
            df_in=self.filtered_data,
            position_columns=self.columns.posCol,
            measurement_name=self.columns.measurement_column,
            frame_col_name=self.columns.frame_column,
            track_id_col_name=self.columns.object_id,
            progress_update_signal=self.arcos_progress_update,
        )
        if self.abort_requested:
            return

        self.arcos_object = binarization(
            arcos=self.arcos_object,
            interpolate_meas=self.arcos_parameters.interpolate_meas.value,
            clip_measurements=self.arcos_parameters.clip_measurements.value,
            clip_low=self.arcos_parameters.clip_low.value,
            clip_high=self.arcos_parameters.clip_high.value,
            smooth_k=self.arcos_parameters.smooth_k.value,
            bias_k=self.arcos_parameters.bias_k.value,
            polynomial_degree=self.arcos_parameters.polynomial_degree.value,
            bin_threshold=self.arcos_parameters.bin_threshold.value,
            bin_peak_threshold=self.arcos_parameters.bin_peak_threshold.value,
            bias_method=self.arcos_parameters.bias_method.value,
        )
        if self.abort_requested:
            return
        self.binarization_finished.emit(
            (
                self.arcos_object.binarized_measurement_column,
                self.arcos_object.resc_col,
                self.arcos_object.data,
            )
        )
        self.columns.measurement_bin = self.arcos_object.binarized_measurement_column
        self.columns.measurement_resc = self.arcos_object.resc_col
        self.what_to_run.remove("binarization")

    def run_tracking(self):
        try:
            binarized_measurement_column = self.columns.measurement_bin
            n_bin = self.arcos_object.data[binarized_measurement_column].nunique()

        except KeyError:
            n_bin = 0
        if n_bin < 2:
            self.std_out("No Binarized Data. Adjust Binazation Parameters.")
            return

        if self.abort_requested:
            return

        eps = get_eps(
            arcos=self.arcos_object,
            method=self.arcos_parameters.eps_method.value,
            minClustersize=self.arcos_parameters.min_clustersize.value,
            current_eps=self.arcos_parameters.neighbourhood_size.value,
        )

        if self.abort_requested:
            return

        self.new_eps.emit(eps)

        self.arcos_raw_output = detect_events(
            arcos=self.arcos_object,
            neighbourhood_size=eps,
            eps_prev=self.arcos_parameters.eps_prev.value,
            min_clustersize=self.arcos_parameters.min_clustersize.value,
            nPrev_value=self.arcos_parameters.nprev.value,
        )
        if self.abort_requested:
            return
        self.tracking_finished.emit()
        self.what_to_run.remove("tracking")

    def run_filtering(self):
        if self.arcos_raw_output.empty:
            self.std_out(
                "No Collective Events detected. Adjust Event Detection Parameters."
            )
            return

        collid_name = "collid"
        if self.abort_requested:
            return
        arcos_df_filtered = filtering_arcos_events(
            detected_events_df=self.arcos_raw_output,
            frame_col_name=self.columns.frame_column,
            collid_name=collid_name,
            track_id_col_name=self.columns.object_id,
            min_dur=self.arcos_parameters.min_dur.value,
            total_event_size=self.arcos_parameters.total_event_size.value,
        )
        if arcos_df_filtered.empty:
            self.std_out("No Collective Events detected.Adjust Filtering parameters.")
            return
        if self.abort_requested:
            return
        arcos_stats = calculate_arcos_stats(
            df_arcos_filtered=arcos_df_filtered,
            frame_column=self.columns.frame_column,
            collid_name=collid_name,
            object_id_name=self.columns.object_id,
            position_columns=self.columns.posCol,
        )
        arcos_stats = arcos_stats.dropna()
        if self.abort_requested:
            return
        self.new_arcos_output.emit((arcos_df_filtered, arcos_stats))
        self.what_to_run.clear()

    def run(
        self,
    ):
        """Run arcos with input parameters.

        Runs only or only from as far as specified in the what_to_run set.
        """
        self.started.emit()

        try:
            if "binarization" in self.what_to_run and not self.abort_requested:
                self.run_binarization()

            if "tracking" in self.what_to_run and not self.abort_requested:
                self.run_tracking()

            if "filtering" in self.what_to_run and not self.abort_requested:
                self.run_filtering()

        except Exception as e:
            self.errored.emit(e)

        finally:
            self.finished.emit()


class TemporaryMatplotlibBackend:
    def __init__(self, backend="Agg"):
        self.temp_backend = backend
        self.original_backend = matplotlib.get_backend()

    def __enter__(self):
        # set svg rc parameters to use real font
        plt.rcParams["svg.fonttype"] = "none"
        plt.switch_backend(self.temp_backend)

    def __exit__(self, *args):
        plt.switch_backend(self.original_backend)


class BatchProcessorSignals(WorkerBaseSignals):
    finished = Signal()
    progress_update_files = Signal()
    progress_update_filters = Signal()
    new_total_files = Signal(int)
    new_total_filters = Signal(int)
    aborted = Signal()


class BatchProcessor(WorkerBase):
    """Runs Arcos in batch mode with the current parameters defined in the ArcosWidget."""

    def __init__(
        self,
        input_path: str,
        data_storage_instance: DataStorage,
        what_to_export: list[str],
    ):
        super().__init__(SignalsClass=BatchProcessorSignals)
        self.input_path = input_path
        self.data_storage_instance = data_storage_instance
        self.arcos_parameters = data_storage_instance.arcos_parameters.value
        self.columnames = data_storage_instance.columns.value
        self.min_track_length = data_storage_instance.min_max_tracklenght.value[0]
        self.max_track_length = data_storage_instance.min_max_tracklenght.value[1]
        self.what_to_export = what_to_export

    def _create_fileendings_list(self):
        """Create a list o file endings for the files to be exported."""
        corresponding_fileendings = [
            ".csv",
            ".csv",
            ".csv",
            ".svg",
            ".svg",
            "",
        ]  # Marker for directory
        endings_or_markers = []
        # Use the globally defined (or class-level) lists
        option_to_ending_map = dict(
            zip(AVAILABLE_OPTIONS_FOR_BATCH, corresponding_fileendings)
        )

        for option in self.what_to_export:
            if option in option_to_ending_map:
                endings_or_markers.append(option_to_ending_map[option])
            else:
                # This case should ideally be caught by validation earlier
                print(
                    f"Warning: Unknown export option '{option}' encountered when creating endings list."
                )
                endings_or_markers.append("")  # Add placeholder if needed
        return endings_or_markers

    def run_arcos_batch(self, df):
        """Run arcos with input parameters.

        Runs only or only from as far as specified in the what_to_run set.
        """
        arcos = init_arcos_object(
            df,
            self.columnames.posCol,
            self.columnames.measurement_column,
            self.columnames.frame_column,
            self.columnames.object_id,
        )
        arcos = binarization(
            arcos=arcos,
            interpolate_meas=self.arcos_parameters.interpolate_meas.value,
            clip_measurements=self.arcos_parameters.clip_measurements.value,
            clip_low=self.arcos_parameters.clip_low.value,
            clip_high=self.arcos_parameters.clip_high.value,
            smooth_k=self.arcos_parameters.smooth_k.value,
            bias_k=self.arcos_parameters.bias_k.value,
            polynomial_degree=self.arcos_parameters.polynomial_degree.value,
            bin_threshold=self.arcos_parameters.bin_threshold.value,
            bin_peak_threshold=self.arcos_parameters.bin_peak_threshold.value,
            bias_method=self.arcos_parameters.bias_method.value,
        )
        eps = get_eps(
            arcos=arcos,
            method=self.arcos_parameters.eps_method.value,
            minClustersize=self.arcos_parameters.min_clustersize.value,
            current_eps=self.arcos_parameters.neighbourhood_size.value,
        )
        arcos_raw_output = detect_events(
            arcos=arcos,
            neighbourhood_size=eps,
            eps_prev=self.arcos_parameters.eps_prev.value,
            min_clustersize=self.arcos_parameters.min_clustersize.value,
            nPrev_value=self.arcos_parameters.nprev.value,
        )
        arcos_df_filtered = filtering_arcos_events(
            detected_events_df=arcos_raw_output,
            frame_col_name=self.columnames.frame_column,
            collid_name="collid",
            track_id_col_name=self.columnames.object_id,
            min_dur=self.arcos_parameters.min_dur.value,
            total_event_size=self.arcos_parameters.total_event_size.value,
        )

        arcos_stats = calculate_arcos_stats(
            df_arcos_filtered=arcos_df_filtered,
            frame_column=self.columnames.frame_column,
            collid_name="collid",
            object_id_name=self.columnames.object_id,
            position_columns=self.columnames.posCol,
        )
        arcos_stats = arcos_stats.dropna()

        return arcos_df_filtered, arcos_stats

    def save_animation_frames(
        self,
        arcos_data: pd.DataFrame,
        all_cells_data: pd.DataFrame,
        output_dir: str,
        # --- Parameters passed to yield_animation_frames ---
        frame_col: str,
        collid_col: str,
        pos_cols: List[str],
        measurement_col: Optional[str] = None,
        bin_col: Optional[str] = None,
        plot_all_cells: bool = True,
        color_all_cells_by_measurement: bool = True,
        plot_bin_cells: bool = True,
        plot_events: bool = True,
        plot_convex_hulls: bool = True,
        point_size: float = 10.0,
        event_alpha: float = 0.9,
        hull_alpha: float = 1,
        hull_linewidth_size_factor: float = 0.25,
        bin_cell_color: str = DEFAULT_BIN_COLOR,
        bin_cell_alpha: float = 0.7,
        bin_cell_marker_factor: float = 0.8,
        all_cells_cmap: str = DEFAULT_ALL_CELLS_CMAP,
        all_cells_fixed_color: str = "gray",
        all_cells_alpha: float = 0.5,
        all_cells_marker_size_factor: float = 0.2,
        measurement_min_max: Optional[Tuple[float, float]] = None,
        add_measurement_colorbar: bool = True,
        # --- Parameters for the save function itself ---
        filename_prefix: str = "frame",
        dpi: int = 150,
    ) -> None:
        # --- Setup Output Directory ---
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving animation frames to directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return  # Cannot proceed without output directory

        # --- Determine Frame Range and Padding (needed for filenames) ---
        min_frame_val = float("inf")
        max_frame_val = float("-inf")
        if not arcos_data.empty and frame_col in arcos_data:
            min_frame_val = min(min_frame_val, arcos_data[frame_col].min())
            max_frame_val = max(max_frame_val, arcos_data[frame_col].max())
        if not all_cells_data.empty and frame_col in all_cells_data:
            min_frame_val = min(min_frame_val, all_cells_data[frame_col].min())
            max_frame_val = max(max_frame_val, all_cells_data[frame_col].max())

        if min_frame_val == float("inf") or max_frame_val == float("-inf"):
            print(
                "Could not determine frame range from input data. No frames will be saved."
            )
            return

        num_total_frames = int(max_frame_val) - int(min_frame_val) + 1
        padding_digits = (
            math.ceil(math.log10(max(1, int(max_frame_val)) + 1))
            if max_frame_val >= 0
            else 1
        )  # Calculate padding based on max frame number

        # --- Instantiate the Generator ---
        frame_generator = _yield_animation_frames(
            arcos_data=arcos_data,
            all_cells_data=all_cells_data,
            frame_col=frame_col,
            collid_col=collid_col,
            pos_cols=pos_cols,
            measurement_col=measurement_col,
            bin_col=bin_col,
            plot_all_cells=plot_all_cells,
            color_all_cells_by_measurement=color_all_cells_by_measurement,
            plot_bin_cells=plot_bin_cells,
            plot_events=plot_events,
            plot_convex_hulls=plot_convex_hulls,
            point_size=point_size,
            event_alpha=event_alpha,
            hull_alpha=hull_alpha,
            hull_linewidth_size_factor=hull_linewidth_size_factor,
            bin_cell_color=bin_cell_color,
            bin_cell_alpha=bin_cell_alpha,
            bin_cell_marker_factor=bin_cell_marker_factor,
            all_cells_cmap=all_cells_cmap,
            all_cells_fixed_color=all_cells_fixed_color,
            all_cells_alpha=all_cells_alpha,
            all_cells_marker_size_factor=all_cells_marker_size_factor,
            measurement_min_max=measurement_min_max,
            add_measurement_colorbar=add_measurement_colorbar,
        )

        # --- Iterate, Save, and Close ---
        saved_frame_count = 0
        print(
            f"Starting frame generation and saving (estimated {num_total_frames} frames)..."
        )

        for fig in tqdm(
            frame_generator, total=num_total_frames, desc="Saving frames", unit="frame"
        ):
            if self.abort_requested:
                self.aborted.emit()
                break
            # Get frame number from the figure title (set by the generator)
            try:
                title = fig.axes[0].get_title()
                # Handle potential variations in title format slightly more robustly
                frame_num_str = title.split(":")[-1].strip()
                frame_num = int(frame_num_str)
            except (IndexError, ValueError, AttributeError) as e:
                warnings.warn(
                    f"Could not reliably determine frame number from figure title ('{title}').\
                        Using counter. Error: {e}"
                )
                # Fallback to a simple counter if title parsing fails
                frame_num = saved_frame_count + int(min_frame_val)  # Estimate frame num

            # Construct filename with padding
            frame_filename = f"{filename_prefix}_{frame_num:0{padding_digits}d}.png"
            output_path = os.path.join(output_dir, frame_filename)

            # Save the figure
            try:
                fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
                saved_frame_count += 1
            except Exception as e:
                print(f"\nError saving frame {output_path}: {e}")
                # Decide whether to continue or stop on error (here we continue)
            finally:
                # CRITICAL: Close the figure to free memory, regardless of save success
                plt.close(fig)

        print(f"\nFinished saving {saved_frame_count} frames to {output_dir}.")
        if saved_frame_count == 0:
            print(
                "Note: No frames were generated or saved. Check input data and parameters."
            )

    def run(self):
        """Run arcos with input parameters.

        Runs only or only from as far as specified in the what_to_run set.
        """
        self.started.emit()
        try:
            summary_stats = summary_stats = {
                key: []
                for key in [
                    "file",
                    "fov",
                    "additional_filter",
                    "event_count",
                    "avg_total_size",
                    "avg_total_size_std",
                    "avg_duration",
                    "avg_duration_std",
                ]
            }  # noqa E501
            with TemporaryMatplotlibBackend("Agg"):
                # check that what_to_export
                if not self.what_to_export:
                    self.errored.emit(ValueError("No export selected"))
                    return

                print(f"Exporting {self.what_to_export}")

                # check that there are only valid export options
                valid_export_options = AVAILABLE_OPTIONS_FOR_BATCH
                for option in self.what_to_export:
                    if option not in valid_export_options:
                        self.errored.emit(ValueError(f"Invalid export option {option}"))
                        return

                # remove statsplot and noodleplot from what_to_export if no object id is given
                if self.columnames.object_id is None:
                    self.what_to_export = [
                        option
                        for option in self.what_to_export
                        if option not in ["statsplot", "noodleplot"]
                    ]
                    print("No object id given. Skipping statsplot and noodleplot.")

                file_list = [
                    os.path.join(self.input_path, file)
                    for file in os.listdir(self.input_path)
                    if file.endswith(".csv") or file.endswith(".csv.gz")
                ]
                self.new_total_files.emit(len(file_list))

                base_path, _ = create_output_folders(
                    self.input_path, self.what_to_export
                )
                self.data_storage_instance.export_to_yaml(
                    filepath=os.path.join(base_path, "arcos_parameters.yaml"),
                )
                for file in file_list:
                    if self.abort_requested:
                        self.aborted.emit()
                        break

                    pth = Path(file)

                    file_name = pth.with_suffix("").stem
                    print(f"Processing file {file_name}")
                    df = pd.read_csv(file, engine="pyarrow")

                    meas_col, df = calculate_measurement(
                        data=df,
                        operation=self.columnames.measurement_math_operation,
                        in_meas_1_name=self.columnames.measurement_column_1,
                        in_meas_2_name=self.columnames.measurement_column_2,
                        op_dict=OPERATOR_DICTIONARY,
                    )
                    self.columnames.measurement_column = meas_col

                    if self.columnames.position_id is not None:
                        position_ids = df[self.columnames.position_id].unique()
                    else:
                        position_ids = [None]

                    if self.columnames.additional_filter_column is not None:
                        additional_filters = df[
                            self.columnames.additional_filter_column
                        ].unique()
                    else:
                        additional_filters = [None]

                    iterator_fov_filter = list(
                        product(position_ids, additional_filters)
                    )
                    self.new_total_filters.emit(len(iterator_fov_filter))

                    for fov, additional_filter in iterator_fov_filter:
                        if self.abort_requested:
                            self.aborted.emit()
                            break

                        # Add new row to summary stats
                        for key in summary_stats.keys():
                            summary_stats[key].append(pd.NA)

                        # Update general stats
                        summary_stats["file"][-1] = file_name
                        summary_stats["fov"][-1] = fov if fov is not None else pd.NA
                        summary_stats["additional_filter"][-1] = (
                            additional_filter
                            if additional_filter is not None
                            else pd.NA
                        )

                        # Filter input data
                        df_filtered = filter_data(
                            df_in=df,
                            field_of_view_id_name=self.columnames.position_id,
                            frame_name=self.columnames.frame_column,
                            track_id_name=self.columnames.object_id,
                            measurement_name=self.columnames.measurement_column,
                            additional_filter_column_name=self.columnames.additional_filter_column,
                            position_columns=self.columnames.posCol,
                            fov_val=fov,
                            additional_filter_value=additional_filter,
                            min_tracklength_value=self.min_track_length,
                            max_tracklength_value=self.max_track_length,
                            frame_interval=1,
                            st_out=empty_std_out,
                        )[0]

                        # Handle empty filtered data
                        if df_filtered.empty:
                            summary_stats["event_count"][-1] = 0

                            # Construct detailed error message
                            position_id_str = (
                                f"{self.columnames.position_id}:{fov}"
                                if self.columnames.position_id and fov is not None
                                else ""
                            )
                            additional_filter_str = (
                                f"{self.columnames.additional_filter_column}:{additional_filter}"
                                if self.columnames.additional_filter_column
                                and additional_filter is not None
                                else ""
                            )
                            connector = (
                                " and "
                                if position_id_str and additional_filter_str
                                else ""
                            )
                            for_str = (
                                "for "
                                if position_id_str or additional_filter_str
                                else ""
                            )

                            print(
                                f"No data for file {file} {for_str}{position_id_str}{connector}{additional_filter_str}"
                            )

                        # Determine dimensionality
                        posx, posy = (
                            self.columnames.posCol[0],
                            self.columnames.posCol[1],
                        )
                        posz = (
                            self.columnames.posCol[2]
                            if len(self.columnames.posCol) == 3
                            else None
                        )

                        # Run ARCOS batch analysis
                        arcos_df_filtered, arcos_stats = self.run_arcos_batch(
                            df_filtered
                        )

                        if arcos_df_filtered.empty:
                            summary_stats["event_count"][-1] = 0
                            print(
                                f"No events detected for file {file} filters fov:{fov} additional:{additional_filter}"
                            )
                        else:
                            # Update summary stats
                            summary_stats["event_count"][-1] = arcos_stats[
                                "collid"
                            ].nunique()
                            summary_stats["avg_total_size"][-1] = arcos_stats[
                                "total_size"
                            ].mean()
                            summary_stats["avg_total_size_std"][-1] = arcos_stats[
                                "total_size"
                            ].std()
                            summary_stats["avg_duration"][-1] = arcos_stats[
                                "duration"
                            ].mean()
                            summary_stats["avg_duration_std"][-1] = arcos_stats[
                                "duration"
                            ].std()

                        out_file_name = create_file_names(
                            base_path,
                            file_name,
                            self.what_to_export,
                            self._create_fileendings_list(),
                            fov,
                            additional_filter,
                            self.columnames.position_id,
                            self.columnames.additional_filter_column,
                        )
                        if "arcos_output" in self.what_to_export:
                            arcos_df_filtered.to_csv(
                                out_file_name["arcos_output"],
                                index=False,
                            )
                        if "arcos_stats" in self.what_to_export:
                            arcos_stats.to_csv(
                                out_file_name["arcos_stats"],
                                index=False,
                            )
                        if "per_frame_statistics" in self.what_to_export:
                            # Compute stats per frame
                            arcos_stats_per_frame = calculate_statistics_per_frame(
                                data=arcos_df_filtered,
                                frame_column=self.columnames.frame_column,
                                clid_column="collid",
                                pos_columns=self.columnames.posCol,
                            )
                            arcos_stats_per_frame.to_csv(
                                out_file_name["per_frame_statistics"],
                                index=False,
                            )

                        if "statsplot" in self.what_to_export:
                            # seaborn future warning is annoying
                            with warnings.catch_warnings():
                                warnings.simplefilter(
                                    action="ignore", category=FutureWarning
                                )
                                arcos_stats_plot = statsPlots(arcos_stats)
                                arcos_stats_plot.plot_events_duration(
                                    "total_size", "duration"
                                )
                                plt.savefig(out_file_name["statsplot"])
                                plt.close()

                        if "noodleplot" in self.what_to_export:
                            noodle_plot = NoodlePlot(
                                df=arcos_df_filtered,
                                colev=self.columnames.collid_name,
                                trackid=self.columnames.object_id,
                                frame=self.columnames.frame_column,
                                posx=posx,
                                posy=posy,
                                posz=posz,
                            )
                            _, ax = noodle_plot.plot(posx)
                            min_frame = df[self.columnames.frame_column].min()
                            max_frame = df[self.columnames.frame_column].max()
                            ax.set_xlim(min_frame, max_frame)
                            min_pos = df[posx].min()
                            max_pos = df[posx].max()
                            ax.set_ylim(min_pos, max_pos)
                            ax.set_xlabel(self.columnames.frame_column)
                            ax.set_ylabel(posx)

                            plt.savefig(out_file_name["noodleplot"])
                            plt.close()

                        if "timelapse_frames" in self.what_to_export:
                            frame_output_dir = out_file_name["timelapse_frames"]
                            # Create a prefix based on the original file/filters
                            frame_prefix = "frame"
                            self.save_animation_frames(
                                arcos_data=arcos_df_filtered,
                                all_cells_data=df_filtered,
                                frame_col=self.columnames.frame_column,
                                collid_col="collid",
                                pos_cols=self.columnames.posCol,
                                measurement_col=self.columnames.measurement_column,
                                bin_col=self.columnames.measurement_bin,
                                plot_all_cells=self.arcos_parameters.add_all_cells.value,
                                plot_bin_cells=self.arcos_parameters.add_bin_cells.value,
                                plot_events=True,
                                plot_convex_hulls=self.arcos_parameters.add_convex_hull.value,
                                point_size=self.data_storage_instance.point_size.value,
                                event_alpha=0.9,
                                hull_alpha=0.9,
                                bin_cell_color="black",
                                bin_cell_alpha=0.7,
                                bin_cell_marker_factor=0.5,
                                all_cells_cmap=self.data_storage_instance.lut.value,
                                all_cells_alpha=0.7,
                                all_cells_marker_size_factor=10,
                                measurement_min_max=self.data_storage_instance.min_max_meas.value,
                                add_measurement_colorbar=True,
                                output_dir=frame_output_dir,
                                filename_prefix=frame_prefix,
                            )

                        self.progress_update_filters.emit()

                    self.progress_update_files.emit()

            summary_stats_df = pd.DataFrame(summary_stats).round(4)
            # drop rows with all nan
            summary_stats_df = summary_stats_df.dropna(how="all", axis=1)
            summary_stats_df.to_csv(
                os.path.join(base_path, "per_file_summary.csv"),
                index=False,
                na_rep="NA",
            )

        except Exception as e:
            self.errored.emit(e)
        finally:
            self.finished.emit()
