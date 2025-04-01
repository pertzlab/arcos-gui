"""Contains functions for preprocessing csv data."""

from __future__ import annotations

import csv
import gzip
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Union

import pandas as pd
from arcos_gui.tools import OPERATOR_DICTIONARY
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from qtpy.QtCore import Signal
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    import napari

    from ._data_storage import columnnames


def create_output_folders(base_path, subfolders):
    # Get the current date in the desired format
    current_date = datetime.today().strftime("%Y%m%d")

    # Create the base directory structure
    base_directory = os.path.join(base_path, "arcos_output", current_date)
    counter = 0
    while os.path.exists(base_directory):
        counter += 1
        base_directory = os.path.join(
            base_path, "arcos_output", f"{current_date}_{counter}"
        )

    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)

    # Create subfolders inside the base directory
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_directory, subfolder), exist_ok=True)

    return base_directory, subfolders


def create_file_names(
    base_path: str,
    file_name: str,
    what_to_export: list[str],
    file_extension: list[str],
    fov=None,
    additional_filter=None,
    fov_name=None,
    additional_filter_name=None,
):
    # Create a dictionary to hold the file paths
    file_paths = {}

    # Create a file name for each item in what_to_export
    for file_to_export, ext in zip(what_to_export, file_extension):
        file_name_current = file_name
        if fov is not None and fov_name is not None:
            file_name_current += f"_{fov_name}{fov}"
        if additional_filter is not None and additional_filter_name is not None:
            file_name_current += f"_{additional_filter_name}{additional_filter}"
        file_paths[file_to_export] = os.path.join(
            base_path, file_to_export, f"{file_name_current}{ext}"
        )

    return file_paths


def subtract_timeoffset(data, frame_column):
    """Method to subtract the timeoffset in the frame column of data"""
    if data.empty:
        return data
    data[frame_column] -= min(data[frame_column])
    return data


def calculate_measurement(
    data: pd.DataFrame,
    operation: str | None,
    in_meas_1_name: str,
    in_meas_2_name: str | None,
    op_dict: dict,
):
    """Perform operation on the two measurement columns.

    Calcualates new column that will be used to detect collective events.
    Operation is determined in the columnpicker dialog and loaded from
    the OPERATOR_DICTIONARY.


    Parameters
    ----------
    data : pd.DataFrame
        Input data
    operation : str
        Operation to perform on the two measurement columns
    in_meas_1_name : str
        Name of the first measurement column
    in_meas_2_name : str | None
        Name of the second measurement column
    op_dict : dict
        Dictionary containing the operations to perform on the two measurement columns
    """
    data_in = data.copy()
    if not operation:
        return in_meas_1_name, data_in
    if operation in op_dict.keys() and in_meas_2_name is not None:
        out_meas_name = op_dict[operation][1]
        data_in[out_meas_name] = op_dict[operation][0](
            data[in_meas_1_name], data[in_meas_2_name]
        )

    else:
        out_meas_name = in_meas_1_name
    return out_meas_name, data_in


def get_tracklengths(
    df: pd.DataFrame, field_of_view_id: str, track_id: str, second_filter_id: str
) -> tuple:
    """
    Groups filtered data by track_id and
    returns minimum and maximum tracklenght.
    Updates min and max tracklenght in
    the widget spinbox and sliders.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    field_of_view_id : str
        Name of the field of view column
    track_id : str

    Returns
    -------
    min_tracklength : int
        Minimum tracklenght
    max_tracklength : int
        Maximum tracklenght
    """
    if df.empty:
        return 0, 0

    if field_of_view_id and second_filter_id:
        track_lenths = df.groupby([field_of_view_id, second_filter_id, track_id])[
            track_id
        ].agg("count")
    elif field_of_view_id:
        track_lenths = df.groupby([field_of_view_id, track_id])[track_id].agg("count")
    elif second_filter_id:
        track_lenths = df.groupby([second_filter_id, track_id])[track_id].agg("count")
    else:
        track_lenths = df.groupby([track_id])[track_id].agg("count")
    minmax = (track_lenths.min(), track_lenths.max())
    return minmax


def filter_data(
    df_in: pd.DataFrame,
    field_of_view_id_name: str,
    frame_name: str,
    track_id_name: str | None,
    measurement_name: str,
    additional_filter_column_name: str,
    position_columns: list,
    fov_val: Union[int, float, str],
    additional_filter_value: Union[int, float, str],
    min_tracklength_value: int,
    max_tracklength_value: int,
    frame_interval: float,
    st_out: Callable,
):
    """Used to filter the input data to contain a single position.
    If selected in the columpicker dialog, an additional filter option
    is displayed.
    Filter options also include minimum and maximum tracklength.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input data
    pos_val : Union[int, float, str]
        Position value to filter for
    additional_filter_value : Union[int, float, str]
        Additional filter value
    min_tracklength_value : int
        Minimum tracklenght
    max_tracklength_value : int
        Maximum tracklenght
    field_of_view_id_name : str
        Name of the field of view column
    frame_name : str
        Name of the frame column
    track_id_name : str | None
        Name of the track id column
    measurement_name : str
        Name of the measurement column
    additional_filter_column_name : str
        Name of the additional filter column
    st_out : Callable
        Output functoin i.e. print() or show_info from Napari
    position_columns : list
        List containing all position names
    frame_interval : float
        Frame interval
    """
    # gets raw data read in by arcos_widget from stored_variables object
    # and columns from columnpicker value
    in_data = process_input(
        df=df_in,
        field_of_view_column=field_of_view_id_name,
        frame_column=frame_name,
        pos_columns=position_columns,
        track_id_column=track_id_name,
        measurement_column=measurement_name,
    )

    if df_in.empty or field_of_view_id_name == measurement_name:
        st_out("No data loaded, or not loaded correctly")
        return pd.DataFrame([]), 1, 1

    # if the position column was not chosen in columnpicker,
    # dont filter by position
    if field_of_view_id_name:
        # filter by position
        if len(df_in[field_of_view_id_name].unique()) > 1:
            # hast to be done before .filter_tracklenght otherwise code could break
            # if track ids are not unique to positions
            in_data.filter_position(fov_val)

    if additional_filter_column_name:
        in_data.filter_second_column(
            additional_filter_column_name,
            additional_filter_value,
        )

    # filter by tracklenght
    if track_id_name:
        in_data.filter_tracklength(
            min_tracklength_value,
            max_tracklength_value,
        )
    # option to set frame interval
    in_data.frame_interval(frame_interval)

    filtered_data = in_data.return_pd_df()
    filtered_data.reset_index(drop=True, inplace=True)
    try:
        filtered_data = subtract_timeoffset(filtered_data, frame_name)
    except (KeyError, TypeError) as err:
        raise type(err)(
            f"Frame column not set properly, has to be int or float. Set column is '{frame_name}' and available columns are {filtered_data.columns.tolist()}"  # noqa: E501
        ) from err

    # get min and max values
    if filtered_data.empty:
        st_out("No data loaded, or not loaded correctly")
        return pd.DataFrame([]), None, None
    max_meas = max(filtered_data[measurement_name])
    min_meas = min(filtered_data[measurement_name])
    st_out("Data Filtered!")
    return filtered_data, max_meas, min_meas


def check_for_collid_column(data: pd.DataFrame, clid_column="collid", suffix="old"):
    """If clid_column is present in input data,
    add suffix to prevent dataframe merge conflic."""
    if clid_column in data.columns:
        data.rename(columns={clid_column: f"{clid_column}_{suffix}"}, inplace=True)
    return data


def preprocess_data(
    df: pd.DataFrame, columnames: columnnames, op_dict: dict | None = None
):
    """Preprocesses data for calculation of measurement columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columnames : columnnames
        Object containing selected column names
    op_dict : dict | None, optional
        Dictionary containing the operation to perform. If None,
        the OPERATOR_DICTIONARY is used, by default None.

    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    if op_dict is None:
        op_dict = OPERATOR_DICTIONARY
    op = columnames.measurement_math_operation
    meas_1 = columnames.measurement_column_1
    meas_2 = columnames.measurement_column_2
    try:
        return calculate_measurement(df, op, meas_1, meas_2, op_dict)
    except (KeyError, TypeError, ValueError) as err:
        raise type(err)(
            f"Measurement columns not set properly, has to be int or float. Trying to {str(op).lower()} columns '{meas_1}' and '{str(meas_2)}' and available columns are {df.columns.tolist()}"  # noqa: E501
        ) from err


def match_dataframes(
    df1,
    df2,
    threshold_percentage=0.5,
    frame_column="frame",
    coord_cols1=["centroid-0", "centroid-1"],
    coord_cols2=None,
    std_out: Callable = print,
):
    """
    Match the dataframes df1 and df2 based on centroid coordinates and frame.
    Tries a direct merge first, then falls back to nearest neighbor approach.

    Parameters:
    - df1, df2: Input dataframes with coordinate and frame columns.
    - threshold_percentage: Percentage of the data range to be used as the threshold.
    - frame_column: Column name for frame identifier.
    - coord_cols1: List of column names for coordinates in df1.
    - coord_cols2: List of column names for coordinates in df2.
    If None, it will be assumed to be the same as coord_cols1.

    Returns:
    - Combined dataframe of matched points.
    """
    if coord_cols2 is None:
        coord_cols2 = coord_cols1

    # Ensure both dataframes have the same number of coordinate columns
    assert len(coord_cols1) == len(
        coord_cols2
    ), "Number of coordinate columns must be the same for both dataframes"

    # Try a direct merge first
    merge_cols = [frame_column] + coord_cols1
    try:
        merged_df = pd.merge(
            df1,
            df2,
            left_on=merge_cols,
            right_on=[frame_column] + coord_cols2,
            how="inner",
        )
        # If the merged dataframe has the same length as df1, return it
        if len(merged_df) == len(df1):
            return merged_df
    except Exception:
        pass

    std_out(
        "Direct merge of tracking data failed, falling back to nearest neighbor approach"
    )
    # If there's an error during merge, we'll proceed with nearest neighbor

    # If direct merge is unsuccessful or results in a dataframe of different length,
    # proceed with nearest neighbor approach
    # Calculate threshold based on the provided percentage
    coord_ranges = [df1[col].max() - df1[col].min() for col in coord_cols1]
    threshold = (threshold_percentage / 100.0) * max(coord_ranges)

    results = []

    # Iterate through unique frames
    for frame in df1[frame_column].unique():
        # Filter dataframes for the current frame
        df1_frame = df1[df1[frame_column] == frame].copy()
        df2_frame = df2[df2[frame_column] == frame].copy()

        # If either dataframe segment is empty, skip to the next frame
        if df1_frame.empty or df2_frame.empty:
            continue

        # Fit the NearestNeighbors model for df2's coordinates
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(df2_frame[coord_cols2])

        # Query the nearest neighbor for each point in df1
        distances, indices = neigh.kneighbors(df1_frame[coord_cols1])

        # Filter out matches that exceed the threshold
        mask = distances.ravel() < threshold
        matched_df1 = df1_frame[mask]
        matched_df2 = df2_frame.iloc[indices[mask].ravel()]

        # Combine the matched dataframes and append to results
        combined = pd.concat(
            [
                matched_df1.reset_index(drop=True),
                matched_df2.track_id.reset_index(drop=True),
            ],
            axis=1,
        )
        results.append(combined)
    merged_df = pd.concat(results)

    if len(merged_df) < 0.9 * len(df1):
        raise ValueError(
            "Failed to match a significant number of points from tracking data"
        )
    return merged_df


class DataFrameMatcherSignals(WorkerBaseSignals):
    finished = Signal()
    aborted = Signal(object)
    matched = Signal(pd.DataFrame)


class DataFrameMatcher(WorkerBase):
    """Match dataframes in a separate thread.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Dataframes to match
    threshold_percentage : float, optional
        Percentage of the data range to be used as the threshold
    frame_column : str, optional
        Column name for frame identifier
    coord_cols1 : list of str, optional
        List of column names for coordinates in df1
    coord_cols2 : list of str, optional
        List of column names for coordinates in df2. If None, it will be assumed to be the same as coord_cols1
    parent : QObject, optional
        Parent object

    Signals
    -------
    finished : Signal
        Emitted when the task is finished
    matched : Signal
        Emitted when matching is finished
    aborted : Signal
        Emitted when the task is aborted either by the user or an error
    """

    def __init__(
        self,
        df1,
        df2,
        threshold_percentage=0.5,
        frame_column="frame",
        coord_cols1=["centroid-0", "centroid-1"],
        coord_cols2=None,
    ):
        super().__init__(SignalsClass=DataFrameMatcherSignals)
        self.df1 = df1
        self.df2 = df2
        self.threshold_percentage = threshold_percentage
        self.frame_column = frame_column
        self.coord_cols1 = coord_cols1
        self.coord_cols2 = coord_cols2

    def run(self):
        """Task to match dataframes."""
        try:
            matched_df = match_dataframes(
                self.df1,
                self.df2,
                self.threshold_percentage,
                self.frame_column,
                self.coord_cols1,
                self.coord_cols2,
            )
            self.matched.emit(matched_df)
        except Exception as e:
            self.aborted.emit(e)
        finally:
            self.finished.emit()


def get_delimiter(file_path: str, bytes_to_load=4096):
    """Returns the delimiter used in a csv file.

    Parameters
    ----------
    file_path : str
        Path to the csv file
    bytes : int, optional
        Number of bytes to read, by default 4096

    Returns
    -------
    delimiter : str
        Delimiter used in the csv file
    """
    sniffer = csv.Sniffer()
    if file_path.endswith(".csv"):
        with open(file_path) as _file:
            data = _file.read(bytes_to_load)
            delimiter = sniffer.sniff(data).delimiter
    if file_path.endswith("csv.gz"):
        with gzip.open(file_path, mode="rt") as _file:
            data = _file.read(bytes_to_load)
            delimiter = sniffer.sniff(data).delimiter
    return delimiter


def read_data_header(filename: str):
    """Reads the header of a csv file and returns it as a list.

    Parameters
    ----------
    filename : str
        Path to the csv file

    Returns
    -------
    headers : list
        List containing the header of the csv file
    delimiter_value : str
        Delimiter used in the csv file
    """
    delimiter_value = get_delimiter(filename)
    if filename.endswith(".csv"):
        with open(filename) as f:
            d_reader = csv.DictReader(f, delimiter=delimiter_value)
            # get fieldnames from DictReader object and store in list
            headers = d_reader.fieldnames
    if filename.endswith("csv.gz"):
        with gzip.open(filename, mode="rt") as f:
            d_reader = csv.DictReader(f, delimiter=delimiter_value)
            # get fieldnames from DictReader object and store in list
            headers = d_reader.fieldnames
    return headers, delimiter_value


class process_input:
    """Class to process input data."""

    def __init__(
        self,
        df: pd.DataFrame,
        field_of_view_column: str,
        frame_column: str,
        pos_columns: list,
        track_id_column: str | None,
        measurement_column: str,
    ):
        """Process input data. Optionally filter by position and track length.

        Parameters
        ----------
        df : pd.DataFrame
            input dataframe
        field_of_view_column : str
            name of column containing field of view
        frame_column : str
            name of column containing frame number
        pos_columns : list
            list of column names containing position

        track_id_column : str | None
            name of column containing track id
        measurement_column : str
            name of column containing measurement
        """
        self.field_of_view_column = field_of_view_column
        self.frame_column = frame_column
        self.pos_columns = pos_columns
        self.measurment_column = measurement_column
        self.track_id_column = track_id_column
        self.df = df

    def filter_position(self, fov_to_select=None, return_dataframe=False):
        """Filters dataframe by position passed in as argument.

        Parameters
        ----------
        fov_to_select : str, int, float, optional
            position to filter by, by default None
        return_dataframe : bool, optional
            if True, returns filtered dataframe, by default False
        """
        if fov_to_select is not None:
            self.df = self.df.loc[
                self.df.loc[:, self.field_of_view_column] == fov_to_select
            ].copy(deep=True)
        if return_dataframe:
            return self.df

    def filter_second_column(
        self, column, value_to_select=None, return_dataframe=False
    ):
        """Filters dataframe by column and value passed in as argument.

        Parameters
        ----------
        column : str
            column to filter by
        value_to_select : str, optional
            value to filter by, by default None
        return_dataframe : bool, optional
            if True, returns filtered dataframe, by default False
        """
        if value_to_select is not None:
            self.df = self.df.loc[self.df.loc[:, column] == value_to_select].copy(
                deep=True
            )
        if return_dataframe:
            return self.df

        # filter_tracklenght works only if previously filtered by pos
        # or if track_ids dont overlapp between fov

    def filter_tracklength(self, min_val, max_val, return_dataframe=False):
        """Filters tracklength by min and max value passed in as argument.

        Parameters
        ----------
        min_val : int
            minimum tracklength
        max_val : int
            maximum tracklength
        return_dataframe : bool, optional
            if True, returns filtered dataframe, by default False
        """
        if self.track_id_column is None:
            raise ValueError(
                "Track id column not set, can not filter by track length. Set track id column in constructor."  # noqa: E501
            )
        track_length = self.df.groupby(self.track_id_column).size()
        track_length_filtered = track_length.between(min_val, max_val)
        track_length_filtered_names = track_length_filtered[track_length_filtered].index
        self.df = self.df.loc[
            self.df.loc[:, self.track_id_column].isin(track_length_filtered_names)
        ].copy(deep=True)
        if return_dataframe:
            return self.df

    def rescale_measurment(self, rescale_factor: int, return_dataframe=False):
        """
        rescales measurment column by factor passed in as argument
        """
        self.df[self.measurment_column] = (
            rescale_factor * self.df[self.measurment_column]
        )
        if return_dataframe:
            return self.df

    def frame_interval(self, factor):
        """Rescales the frame column by a factor."""
        if factor > 1:
            self.df[self.frame_column] = self.df[self.frame_column] // factor
            self.df[self.frame_column] = self.df[self.frame_column].astype(int)

    def return_pd_df(self) -> pd.DataFrame:
        """Returns the dataframe."""
        return self.df


def layer_to_df(layer: napari.layers.Layer) -> pd.DataFrame:
    """Returns a dataframe from a napari layer properties object.

    Parameters
    ----------
    layer : napari.layers.Layer
        Napari layer

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(layer.properties)
    return df


def dataframe_from_layers(layers: list[napari.layers.Layer]):
    """Returns a dataframe from a list of napari layers.

    Parameters
    ----------
    layers : list[napari.layers.Layer]
        List of napari layers

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: fov, frame, x, y, track_id, measurement
    """
    dfs = [layer_to_df(layer) for layer in layers]
    df = pd.concat(dfs, ignore_index=True)
    return df


class DataLoaderSignals(WorkerBaseSignals):
    finished = Signal()
    loading_finished = Signal()
    new_data = Signal(pd.DataFrame)
    aborted = Signal(object)


class DataLoader(WorkerBase):
    """Load data in a separate thread.

    Parameters
    ----------
    filepath : str
        Path to the csv file
    delimiter : str
        Delimiter used in the csv file
    wait_for_columnpicker : bool, optional
        If True, waits for columnpicker to be finished, by default False
    parent : QObject, optional
        Parent object, by default None

    Signals
    -------
    finished : Signal
        Emitted when the task is finished
    loading_finished : Signal
        Emitted when loading is finished
    new_data : Signal
        Emitted when new data is available
    aborted : Signal
        Emitted when the task is aborted either by the user or an error.
    """

    # finished = Signal()
    # loading_finished = Signal()
    # new_data = Signal(pd.DataFrame)
    # aborted = Signal(object)

    def __init__(
        self,
        filepath: str,
        delimiter: str,
        wait_for_columnpicker=False,
    ):
        super().__init__(SignalsClass=DataLoaderSignals)
        self.filepath = filepath
        self.delimiter = delimiter
        self.wait_for_columnpicker = wait_for_columnpicker
        self.abort_loading = False

    def run(self):
        """Task to load data."""
        try:
            df = self._load_data(self.filepath, self.delimiter)
            self._emit_when_ready(df)
            self.aborted.emit(0)
        except Exception as e:
            # print(e)
            self.aborted.emit(e)
        finally:
            self.finished.emit()

    def _load_data(self, filepath, delimiter=None):
        """Loads data from a csv file and stores it in the data storage."""
        df = pd.read_csv(filepath, delimiter=delimiter, engine="pyarrow")
        self.loading_finished.emit()

        return df

    def _emit_when_ready(self, df):
        """Preprocesses data and stores it in the data storage."""
        while self.wait_for_columnpicker:
            time.sleep(0.1)

        if self.abort_loading:
            self.aborted.emit(1)
            return

        self.new_data.emit(df)
