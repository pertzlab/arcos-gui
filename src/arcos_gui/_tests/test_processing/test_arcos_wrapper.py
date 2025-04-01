from __future__ import annotations

import pandas as pd
import pytest
from arcos4py import ARCOS
from arcos_gui.processing import BatchProcessor, DataStorage
from arcos_gui.processing._arcos_wrapper import (
    arcos_worker,
    binarization,
    calculate_arcos_stats,
    detect_events,
    filtering_arcos_events,
    get_eps,
    init_arcos_object,
)

# some tests that just validate that arcos4py in fact creates arcos objects and outputs data
# since this package relies on arcos4py for its main data processing the tests there are relevant and
# should be sufficient and are not repeated here
# maybe at a later point specific tests for the arcos functionallity should be added


def test_init_arcos_object():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    assert isinstance(arcos_object, ARCOS)


def test_binarization():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    assert isinstance(arcos_object, ARCOS)
    assert not arcos_object.data.empty
    for i in arcos_object.data["m.bin"].unique():
        assert i in [0, 1]
    assert arcos_object.data["m.resc"].min() == 0
    assert arcos_object.data["m.resc"].max() == 1


def test_detect_events():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    arcos_events = detect_events(
        arcos=arcos_object,
        neighbourhood_size=20,
        eps_prev=20,
        min_clustersize=4,
        nPrev_value=1,
    )
    assert isinstance(arcos_events, pd.DataFrame)
    assert not arcos_events.empty


def test_filtering_arcos_events_high_values():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    arcos_events = detect_events(
        arcos=arcos_object,
        neighbourhood_size=20,
        eps_prev=20,
        min_clustersize=4,
        nPrev_value=1,
    )
    arcos_filtered = filtering_arcos_events(
        detected_events_df=arcos_events,
        frame_col_name="t",
        collid_name="collid",
        track_id_col_name="id",
        min_dur=1,
        total_event_size=1,
    )
    assert isinstance(arcos_filtered, pd.DataFrame)
    assert not arcos_filtered.empty
    pd.testing.assert_frame_equal(arcos_events, arcos_filtered, check_dtype=False)


def test_filtering_arcos_events_low_values():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    arcos_events = detect_events(
        arcos=arcos_object,
        neighbourhood_size=20,
        eps_prev=20,
        min_clustersize=4,
        nPrev_value=1,
    )
    arcos_filtered = filtering_arcos_events(
        detected_events_df=arcos_events,
        frame_col_name="t",
        collid_name="collid",
        track_id_col_name="id",
        min_dur=20,
        total_event_size=1,
    )
    assert isinstance(arcos_filtered, pd.DataFrame)
    assert arcos_filtered.empty


def test_calculate_arcos_stats():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    arcos_events = detect_events(
        arcos=arcos_object,
        neighbourhood_size=20,
        eps_prev=20,
        min_clustersize=4,
        nPrev_value=1,
    )
    arcos_filtered = filtering_arcos_events(
        detected_events_df=arcos_events,
        frame_col_name="t",
        collid_name="collid",
        track_id_col_name="id",
        min_dur=1,
        total_event_size=1,
    )
    arcos_stats = calculate_arcos_stats(
        df_arcos_filtered=arcos_filtered,
        frame_column="t",
        collid_name="collid",
        object_id_name="id",
        position_columns=["x", "y"],
    )
    assert isinstance(arcos_stats, pd.DataFrame)
    assert not arcos_stats.empty
    assert arcos_stats["collid"].nunique() == 1


def test_get_eps():
    df = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    arcos_object = init_arcos_object(df, ["x", "y"], "m", "t", "id")
    arcos_object = binarization(
        arcos_object, True, False, 0.01, 0.99, 1, 3, 0.5, 0.5, 1, "none"
    )
    eps = get_eps(arcos_object, "mean", 3, 20)
    assert isinstance(eps, float)
    eps = get_eps(arcos_object, "kneepoint", 3, 20)
    assert isinstance(eps, float)
    eps = get_eps(arcos_object, "manual", 3, 20)
    assert eps == 20
    with pytest.raises(ValueError):
        eps = get_eps(arcos_object, "wrong", 3, 20)


def test_arcos_wrapper_run_all():
    what_to_run = {"binarization", "filtering", "tracking"}
    ds = DataStorage()
    filtered_data = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")
    ds.set_callbacks_verbose(True)
    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = "None"
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"

    worker = arcos_worker(what_to_run, print)
    worker.filtered_data = filtered_data
    worker.columns = ds.columns.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = True
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 0.5
    worker.arcos_parameters.bin_peak_threshold.value = 0.5
    worker.arcos_parameters.neighbourhood_size.value = 20
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 20
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 1
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert not worker.arcos_raw_output.empty


def test_arcos_wrapper_run_no_data(capsys):
    what_to_run = {"binarization", "filtering", "tracking"}
    ds = DataStorage()
    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = "None"
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = "None"
    ds.columns.value.additional_filter_column = "None"
    ds.columns.value.measurement_math_operation = "None"
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    worker = arcos_worker(what_to_run, print)
    worker.columns = ds.columns.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = True
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 10
    worker.arcos_parameters.bin_peak_threshold.value = 10
    worker.arcos_parameters.neighbourhood_size.value = 20
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 20
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 1
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    captured_data = capsys.readouterr()
    assert worker.filtered_data.empty
    assert worker.arcos_object.data.empty
    assert worker.arcos_raw_output.empty
    assert "No data loaded. Load first using the import data tab." in captured_data.out


def test_arcos_wrapper_run_no_bin_data(capsys):
    what_to_run = {"binarization", "filtering", "tracking"}
    ds = DataStorage()
    ds.filtered_data.value = pd.read_csv(
        "src/arcos_gui/_tests/test_data/arcos_data.csv"
    )

    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = "None"
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = "None"
    ds.columns.value.additional_filter_column = "None"
    ds.columns.value.measurement_math_operation = "None"
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    worker = arcos_worker(what_to_run, print)
    worker.columns = ds.columns.value
    worker.filtered_data = ds.filtered_data.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = True
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 10
    worker.arcos_parameters.bin_peak_threshold.value = 10
    worker.arcos_parameters.neighbourhood_size.value = 20
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 20
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 1
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    captured_data = capsys.readouterr()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert worker.arcos_raw_output.empty
    assert "No Binarized Data. Adjust Binazation Parameters." in captured_data.out


def test_arcos_wrapper_run_no_detected_events_data(capsys):
    what_to_run = {"binarization", "filtering", "tracking"}
    ds = DataStorage()
    ds.filtered_data.value = pd.read_csv(
        "src/arcos_gui/_tests/test_data/arcos_data.csv"
    )

    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = None
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    worker = arcos_worker(what_to_run, print)
    worker.columns = ds.columns.value
    worker.filtered_data = ds.filtered_data.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = False
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 0.5
    worker.arcos_parameters.bin_peak_threshold.value = 0.5
    worker.arcos_parameters.neighbourhood_size.value = 0.01
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 1
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 50
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    captured_data = capsys.readouterr()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert worker.arcos_raw_output.empty
    assert (
        "No Collective Events detected. Adjust Event Detection Parameters."
        in captured_data.out
    )


def test_arcos_wrapper_run_no_filtered_data(capsys):
    what_to_run = {"binarization", "filtering", "tracking"}
    ds = DataStorage()
    ds.filtered_data.value = pd.read_csv(
        "src/arcos_gui/_tests/test_data/arcos_data.csv"
    )

    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = None
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    worker = arcos_worker(what_to_run, print)
    worker.columns = ds.columns.value
    worker.filtered_data = ds.filtered_data.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = False
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 0.5
    worker.arcos_parameters.bin_peak_threshold.value = 0.5
    worker.arcos_parameters.neighbourhood_size.value = 20
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 20
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 50
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    captured_data = capsys.readouterr()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert not worker.arcos_raw_output.empty
    assert (
        "No Collective Events detected.Adjust Filtering parameters."
        in captured_data.out
    )


def test_arcos_wrapper_run_specific_parts():
    class get_data_from_filtering:
        filtered_data = pd.DataFrame()
        stats = pd.DataFrame()

        @classmethod
        def get_data_from_callback(cls, data):
            cls.filtered_data = data[0]
            cls.stats = data[1]

    what_to_run = {"binarization"}
    ds = DataStorage()
    ds.filtered_data.value = pd.read_csv(
        "src/arcos_gui/_tests/test_data/arcos_data.csv"
    )

    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = None
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    worker = arcos_worker(what_to_run, print)

    worker.new_arcos_output.connect(get_data_from_filtering.get_data_from_callback)

    worker.columns = ds.columns.value
    worker.filtered_data = ds.filtered_data.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = False
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 0.5
    worker.arcos_parameters.bin_peak_threshold.value = 0.5
    worker.arcos_parameters.neighbourhood_size.value = 20
    worker.arcos_parameters.eps_method.value = "manual"
    worker.arcos_parameters.eps_prev.value = 20
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 1
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert worker.arcos_raw_output.empty

    what_to_run.clear()
    what_to_run.add("tracking")
    worker.run()

    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert not worker.arcos_raw_output.empty
    assert get_data_from_filtering.filtered_data.empty
    assert get_data_from_filtering.stats.empty

    what_to_run.clear()
    what_to_run.add("filtering")
    worker.run()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert not worker.arcos_raw_output.empty
    assert not get_data_from_filtering.filtered_data.empty
    assert not get_data_from_filtering.stats.empty


def test_arcos_wrapper_epsMethod():
    class get_data_from_eps:
        eps: float

        @classmethod
        def get_data_from_callback(cls, data):
            cls.eps = data

    what_to_run = {"binarization", "tracking", "filtering"}
    ds = DataStorage()
    ds.filtered_data.value = pd.read_csv(
        "src/arcos_gui/_tests/test_data/arcos_data.csv"
    )
    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = None
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"

    worker = arcos_worker(what_to_run, print)
    worker.new_eps.connect(get_data_from_eps.get_data_from_callback)

    worker.columns = ds.columns.value
    worker.filtered_data = ds.filtered_data.value
    worker.arcos_parameters.interpolate_meas.value = True
    worker.arcos_parameters.clip_measurements.value = False
    worker.arcos_parameters.clip_low.value = 0.01
    worker.arcos_parameters.clip_high.value = 0.99
    worker.arcos_parameters.bias_method.value = "none"
    worker.arcos_parameters.smooth_k.value = 1
    worker.arcos_parameters.bias_k.value = 3
    worker.arcos_parameters.polynomial_degree.value = 1
    worker.arcos_parameters.bin_threshold.value = 0.5
    worker.arcos_parameters.bin_peak_threshold.value = 0.5
    worker.arcos_parameters.neighbourhood_size.value = 0
    worker.arcos_parameters.eps_method.value = "mean"
    worker.arcos_parameters.eps_prev.value = 0
    worker.arcos_parameters.min_clustersize.value = 4
    worker.arcos_parameters.nprev.value = 1
    worker.arcos_parameters.min_dur.value = 1
    worker.arcos_parameters.total_event_size.value = 1
    worker.run()
    assert not worker.filtered_data.empty
    assert not worker.arcos_object.data.empty
    assert not worker.arcos_raw_output.empty

    assert get_data_from_eps.eps != 0


def test_init_batch():
    # Test the initialization of the BatchProcessor class

    # Replace these with the correct initialization based on your class definitions
    data_storage = DataStorage()
    bp = BatchProcessor(
        input_path="path/to/directory",
        data_storage_instance=data_storage,
        what_to_export=["arcos_output", "arcos_stats"],
    )

    assert bp.input_path == "path/to/directory"
    assert bp.arcos_parameters == data_storage.arcos_parameters
    assert bp.columnames == data_storage.columns


def test_create_fileendings_list():
    # Test the _create_fileendings_list method
    data_storage = DataStorage()

    bp = BatchProcessor(
        input_path="path/to/directory",
        data_storage_instance=data_storage,
        what_to_export=["arcos_output", "statsplot"],
    )

    fileendings = bp._create_fileendings_list()
    assert fileendings == [
        ".csv",
        ".svg",
    ]  # replace with the expected list of file endings


def test_run_arcos_batch():
    ds = DataStorage()
    ds.columns.value.frame_column = "t"
    ds.columns.value.object_id = "id"
    ds.columns.value.x_column = "x"
    ds.columns.value.y_column = "y"
    ds.columns.value.z_column = None
    ds.columns.value.measurement_column = "m"
    ds.columns.value.position_id = None
    ds.columns.value.additional_filter_column = None
    ds.columns.value.measurement_math_operation = None
    ds.columns.value.measurement_bin = "m.bin"
    ds.columns.value.measurement_resc = "m.resc"
    ds.min_max_tracklenght.value = [1, 100]

    bp = BatchProcessor(
        input_path="src/arcos_gui/_tests/test_data/arcos_data.csv",
        data_storage_instance=ds,
        what_to_export=["arcos_output", "arcos_stats"],
    )

    df_in = pd.read_csv("src/arcos_gui/_tests/test_data/arcos_data.csv")

    # Call the method with the test data
    arcos_df_filtered, arcos_stats = bp.run_arcos_batch(df_in)
    assert arcos_df_filtered is not None
    assert arcos_stats is not None
