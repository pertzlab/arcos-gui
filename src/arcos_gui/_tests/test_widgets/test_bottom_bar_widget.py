import pandas as pd
import pytest
from arcos_gui.processing import DataStorage
from arcos_gui.widgets import BottombarController
from pytestqt.qtbot import QtBot
from qtpy import QtCore, QtGui


@pytest.fixture
def data_storage_instance():
    return DataStorage()


@pytest.fixture
def bb_controller(data_storage_instance: DataStorage, qtbot: QtBot):
    bb_controller = BottombarController(data_storage_instance)
    qtbot.addWidget(bb_controller.widget)
    return bb_controller


def test_bottom_bar_widget_initialization(bb_controller):
    assert isinstance(bb_controller, BottombarController)


def test_update_event_counter(
    bb_controller: BottombarController, data_storage_instance: DataStorage
):
    # Test with empty dataframe
    df = pd.DataFrame()
    data_storage_instance.arcos_stats.value = df
    bb_controller.update_event_counter()
    assert bb_controller.widget.collev_number_display.intValue() == 0

    # Test with non-empty dataframe
    df = pd.DataFrame({"collid": [1, 2, 2, 3, 3, 3]})
    data_storage_instance.arcos_stats.value = df
    bb_controller.update_event_counter()
    assert bb_controller.widget.collev_number_display.intValue() == 3


def test_update_help_pressed(bb_controller, mocker):
    mocker.patch("PyQt5.QtCore.QUrl")
    mocker.patch("PyQt5.QtGui.QDesktopServices.openUrl")
    bb_controller.widget.arcos_help_button.click()
    QtGui.QDesktopServices.openUrl.assert_called_once_with(
        QtCore.QUrl("https://pertzlab.github.io/arcos-gui/Usage/")
    )
