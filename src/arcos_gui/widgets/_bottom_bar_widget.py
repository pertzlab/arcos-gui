"""Module containing the bottom bar widget class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtWidgets
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices

if TYPE_CHECKING:
    from arcos_gui.processing import DataStorage


class _bottombar_widget(QtWidgets.QWidget):
    """Bottom bar widget. Displays the number of detected collective events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupui()

    def setupui(self):
        """Setup the UI. Add widgets to the layout."""
        self.bottom_bar_layout = QtWidgets.QHBoxLayout()
        self.collev_number_display_label = QtWidgets.QLabel(
            "Number of detected collective events: "
        )
        self.collev_number_display = QtWidgets.QLCDNumber()
        self.collev_number_display.setDigitCount(4)
        self.collev_number_display.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.arcos_help_button = QtWidgets.QPushButton("Open Documentation")
        self.bottom_bar_layout.addWidget(self.collev_number_display_label)
        self.bottom_bar_layout.addWidget(self.collev_number_display)
        self.bottom_bar_layout.addWidget(self.arcos_help_button)
        self.setLayout(self.bottom_bar_layout)
        self._connect_signals()

    def _update_help_pressed(self):
        url = QUrl("https://pertzlab.github.io/arcos-gui/Usage/")
        QDesktopServices.openUrl(url)

    def _connect_signals(self):
        self.arcos_help_button.clicked.connect(self._update_help_pressed)


class BottombarController:
    """Controller for the bottom bar widget."""

    def __init__(self, data_storage_instance: DataStorage, parent=None):
        self.widget = _bottombar_widget(parent)
        self.data_storage_instance = data_storage_instance
        self._connect_signals()

    def _connect_signals(self):
        self.data_storage_instance.arcos_stats.value_changed.connect(
            self.update_event_counter
        )

    def update_event_counter(self):
        """Update the number of detected collective events."""
        df = self.data_storage_instance.arcos_stats.value

        if df.empty:
            self.widget.collev_number_display.display(0)
        else:
            collev_number = df["collid"].nunique()
            self.widget.collev_number_display.display(collev_number)


if __name__ == "__main__":
    import sys

    from arcos_gui.processing import DataStorage  # noqa: F811

    app = QtWidgets.QApplication(sys.argv)
    ds = DataStorage()
    controller = BottombarController(ds)
    controller.widget.show()
    sys.exit(app.exec_())
