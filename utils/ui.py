import logging
import math
import re
from colorsys import hsv_to_rgb
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import PySide2.QtWidgets as Qt
from PySide2.QtCore import QProcess, Qt as QtFlags, Signal
from PySide2.QtGui import QColor, QFont, QPalette
from numpy import ndarray
from vedo import LegendBox, Mesh, Plotter, Points
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .common import to_full_list


class DarkPalette(QPalette):
    r"""
    Qt Dark Palette for QApplication
    """

    def __init__(self):
        super(DarkPalette, self).__init__()
        self.setColor(QPalette.Window, QColor(53, 53, 53))
        self.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setColor(QPalette.Base, QColor(40, 40, 40))
        self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        self.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        self.setColor(QPalette.Text, QColor(255, 255, 255))
        self.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        self.setColor(QPalette.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setColor(QPalette.BrightText, QColor(255, 0, 0))
        self.setColor(QPalette.Link, QColor(42, 130, 218))
        self.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.HighlightedText, QColor(0, 0, 0))


class CoinPlot(Qt.QWidget):
    r"""
    Plot widget designed to preview coins and relative points, used in
    RESP applications.
    """

    def __init__(
        self,
        parent: Qt.QWidget = None,
        f: QtFlags.WindowFlags = QtFlags.Widget
    ):
        """
        Plot widget designed to preview coins and relative points, used
        in RESP applications.

        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        :param f: Qt window flags. Defaults to Widget.
        :type f: QtFlags.WindowFlags, optional
        """
        super(CoinPlot, self).__init__(parent, f)
        layout = Qt.QVBoxLayout()
        layout.setMargin(0)
        plot_widget = QVTKRenderWindowInteractor(self)
        self.plot = Plotter(bg="#555555", axes=8, qt_widget=plot_widget)
        self.plot.show()
        self.mesh = None
        self.points = None
        self.rg = []
        self.kp = []
        self.bh = []
        self._rg_actors = []
        self._kp_actors = []
        self._bh_actors = []
        self._labels = None
        layout.addWidget(plot_widget)
        self._mode_selector = Qt.QComboBox()
        self._mode_selector.addItems(["Regions", "Landmarks", "Beard/Hair", "Coin only"])
        self._mode_selector.currentIndexChanged.connect(lambda: self.update_plot())
        layout.addWidget(self._mode_selector)
        self.setLayout(layout)

    def __getattr__(self, item) -> Any:
        return getattr(self.plot, item)

    def update_plot(
        self,
        mesh: Mesh = None,
        points: ndarray = None,
        rg: list[tuple[list[int], str]] = None,
        kp: list[int] = None,
        bh: list[list[int]] = None,
        *,
        clear: bool = False
    ):
        r"""
        Modify the plot by adding/removing elements. Leaving a parameter
        set to None will not change its previous value.

        :param mesh: The coin mesh.
        :type mesh: Mesh, optional
        :param points: The list of the points. Its shape must be (N, 3).
        :type points: ndarray, optional
        :param rg: A list containing the classifications of the points.
            Each element is a region and consists in a tuple containing
            an indices list of the points which belong to that region
            and its name.
        :type rg: list[tuple[list[int], str]], optional
        :param kp: The list of the landmarks indices.
        :type kp: list[int], optional
        :param bh: A list containing the beard/hair regions. Each
            element is an area consists in an indices list of the points
            which enclose that area.
        :type bh: list[list[int]], optional
        :param clear: If True, the plot is completed cleared from any
            previous value. Defaults to False
        :type clear: bool, optional

        Note: no check is performed on the indices.
        """
        reset_camera = False

        # Remove any previous actor
        if clear:
            self.mesh = None
            self.points = None
            self.rg = []
            self.kp = []
            self.bh = []
            self._rg_actors = []
            self._kp_actors = []
            self._bh_actors = []
            self._labels = None
            reset_camera = True

        # Mesh
        if mesh is not None:
            self.mesh = mesh
            reset_camera = True

        # Points
        if pts_modified := (points is not None):
            self.points = points
            self._labels = Points(self.points, r=5).labels2d("id")
            reset_camera = True

        if rg_modified := (rg is not None):
            self.rg = rg
        if kp_modified := (kp is not None):
            self.kp = kp
        if bh_modified := (bh is not None):
            # Concat sub-regions
            self.bh = [i for r in bh for i in r]

        # Regions
        if rg_modified or pts_modified:
            n = len(self.rg)
            not_assigned = set(range(len(self.points)))
            self._rg_actors = []
            for i, reg_lab in enumerate(self.rg):
                reg, lab = reg_lab
                c = hsv_to_rgb(i / n, 1.0, 1.0)
                not_assigned.difference_update(reg)
                self._rg_actors.append(
                    Points(self.points[reg], r=10, c=c).legend(lab)
                )
            # Points not assigned
            not_assigned = list(not_assigned)
            self._rg_actors.append(
                Points(self.points[not_assigned], r=9, c="black")
                .legend("Not assigned")
            )
            # Legend
            self._rg_actors.append(
                LegendBox(self._rg_actors, markers="o")
            )

        # Landmarks
        if kp_modified or pts_modified:
            not_assigned = set(range(len(self.points)))
            self._kp_actors = []
            if len(self.kp) > 0:
                not_assigned.difference_update(self.kp)
                self._kp_actors.append(
                    Points(self.points[self.kp], r=10, c="lime")
                    .legend("Landmarks")
                )
            # Points not assigned
            not_assigned = list(not_assigned)
            self._kp_actors.append(
                Points(self.points[not_assigned], r=9, c="black")
                .legend("Not assigned")
            )
            # Legend
            self._kp_actors.append(
                LegendBox(self._kp_actors, markers="o")
            )

        # Beard-Hair
        if bh_modified or pts_modified:
            not_assigned = set(range(len(self.points)))
            self._bh_actors = []
            if len(self.bh) > 0:
                not_assigned.difference_update(self.bh)
                self._bh_actors.append(
                    Points(self.points[self.bh], r=10, c="lime")
                    .legend("Beard/Hair")
                )
            # Points not assigned
            not_assigned = list(not_assigned)
            self._bh_actors.append(
                Points(self.points[not_assigned], r=9, c="black")
                .legend("Not assigned")
            )
            # Legend
            self._bh_actors.append(
                LegendBox(self._bh_actors, markers="o")
            )

        # Draw
        mode = self._mode_selector.currentIndex()
        self.plot.clear()
        if mode == 0:    # Regions
            actors = self._rg_actors + [self._labels]
        elif mode == 1:  # Landmarks
            actors = self._kp_actors + [self._labels]
        elif mode == 2:  # Beard/hair
            actors = self._bh_actors + [self._labels]
        elif mode == 3:  # Coin only
            actors = None
        else:
            raise ValueError(f"Invalid mode selected: {mode}")
        self.plot += self.mesh, actors
        if reset_camera:
            self.plot.reset_camera()
        self.plot.render()

    def closeEvent(self, e):
        self.plot.close()
        e.accept()


class QProcessButton(Qt.QPushButton):
    r"""
    Qt QPushButton designed to start/stop a QProcess.
    """

    def __init__(
        self,
        process: QProcess,
        parent: Qt.QWidget = None
    ):
        r"""
        Qt QPushButton designed to start/stop a QProcess.

        :param process: The QProcess instance to interact with.
        :type process: QProcess
        :param parent: The parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QProcessButton, self).__init__(parent)
        process.started.connect(self._to_stop)
        process.finished.connect(self._to_start)
        process.error.connect(self._to_start)
        self._to_start()

    def _to_start(self):
        self.setStyleSheet("background-color: rgb(0, 73, 0);")
        self.setText("Start")

    def _to_stop(self):
        self.setStyleSheet("background-color: rgb(73, 0, 0);")
        self.setText("Stop")


class QEnhancedLineEdit(Qt.QWidget):
    r"""
    QLineEdit enhanced with its own label.
    """

    def __init__(
        self,
        label: str,
        tooltip: str = None,
        parent: Qt.QWidget = None
    ):
        r"""
        QLineEdit enhanced with its own label.

        :param label: The entry label.
        :type label: str
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QEnhancedLineEdit, self).__init__(parent)
        layout = Qt.QVBoxLayout()
        layout.setMargin(0)
        layout.addWidget(Qt.QLabel(label))
        self._entry = Qt.QLineEdit()
        if tooltip is not None:
            self._entry.setToolTip(tooltip)
        layout.addWidget(self._entry)
        self.setLayout(layout)

    def __getattr__(self, item):
        return getattr(self._entry, item)

    def value(self) -> str:
        r"""
        Get the processed value of the entry content.

        :return: The value processed.
        :rtype: str
        """
        return self._entry.text()


class QFloatLineEdit(QEnhancedLineEdit):
    r"""
    QLineEdit designed to handle float values.
    """

    def __init__(
        self,
        label: str,
        tooltip: str = None,
        upper: float = None,
        lower: float = None,
        parent: Qt.QWidget = None
    ):
        r"""
        QLineEdit designed to handle float values.

        :param label: The entry label.
        :type label: str
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param upper: The upper bound of the float value.
        :type upper: float, optional
        :param lower: The lower bound of the float value.
        :type lower: float, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QFloatLineEdit, self).__init__(label, tooltip, parent)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound must be less than upper bound")
        self.lower = lower
        self.upper = upper

    def value(self) -> float:
        r"""
        Get the processed value of the entry content.

        :return: The value processed.
        :rtype: float
        """
        try:
            val = float(self.text())
        except ValueError:
            raise ValueError(f"Cannot parse {self._label}!")
        lower_ok = self.lower is None or val >= self.lower
        upper_ok = self.upper is None or val <= self.upper
        if not lower_ok and self.upper is None:
            raise ValueError(f"{self._label} must be greater than {self.lower}!")
        elif not upper_ok and self.lower is None:
            raise ValueError(f"{self._label} must be lower than {self.upper}!")
        elif not lower_ok or not upper_ok:
            raise ValueError(f"{self._label} must be between {self.lower} and {self.upper}!")
        else:
            return val


class QListLineEdit(QEnhancedLineEdit):
    r"""
    QLineEdit designed to handle list of integers.
    """

    def __init__(
        self,
        label: str,
        upper: int = None,
        n: int = None,
        tooltip: str = None,
        parent: Qt.QWidget = None,
    ):
        r"""
        :param label: The entry label.
        :type label: str
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param upper: The upper bound of the list items.
        :type upper: int, optional
        :param n: Fixed number of items in the list.
        :type n: int, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QListLineEdit, self).__init__(label, tooltip, parent)
        if upper is not None and upper <= 0:
            raise ValueError(f"The Upper bound must be greater than 0!")
        self.upper = upper if upper is not None else math.inf
        if n is not None and n <= 0:
            raise ValueError(f"n must be greater than 0!")
        self.n = n
        self.pattern = re.compile(r"(\d+|\d+-\d+)(,(\d+|\d+-\d+))*")

    def value(self) -> list[int]:
        r"""
        Get the processed value of the entry content.

        :return: The value processed.
        :rtype: list[int]
        """
        txt = self.text().replace(" ", "")
        if not self.pattern.fullmatch(txt):
            raise ValueError(f"Cannot parse {self._label}!")
        full_list = to_full_list(txt)
        if self.n is not None and len(full_list) != self.n:
            raise ValueError(f"Wrong number of values for {self._label}!")
        if any(not 0 <= x < self.upper for x in full_list):
            raise ValueError(f"Some values of {self._label} are out of bounds!")
        return full_list


class QNestedListLineEdit(QEnhancedLineEdit):
    r"""
    QLineEdit designed to handle nested list of integers.
    """

    def __init__(
        self,
        label: str,
        upper: int = None,
        tooltip: str = None,
        parent: Qt.QWidget = None,
    ):
        r"""
        QLineEdit designed to handle nested list of integers.

        :param label: The entry label.
        :type label: str
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param upper: The upper bound of the items' values.
        :type upper: int, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QNestedListLineEdit, self).__init__(label, tooltip, parent)
        if upper is not None and upper < 0:
            raise ValueError(f"The upper bound must be greater than 0!")
        self.upper = upper if upper is not None else math.inf
        self.pattern = re.compile(r"(\d+|\d+-\d+)([,|](\d+|\d+-\d+))*")

    def value(self) -> list[list[int]]:
        r"""
        Get the processed value of the entry content.

        :return: The value processed.
        :rtype: list[list[int]]
        """
        txt = self.text().replace(" ", "")
        if not self.pattern.fullmatch(txt):
            raise ValueError(f"Cannot parse {self._label}!")
        full_list = to_full_list(txt.replace("|", ""))  # Workaround
        if any(not 0 <= x < self.upper for x in full_list):
            raise ValueError(f"Some values of {self._label} are out of bounds!")
        full_list = [to_full_list(inner) for inner in txt.split("|")]
        return full_list


class QFileSelector(Qt.QWidget):
    r"""
    Widget to select a file from the filesystem.
    """

    file_selected = Signal(str)

    def __init__(
        self,
        label: str,
        tooltip: str = None,
        filter: str = "*",
        parent: Qt.QWidget = None,
    ):
        r"""
        Widget to select a file from the filesystem.

        :param label: The entry label.
        :type label: str
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param filter: Extension filter. Defaults to "*".
        :type filter: str, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QFileSelector, self).__init__(parent)
        vlayout = Qt.QVBoxLayout()
        vlayout.setMargin(0)
        vlayout.addWidget(Qt.QLabel(label))
        layout = Qt.QHBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(0)
        self._btn = Qt.QPushButton("Select")
        self._btn.clicked.connect(self._select_file)
        layout.addWidget(self._btn)
        self._entry = Qt.QLineEdit()
        self._entry.setReadOnly(True)
        self._entry.setDisabled(True)
        if tooltip is not None:
            self._entry.setToolTip(tooltip)
        layout.addWidget(self._entry)
        vlayout.addLayout(layout)
        self.setLayout(vlayout)
        self._filter = filter

    def __getattr__(self, item) -> Any:
        return getattr(self._entry, item)

    def value(self) -> Path:
        r"""
        Path to the selected file.

        :return: The absolute path to the selected file.
        :rtype: Path
        """
        if self._entry.text() == "":
            raise ValueError(f"{self._label}: no file selected.")
        path = Path(self._entry.text()).resolve()
        if self._filter != "*" and f"*{path.suffix.lower()}" != self._filter:
            raise ValueError(f"{self._label}: wrong file type!")
        return path

    def _select_file(self):
        path, _ = Qt.QFileDialog.getOpenFileName(
            self,
            caption=f"Select a {self._label.lower()} file",
            filter=self._filter,
        )
        self.set_path(path)

    def set_path(self, path: str):
        if path != "":
            self._entry.setText(path)
            self.file_selected.emit(path)


class QEnhancedComboBox(Qt.QWidget):
    r"""
    QComboBox enhanced with its own label.
    """

    def __init__(
        self,
        label: str,
        items: list[str],
        tooltip: str = None,
        parent: Qt.QWidget = None,
    ):
        r"""
        QComboBox enhanced with its own label.

        :param label: The entry label.
        :type label: str
        :param items: Combo-box items.
        :type items: list[str]
        :param tooltip: A hover tooltip.
        :type tooltip: str, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(QEnhancedComboBox, self).__init__(parent)
        layout = Qt.QVBoxLayout()
        layout.setMargin(0)
        self._label = label
        layout.addWidget(Qt.QLabel(label))
        self._combo = Qt.QComboBox()
        self._combo.addItems(items)
        layout.addWidget(self._combo)
        if tooltip is not None:
            self._combo.setToolTip(tooltip)
        self.setLayout(layout)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._combo, item)

    def value(self) -> str:
        return self._combo.currentText().lower()


class Logger(Qt.QLineEdit):
    r"""
    QLineEdit logging widget.
    """

    _levelToName = {
        logging.CRITICAL: 'CRITICAL',
        logging.ERROR: 'ERROR',
        logging.WARNING: 'WARNING',
        logging.INFO: 'INFO',
        logging.DEBUG: 'DEBUG',
        logging.NOTSET: 'NOTSET',
    }

    def __init__(
        self,
        log_path: Path = "",
        parent: Qt.QWidget = None,
    ):
        """
        QLineEdit logging widget.

        :param log_path: Path to the log file.
        :type log_path: Path, optional
        :param parent: Parent widget.
        :type parent: Qt.QWidget, optional
        """
        super(Logger, self).__init__(parent=parent)
        self.setReadOnly(True)
        self.setFont(QFont("Monospace"))

        if log_path is not None:
            log_path = (
                Path(__file__).parent.parent / "app.log"
                if log_path == ""
                else Path(log_path)
            )
            logging.basicConfig(level=logging.NOTSET)
            log_handler = RotatingFileHandler(
                filename=log_path,
                mode="a",
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=5,
                encoding="utf-8",
                delay=True
            )
            self._logger = logging.getLogger("resp.app")
            self._logger.addHandler(log_handler)
            self._logger.propagate = False
        else:
            self._logger = None

    def log(
        self,
        msg: str,
        level: int = logging.NOTSET
    ):
        r"""
        Print a log message to gui and file.

        :param msg: Log message.
        :type msg: str
        :param level: Logging level.
        :type level: int
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            level_name = Logger._levelToName[level]
        except KeyError:
            raise ValueError(f"Unknown log level: {level}")
        else:
            txt = f"[{ts}] {level_name}: {msg}"
            self.setText(txt)
            self.log_to_file(txt, level)

    def log_to_file(self, msg: str, level: int):
        r"""
        Print log message to file

        :param msg: Log message.
        :type msg: str
        :param level: Logging level.
        :type level: int
        """
        if self._logger is not None:
            self._logger.log(level, msg)

    def closeEvent(self, e):
        if self._logger is not None:
            logging.shutdown()
        e.accept()
