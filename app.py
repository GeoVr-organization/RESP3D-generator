#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ██████████████████████████████████
# █                                █
# █    █████████╗    ██████████╗   █
# █    ██╔═════██╗   ██╔═══════╝   █
# █    █████████╔╝   ████████╗     █
# █    ██╔═════██╗   ██╔═════╝     █
# █    ██║     ██║   ██████████╗   █
# █    ╚═╝     ╚═╝   ╚═════════╝   █
# █    ██████████╗   █████████╗    █
# █    ██╔═══════╝   ██╔═════██╗   █
# █    ██████████╗   █████████╔╝   █
# █    ╚═══════██║   ██╔══════╝    █
# █    ██████████║   ██║           █
# █    ╚═════════╝   ╚═╝           █
# █                                █
# ██████████████████████████████████
#
# https://ercresp.info/

import json
import logging
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import PySide2.QtWidgets as Qt
from PySide2.QtCore import QObject, QProcess, Qt as QtFlags
from PySide2.QtGui import QIcon
from vedo import Mesh

# Library path
_cwd = Path(__file__).parent.resolve()
sys.path.insert(0, str(_cwd))

import utils.ui as ui
from editor import EditorApp
from utils.common import relpath
from utils.io import load_pp
from utils.plot import plot_objects

# 0: label, 1: tooltip
_profile_scan = {
    "scan": ["Coin scan", "Coin scan file (.OBJ)"],
    "gender": ["Gender", "Emperor gender"],
    "direction": ["Direction", "Face orientation"]
}

# 0: label, 1: tooltip, 2: plot label
_profile_points = {
    "points": [
        "Coin points",
        "Coin points file (.pp)",
        None
    ],
    "front_neck": [
        "Front-neck indices",
        "From the beginning of the frontal neck to the cervical point",
        "Front-neck"
    ],
    "jaw": [
        "Jaw indices",
        "From the cervical point to the stomion",
        "Jaw"
    ],
    "upper_lip": [
        "Upper lip indices",
        "From the stomion to the subnasale",
        "Upper lip"],
    "nose": [
        "Nose and forehead indices",
        "From the subnasale to the front hairline",
        "Nose and forehead"
    ],
    "hair": [
        "Hair indices",
        "From the front hairline to the back hairline",
        "Hair"
    ],
    "back_neck": [
        "Back-neck indices",
        "From the back hairline to the and of end of the neck",
        "Back-neck"
    ],
    "eye": [
        "Eye index",
        "Pupil of the eye",
        "Eye"
    ],
    "lmks": [
        "Landmarks indices",
        "5 landmarks: gnathion, stomion, subnasale, nose tip, glabella",
        "Landmarks"
    ],
    "beard_hair": [
        "Hair and beard regions",
        "Indices of the points contouring the hair and beard regions",
        "Hair and beard"
    ],
}

# 0: label, 1: default value
_profile_generation = {
    "front_neck_w": ["Front-neck weight", "1.0"],
    "jaw_w": ["Jaw weight", "1.0"],
    "upper_lip_w": ["Upper lip weight", "1.0"],
    "nose_w": ["Nose-forehead weight", "1.0"],
    "hair_w": ["Hair weight", "1.0"],
    "back_neck_w": ["Back-neck weight", "1.0"],
    "eye_w": ["Eye weight", "1.0"],
    "tolerance": ["Hair tolerance", "0.1"],
    "hair_length": ["Hair length", "1.25"],
    "beard_length": ["Beard length", "0.5"],
    "smoothing_w": ["Smoothing weight", "1.0"]
}

# 0: label, 1: tooltip
_front_scan = {
    "front_scan": ["Coin scan", "Coin scan file (.OBJ)"],
}

# 0: label, 1: tooltip, 2: plot label
_front_points = {
    "front_points": [
        "Coin points",
        "Coin points file (.pp)",
        None
    ],
    "front_contour": [
        "Head contour indices",
        "Head contour",
        "Head contour"
    ],
    "front_lmks": [
        "Landmarks indices",
        "5 Landmarks: left eye, right eye, subnasale, stomion, gnathion",
        "Landmarks"
    ],
}

# 0: label, 1: default value
_front_generation = {
    "front_contour_w": ["Head contour weight", "0.01"],
}


class PipelineProcess(QProcess):
    _pipeline_script = str(_cwd / "pipelines.py")

    def __init__(self, parent: QObject = None):
        super(PipelineProcess, self).__init__(parent)
        self.setProcessChannelMode(QProcess.SeparateChannels)
        self.wp = None
        self.editor_window = None
        self.finished.connect(self._on_finished)

    def start_process(self, wp: Namespace):
        self.wp = wp
        # FIXME: VTK (on which Vedo is based) does not natively support
        #  WSL, causing a segmentation fault at the end of the pipeline,
        #  so a flag must be saved if the process is successful
        (wp.output / "success_flag").unlink(missing_ok=True)
        wp_path = str(wp.output.resolve() / "wp.pkl")
        with open(wp_path, "wb") as f:
            pickle.dump(wp, f)
        self.start(PipelineProcess._pipeline_script, [wp_path])

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        # FIXME: VTK (on which Vedo is based) does not natively support
        #  WSL, causing a segmentation fault at the end of the pipeline,
        #  so a flag must be saved if the process is successful
        # if exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit:
        if (self.wp.output / "success_flag").exists():
            mesh_path = str(self.wp.output / "final.obj")
            # Result preview
            plot_objects(
                Mesh(mesh_path, c="white"),
                title="Final mesh"
            )
            # Edit result
            self.editor_window = EditorApp(
                results_path=self.wp.output,
                parent=self.parent(),
                f=QtFlags.Window
            )
            self.editor_window.show()

    def close(self):
        if self.editor_window is not None:
            self.editor_window.close()
        super(PipelineProcess, self).close()


class RESPApp(Qt.QWidget):
    _title = "RESP 3D face generator"

    def __init__(
        self,
        parent: Qt.QWidget = None,
        f: QtFlags.WindowFlags = QtFlags.Widget
    ):
        super(RESPApp, self).__init__(parent, f)
        self.setWindowIcon(QIcon(str(_cwd / "other/app.ico")))
        self.setWindowTitle(RESPApp._title)
        self.resize(1280, 720)
        self.project_path = None

        # Subprocess
        self.worker = PipelineProcess(self)
        self.worker.readyReadStandardOutput.connect(self._update_output)
        self.worker.readyReadStandardError.connect(self._update_error)

        window_layout = Qt.QVBoxLayout()
        # Project controls #############################################
        project_layout = Qt.QHBoxLayout()
        for name in ["load", "save"]:
            btn = Qt.QPushButton(name.capitalize())
            btn.clicked.connect(getattr(self, f"_{name}_project"))
            project_layout.addWidget(btn)
        project_layout.addStretch()
        window_layout.addLayout(project_layout)
        ################################################################

        tabs = Qt.QTabWidget()
        tabs.setSizePolicy(Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Expanding)

        # Profile Tab ##################################################
        profile_tab = Qt.QSplitter()
        profile_tab.setContentsMargins(5, 5, 5, 5)
        # Options ======================================================
        left_pane = Qt.QScrollArea()
        left_pane.setWidgetResizable(True)
        left_pane_w = Qt.QWidget()
        left_pane_layout = Qt.QVBoxLayout()
        # Scan options -------------------------------------------------
        box = Qt.QGroupBox("Scan options")
        box_layout = Qt.QVBoxLayout()
        name = "scan"
        entry = ui.QFileSelector(
            label=_profile_scan[name][0],
            tooltip=_profile_scan[name][1],
            filter="*.obj"
        )
        entry.file_selected.connect(self._load_scan)
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        others_layout = Qt.QHBoxLayout()
        for name, items in zip(
            ["gender", "direction"],
            [["Male", "Female"], ["Left", "Right"]]
        ):
            cbox = ui.QEnhancedComboBox(
                label=_profile_scan[name][0],
                items=items,
                tooltip=_profile_scan[name][1])
            others_layout.addWidget(cbox)
            setattr(self, name, cbox)
        box_layout.addLayout(others_layout)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        # Points options -----------------------------------------------
        box = Qt.QGroupBox("Points options")
        box_layout = Qt.QVBoxLayout()
        name = "points"
        entry = ui.QFileSelector(
            label=_profile_points[name][0],
            tooltip=_profile_points[name][1],
            filter="*.pp"
        )
        entry.file_selected.connect(self._load_points)
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        for name in [
            "front_neck", "jaw", "upper_lip", "nose", "hair",
            "back_neck", "eye"
        ]:
            entry = ui.QListLineEdit(
                label=_profile_points[name][0],
                tooltip=_profile_points[name][1]
            )
            box_layout.addWidget(entry)
            setattr(self, name, entry)
        name = "lmks"
        entry = ui.QListLineEdit(
            label=_profile_points[name][0],
            tooltip=_profile_points[name][1],
            n=5
        )
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        name = "beard_hair"
        entry = ui.QNestedListLineEdit(
            label=_profile_points[name][0],
            tooltip=_profile_points[name][1]
        )
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        apply_indices = Qt.QPushButton("Update plot")
        apply_indices.clicked.connect(self._apply_indices)
        box_layout.addWidget(apply_indices)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        # Generation options -------------------------------------------
        box = Qt.QGroupBox("Generation options")
        box_layout = Qt.QVBoxLayout()
        autofill_btn = Qt.QPushButton("Autofill")
        autofill_btn.clicked.connect(self._autofill)
        box_layout.addWidget(autofill_btn)
        for name in [
            "front_neck_w", "jaw_w", "upper_lip_w", "nose_w", "hair_w",
            "back_neck_w", "eye_w", "tolerance", "beard_length",
            "hair_length", "smoothing_w"
        ]:
            entry = ui.QFloatLineEdit(
                label=_profile_generation[name][0],
                lower=0.0
            )
            box_layout.addWidget(entry)
            setattr(self, name, entry)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        left_pane_w.setLayout(left_pane_layout)
        left_pane.setWidget(left_pane_w)
        profile_tab.addWidget(left_pane)
        # Plot =========================================================
        self.profile_plt = ui.CoinPlot()
        profile_tab.addWidget(self.profile_plt)
        # ==============================================================
        tabs.addTab(profile_tab, "Profile")
        ################################################################

        # Front tab ####################################################
        front_tab = Qt.QSplitter()
        front_tab.setContentsMargins(5, 5, 5, 5)
        # Options ======================================================
        left_pane = Qt.QScrollArea()
        left_pane.setWidgetResizable(True)
        left_pane_w = Qt.QWidget()
        left_pane_layout = Qt.QVBoxLayout()
        # Enabled ------------------------------------------------------
        self.front_enabled = Qt.QCheckBox("Enabled (experimental)")
        left_pane_layout.addWidget(self.front_enabled)
        # Scan options -------------------------------------------------
        box = Qt.QGroupBox("Scan options")
        box_layout = Qt.QVBoxLayout()
        name = "front_scan"
        entry = ui.QFileSelector(
            label=_front_scan[name][0],
            tooltip=_front_scan[name][0],
            filter="*.obj"
        )
        entry.file_selected.connect(self._load_front_scan)
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        # Points options -----------------------------------------------
        box = Qt.QGroupBox("Points options")
        box_layout = Qt.QVBoxLayout()
        name = "front_points"
        entry = ui.QFileSelector(
            label=_front_points[name][0],
            tooltip=_front_points[name][1],
            filter="*.pp"
        )
        entry.file_selected.connect(self._load_front_points)
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        name = "front_contour"
        entry = ui.QListLineEdit(
            label=_front_points[name][0],
            tooltip=_front_points[name][1]
        )
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        name = "front_lmks"
        entry = ui.QListLineEdit(
            label=_front_points[name][0],
            tooltip=_front_points[name][1],
            n=5
        )
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        apply_indices = Qt.QPushButton("Update plot")
        apply_indices.clicked.connect(self._apply_front_indices)
        box_layout.addWidget(apply_indices)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        # Generation options -------------------------------------------
        box = Qt.QGroupBox("Generation options")
        box_layout = Qt.QVBoxLayout()
        autofill_btn = Qt.QPushButton("Autofill")
        autofill_btn.clicked.connect(self._front_autofill)
        box_layout.addWidget(autofill_btn)
        name = "front_contour_w"
        entry = ui.QFloatLineEdit(
            label=_front_generation[name][0],
            lower=0.0
        )
        box_layout.addWidget(entry)
        setattr(self, name, entry)
        box.setLayout(box_layout)
        left_pane_layout.addWidget(box)
        left_pane_w.setLayout(left_pane_layout)
        left_pane.setWidget(left_pane_w)
        profile_tab.addWidget(left_pane)
        left_pane_layout.addStretch()
        left_pane_w.setLayout(left_pane_layout)
        left_pane.setWidget(left_pane_w)
        front_tab.addWidget(left_pane)
        # Plot =========================================================
        self.front_plt = ui.CoinPlot()
        front_tab.addWidget(self.front_plt)
        # ==============================================================
        tabs.addTab(front_tab, "Front")
        ################################################################

        window_layout.addWidget(tabs)

        # Footer #######################################################
        footer_layout = Qt.QHBoxLayout()
        # Logger
        self.logger = ui.Logger()
        self.logger.setSizePolicy(Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Minimum)
        footer_layout.addWidget(self.logger)
        # Controls
        self.interactive = Qt.QCheckBox("Interactive")
        self.interactive.setChecked(True)
        footer_layout.addWidget(self.interactive)
        start_stop_btn = ui.QProcessButton(self.worker)
        start_stop_btn.clicked.connect(self._start_stop_processing)
        footer_layout.addWidget(start_stop_btn)
        window_layout.addLayout(footer_layout)
        ################################################################
        self.setLayout(window_layout)

    def _load_project(self):
        def deserialize(name, value):
            field = getattr(self, name)
            if isinstance(field, ui.QEnhancedComboBox):
                field.setCurrentIndex(value)
            elif isinstance(field, ui.QFileSelector):
                if value != "":
                    # Absolute path
                    value = str(project_path.parent / value)
                field.set_path(value)
            elif isinstance(field, ui.QEnhancedLineEdit):
                field.setText(value)
            elif isinstance(field, Qt.QCheckBox):
                field.setChecked(bool(value))
            else:
                raise ValueError(f"Unknown field: {type(field)}")

        project_path, _ = Qt.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open am existing project...",
            filter="*.json"
        )
        if project_path != "":  # No undo
            try:
                project_path = Path(project_path)
                # Clear the plots
                self.profile_plt.update_plot(clear=True)
                self.front_plt.update_plot(clear=True)
                # Fill profile and front fields
                fields = (
                    list(_profile_scan) + list(_profile_points)
                    + list(_profile_generation) + list(_front_scan)
                    + list(_front_points) + list(_front_generation)
                    + ["front_enabled"]
                )
                with open(project_path, "r") as file:
                    project = json.load(file)
                for k in fields:
                    deserialize(k, project[k])
                # Profile indices update
                self._apply_indices()
                if self.front_enabled.isChecked():
                    # Front indices update
                    self._apply_front_indices()
            except (TypeError, IOError):
                self.project_path = None
                self.setWindowTitle(RESPApp._title)
                self.logger.log(f"Cannot load the project {project_path}", level=logging.ERROR)
            else:
                self.project_path = project_path
                self.setWindowTitle(f"{RESPApp._title} [{project_path}]")
                self.logger.log(f"Loaded project {project_path}", level=logging.INFO)

    def _save_project(self):
        def serialize(name: str) -> int | str:
            field = getattr(self, name)
            if isinstance(field, ui.QEnhancedComboBox):
                return field.currentIndex()
            elif isinstance(field, ui.QFileSelector):
                if field.text() != "":
                    return relpath(project_path, field.value())
                else:
                    return ""
            elif isinstance(field, ui.QEnhancedLineEdit):  # any other field
                return field.text()
            elif isinstance(field, Qt.QCheckBox):
                return int(field.isChecked())
            else:
                raise ValueError(f"Unknown field type {type(field)}")

        project_path, _ = Qt.QFileDialog.getSaveFileName(
            parent=self,
            caption="Save current project...",
            filter="*.json"
        )
        if project_path != "":  # No undo
            try:
                project_path = Path(project_path)
                fields = (
                    list(_profile_scan) + list(_profile_points)
                    + list(_profile_generation) + list(_front_scan)
                    + list(_front_points) + list(_front_generation)
                    + ["front_enabled"]
                )
                project = {k: serialize(k) for k in fields}
                with open(project_path, "w") as file:
                    json.dump(project, file, indent=2)
            except (IOError, ValueError):
                self.project_path = None
                self.setWindowTitle(f"{RESPApp._title}")
                self.logger.log(f"Cannot save {project_path}", level=logging.ERROR)
            else:
                self.project_path = project_path
                self.setWindowTitle(f"{RESPApp._title} [{project_path}]")
                self.logger.log(f"Saved {project_path}", level=logging.INFO)

    def _load_scan(self, path: str):
        try:
            self.profile_plt.update_plot(mesh=Mesh(path, c="#bbbbbb"))
        except AttributeError:
            self.logger.log(f"Cannot load the scan {path}", level=logging.ERROR)
        else:
            self.logger.log(f"Loaded scan {path}", level=logging.INFO)

    def _load_front_scan(self, path: str):
        try:
            self.front_plt.update_plot(mesh=Mesh(path, c="#bbbbbb"))
        except AttributeError:
            self.logger.log(f"Cannot load the scan {path}", level=logging.ERROR)
        else:
            self.logger.log(f"Loaded scan {path}", level=logging.INFO)

    def _load_points(self, path: str):
        try:
            profile_points = load_pp(path)
            # Adjust the upper bound of the points
            for name in _profile_points:
                field = getattr(self, name)
                if isinstance(field, (ui.QListLineEdit, ui.QNestedListLineEdit)):
                    field.upper = len(profile_points)
            # Update profile plot
            self.profile_plt.update_plot(points=profile_points)
        except AttributeError:
            self.logger.log(f"Cannot load the points {path}", level=logging.ERROR)
        else:
            self.logger.log(f"Loaded points {path}", level=logging.INFO)

    def _load_front_points(self, path: str):
        try:
            front_points = load_pp(path)
            # Adjust the upper bound of the points
            for name in _front_points:
                field = getattr(self, name)
                if isinstance(field, (ui.QListLineEdit, ui.QNestedListLineEdit)):
                    field.upper = len(front_points)
            # Update front plot
            self.front_plt.update_plot(points=front_points)
        except AttributeError:
            self.logger.log(f"Cannot load the points {path}", level=logging.ERROR)
        else:
            self.logger.log(f"Loaded points {path}", level=logging.INFO)

    def _autofill(self):
        for name in _profile_generation:
            getattr(self, name).setText(_profile_generation[name][1])
        self.logger.log("Profile generation options filled!", level=logging.INFO)

    def _front_autofill(self):
        for name in _front_generation:
            getattr(self, name).setText(_front_generation[name][1])
        self.logger.log("Front generation options filled!", level=logging.INFO)

    def _apply_indices(self):
        try:
            rg = []
            for name in list(_profile_points)[1:-2]:
                field = getattr(self, name)
                if field.text() != "":
                    rg.append((field.value(), _profile_points[name][2]))
            kp = self.lmks.value()
            bh = self.beard_hair.value()
            # Update plot
            self.profile_plt.update_plot(rg=rg, kp=kp, bh=bh)
        except ValueError as e:
            self.logger.log(str(e), level=logging.ERROR)
        else:
            self.logger.log("Profile indices applied!", level=logging.INFO)

    def _apply_front_indices(self):
        try:
            rg = []
            for name in list(_front_points)[1:-1]:
                field = getattr(self, name)
                if field.text() != "":
                    rg.append((field.value(), _front_points[name][2]))
            kp = self.front_lmks.value()
            # Update front plot
            self.front_plt.update_plot(rg=rg, kp=kp)
        except ValueError as e:
            self.logger.log(str(e), level=logging.ERROR)
        else:
            self.logger.log("Front indices applied!", level=logging.INFO)

    def _start_stop_processing(self):
        if self.worker.state() == QProcess.NotRunning:  # Start
            try:
                wp = Namespace()
                # Create work package
                wp.interactive = self.interactive.isChecked()
                fields = (
                    list(_profile_scan) + list(_profile_points)
                    + list(_profile_generation)
                )
                wp.front_enabled = self.front_enabled.isChecked()
                if wp.front_enabled:
                    fields += (
                        list(_front_scan) + list(_front_points)
                        + list(_front_generation)
                    )
                for k in fields:
                    setattr(wp, k, getattr(self, k).value())
                # Fixes and other stuff
                wp.face = wp.jaw + wp.upper_lip + wp.nose
                wp.contour = wp.front_neck + wp.face + wp.hair + wp.back_neck
                wp.eye = wp.eye[0]  # Workaround for eye
                wp.hair_length /= 100  # to meters
                wp.beard_length /= 100  # to meters
                output = Qt.QFileDialog.getExistingDirectory(self)
                if output != "":  # No undo
                    wp.output = Path(output)
                    self.worker.start_process(wp)
                    self.worker.waitForStarted()
                    self.logger.log("Generation started!", level=logging.INFO)
            except ValueError as e:
                self.logger.log(str(e), level=logging.ERROR)
        else:  # Stop
            self.worker.kill()
            self.worker.waitForFinished()
            self.logger.log("Generation aborted!", level=logging.INFO)

    def _update_output(self):
        # Read the standard output from the subprocess
        output = self.worker.readAllStandardOutput().data().decode()
        # Update the logger with the output (no log)
        self.logger.setText(output)

    def _update_error(self):
        # Read the standard error from the subprocess
        output = self.worker.readAllStandardError().data().decode()
        # Send data directly to the log file
        self.logger.log_to_file(output, logging.DEBUG)

    def closeEvent(self, e):
        self.worker.close()
        self.profile_plt.close()
        self.front_plt.close()
        self.logger.close()
        e.accept()


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    app.setPalette(ui.DarkPalette())
    window = RESPApp()
    window.show()
    exit(app.exec_())
