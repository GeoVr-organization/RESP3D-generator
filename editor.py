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

import argparse
import functools
import os
import pickle
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

# Workaround for opencv
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import numpy as np
import PySide2.QtWidgets as Qt
import torch
from PySide2.QtCore import Qt as QtFlags
from PySide2.QtGui import QIcon
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from tabulate import tabulate
from torch import Tensor
from torch.nn.functional import pad
from torch.optim import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vedo import Mesh, Plotter, Text2D
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Library path
_cwd = Path(__file__).parent.resolve()
sys.path.insert(0, str(_cwd))

from utils.common import torch_var_opts, StyledTerminal as ts
from utils.flame import Flame, features
from utils.geometry import apply_displacement, transform_mat
from utils.io import create_displaced_mesh, save_obj
from utils.torch3dext import SubdivideMeshesUV
from utils.ui import DarkPalette


class Editor(Plotter):
    RESET_EXPR = 0b001
    RESET_FACE = 0b010
    RESET_BH = 0b100
    RESET_ALL = 0b111

    def __init__(
        self,
        results_path: Path,
        qt_widget: QVTKRenderWindowInteractor = None
    ):
        torch.autograd.set_grad_enabled(False)
        super(Editor, self).__init__(
            axes=8,
            bg=(85, 85, 85),
            title="Editor",
            qt_widget=qt_widget
        )
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dev = dev
        self._v = None
        self._f = None
        self._v_uv = None
        self._f_uv = None

        self.results_path = results_path

        # Load the displacement map
        tex = cv2.imread(str(results_path / "disp.png"), cv2.IMREAD_UNCHANGED)
        tex = torch.tensor(tex, device=dev, dtype=torch.float32) / 255
        self.beard_mask = torch.load(
            _cwd / "utils/flame_binaries/beard_uv_mask.pt",
            map_location=dev
        ).to(dtype=tex.dtype)
        # Load hairs/bears displacements
        with open(results_path / "wp.pkl", "rb") as file:
            wp = pickle.load(file)
        self.wp = wp
        self.init_beard_length = wp.beard_length
        self.init_hair_length = wp.hair_length
        eps = torch.finfo(tex.dtype).eps
        # Uniforming the displacement (and preventing division by 0)
        coeff = (
            max(wp.beard_length, eps) * self.beard_mask
            + max(wp.hair_length, eps) * (1 - self.beard_mask)
        )
        # Rescale up
        self.tex = tex / (coeff * max(tex.max(), eps))

        # Load flame optimal parameters
        self.data = torch.load(
            results_path / "mesh_params.pt",
            map_location=dev
        )
        # Flame parameters
        params = self.data[-1]
        self.init_beta = params["beta"].to(device=dev)
        self.curr_beta = self.init_beta.clone()
        self.init_psi = params["psi"].to(device=dev)
        self.curr_psi = self.init_psi.clone()
        theta = params["theta"].to(device=dev)
        self.theta = pad(theta, pad=[0, 15 - theta.shape[0]])
        self._flame = Flame(wp.gender).to(device=dev)
        # Head mask
        areas = features["key_areas"]
        self._head_mask = torch.ones(self._flame.T.shape[0], device=dev, dtype=torch.bool)
        self._head_mask[areas["left_ear"]] = False
        self._head_mask[areas["right_ear"]] = False
        self._head_mask[areas["left_eyeball"]] = False
        self._head_mask[areas["right_eyeball"]] = False
        self._head_mask[areas["nose"]] = False
        self._head_mask[areas["neck"]] = False

        ## Instantiate GUI elements
        self.msg = Text2D("Preview", pos="top-center")
        self.add(self.msg)

        # Head shape sliders
        self.init_eyes_info, self.init_head_info = self._get_local_info(self.init_beta)
        self.eyes_dist_slider = self.add_slider(
            self.optimize,
            xmin=0.0, xmax=10.0,
            value=self.init_eyes_info.width.numpy(force=True) * 100,
            pos=[(0.065, 0.56), (0.065, 0.85)],
            title="Eyes distance (cm)",
            delayed=True,
        )
        self.eyes_dist_slider.GetSliderRepresentation().GetLabelProperty().SetOrientation(90)
        self.eyes_depth_slider = self.add_slider(
            self.optimize,
            xmin=0.0, xmax=10.0,
            value=self.init_eyes_info.depth.numpy(force=True) * 100,
            pos=[(0.94, 0.56), (0.94, 0.85)],
            title="Eyes depth (cm)",
            delayed=True,
        )
        self.eyes_depth_slider.GetSliderRepresentation().GetLabelProperty().SetOrientation(90)
        self.head_width_slider = self.add_slider(
            self.optimize,
            xmin=0.0, xmax=10.0,
            value=self.init_head_info.width.numpy(force=True) * 100,
            pos=[(0.065, 0.15), (0.065, 0.54)],
            title="Head width (std dev., cm)",
            delayed=True,
        )
        self.head_width_slider.GetSliderRepresentation().GetLabelProperty().SetOrientation(90)
        self.head_depth_slider = self.add_slider(
            self.optimize,
            xmin=0.0, xmax=10.0,
            value=self.init_head_info.depth.numpy(force=True) * 100,
            pos=[(0.94, 0.15), (0.94, 0.54)],
            title='Head depth (std dev., cm)',
            delayed=True
        )
        self.head_depth_slider.GetSliderRepresentation().GetLabelProperty().SetOrientation(90)
        self.basis_slider = self.add_slider(
            self._basis_slider_callback,
            xmin=0, xmax=99,
            value=0,
            title="Expression basis",
            pos=1
        )
        self.psi_slider = self.add_slider(
            self._psi_slider_callback,
            xmin=-3.0, xmax=3.0,
            value=self.curr_psi[int(self.basis_slider.value)],
            title="Expr. basis magnitude",
            pos=2,
        )
        self.psi_slider.AddObserver("StartInteractionEvent", self._psi_slider_callback_start)
        self.psi_slider.AddObserver("EndInteractionEvent", self._psi_slider_callback_end)

        # Hair and beard sliders
        self.beard_slider = self.add_slider(
            lambda *_: self._update_mesh(),
            xmin=0.0, xmax=10.0,
            value=wp.beard_length * 100,
            title="Beard length (cm)",
            pos=3
        )
        if wp.beard_length <= 0:
            self.beard_slider.off()
            ts.sprint("Warning: beard not detected, slider disabled!", ts.YELLOW)
        self.hair_slider = self.add_slider(
            lambda *_: self._update_mesh(),
            xmin=0.0, xmax=10.0,
            value=wp.hair_length * 100,
            title="Hair length (cm)",
            pos=4
        )
        if wp.hair_length <= 0:
            self.hair_slider.off()
            ts.sprint("Warning: hair not detected, slider disabled!", ts.YELLOW)

        ## Instantiate expression mesh
        v, _ = self._flame(beta=self.curr_beta, psi=self.curr_psi, theta=self.theta)
        self._expr_mesh = Mesh(
            [v.numpy(force=True), self._flame.faces.numpy(force=True)],
            c="yellow",
        )
        ## Instantiate the final mesh
        subdivider = SubdivideMeshesUV(
            verts=v[None],
            faces=self._flame.faces[None],
            uv_verts=self._flame.uvcoords[None],
            uv_faces=self._flame.uvfaces[None],
        )
        # Densify the mesh
        meshes, uvmaps = subdivider.subdivide(3)
        self._v, self._f = meshes.get_mesh_verts_faces(0)
        self._v_uv, self._f_uv = uvmaps.get_uvmap_verts_faces(0)
        beard_s = self.beard_slider.value / 100
        hair_s = self.hair_slider.value / 100
        c = beard_s * self.beard_mask + hair_s * (1 - self.beard_mask)
        tex = c * self.tex
        # Rescale between 0 and 1 (and preventing division by zero)
        tex /= max(tex.max(), torch.finfo(tex.dtype).eps)
        # Apply displacement
        v = apply_displacement(
            self._v, self._f, self._v_uv, self._f_uv,
            texture=tex,
            midlevel=0.0,
            strength=max(beard_s, hair_s)
        )
        self._final_mesh = Mesh(
            [v.numpy(force=True), self._f.numpy(force=True)],
            c="white",
        )
        self.add(self._final_mesh)

    @staticmethod
    def _plot_processing(text="Processing...", color=(85, 85, 85)):
        def plot_processing_decorator(func):
            @functools.wraps(func)
            def wrapper(self, *_args, **_kwargs):
                prev_msg = self.msg.text()
                prev_color = self.background()
                self.msg.text(text)
                self.background(color)
                self.render()
                tc = f"\033[38;2;{color[0]};{color[1]};{color[2]}m"
                ts.sprint(f"\n{text} ".ljust(80, "="), tc, flush=True)
                func(self, *_args, **_kwargs)
                self.msg.text(prev_msg)
                self.background(prev_color)
                self.render()
            return wrapper
        return plot_processing_decorator

    @_plot_processing(text="Processing...", color=(175, 175, 85))
    def optimize(self, _w, _e):
        # Get new desired values
        new_eyes_dist = torch.tensor(self.eyes_dist_slider.value / 100, device=self.dev)
        new_eyes_depth = torch.tensor(self.eyes_depth_slider.value / 100, device=self.dev)
        new_head_width = torch.tensor(self.head_width_slider.value / 100, device=self.dev)
        new_head_depth = torch.tensor(self.head_depth_slider.value / 100, device=self.dev)

        with torch.enable_grad():
            # Restart from beginning
            self.curr_beta = self.init_beta.clone().requires_grad_(True)
            # Optimize
            optimizer = Adagrad([self.curr_beta], lr=0.1)
            scheduler = ReduceLROnPlateau(optimizer, patience=1000)
            i = 0
            _0 = torch.tensor(0).to(self.curr_beta)
            while True:
                optimizer.zero_grad()
                # Current error
                curr_eyes_info, curr_head_info = self._get_local_info(self.curr_beta)
                err = (
                    torch.abs(curr_head_info.width - new_head_width) ** 2
                    + torch.abs(curr_head_info.depth - new_head_depth) ** 2
                    + torch.abs(curr_eyes_info.width - new_eyes_dist) ** 2
                    + torch.abs(curr_eyes_info.depth - new_eyes_depth) ** 2
                )
                if torch.isclose(err, _0):
                    break
                err.backward(retain_graph=True)
                scheduler.step(err)
                optimizer.step()
                # Log
                if i % 50 == 0:
                    print(f"iter: {i} -- error: {err.item()}", flush=True)
                i += 1
            self.curr_beta.requires_grad = False

        # Summary
        content = [
            ["Eye distance", curr_eyes_info.width.item(), new_eyes_dist.item()],
            ["Eye depth", curr_eyes_info.depth.item(), new_eyes_depth.item()],
            ["Head width", curr_head_info.width.item(), new_head_width.item()],
            ["Head depth", curr_head_info.depth.item(), new_head_depth.item()],
        ]
        print(f"\n{tabulate(content, headers=['Category', 'Current', 'Target'])}")
        self.eyes_dist_slider.value = curr_eyes_info.width.item() * 100
        self.eyes_depth_slider.value = curr_eyes_info.depth.item() * 100
        self.head_width_slider.value = curr_head_info.width.item() * 100
        self.head_depth_slider.value = curr_head_info.depth.item() * 100
        self._update_mesh(self.curr_beta, self.curr_psi)

    @_plot_processing(text="Reset...", color=(175, 85, 85))
    def reset(self, what: int = RESET_ALL):
        if bool(what & Editor.RESET_EXPR):
            # Expression
            self.curr_psi = self.init_psi.clone()
            self.basis_slider.value = 0
            self.psi_slider.value = self.curr_psi[0].item()
        if bool(what & Editor.RESET_FACE):
            # Head shape
            self.curr_beta = self.init_beta.clone()
            self.eyes_dist_slider.value = self.init_eyes_info.width.item() * 100
            self.eyes_depth_slider.value = self.init_eyes_info.depth.item() * 100
            self.head_width_slider.value = self.init_head_info.width.item() * 100
            self.head_depth_slider.value = self.init_head_info.depth.item() * 100
        if bool(what & Editor.RESET_BH):
            # Beard and hair
            self.beard_slider.value = self.init_beard_length * 100
            self.hair_slider.value = self.init_hair_length * 100
        # Update mesh
        self._update_mesh(self.curr_beta, self.curr_psi)

    @_plot_processing(text="Saving...", color=(85, 175, 85))
    def save(self):
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = self.results_path / f"modified_{timestamp}"
        output_path.mkdir(exist_ok=True, parents=True)

        print("Saving modified displacement map and wp... ", end="", flush=True)
        hair_s = self.hair_slider.value / 100
        beard_s = self.beard_slider.value / 100
        c = beard_s * self.beard_mask + hair_s * (1 - self.beard_mask)
        tex = c * self.tex
        tex /= max(tex.max(), torch.finfo(tex.dtype).eps)
        tex = (tex * 255).numpy(force=True).astype(np.uint8)
        cv2.imwrite(str(output_path / "disp.png"), tex)
        self.wp.beard_length = beard_s
        self.wp.hair_length = hair_s
        with open(output_path / "wp.pkl", "wb") as f:
            pickle.dump(self.wp, f)
        print("Done!", flush=True)

        print("Saving new mesh parameters... ", end="", flush=True)
        self.data[-1]["beta"] = self.curr_beta.clone().detach()
        self.data[-1]["psi"] = self.curr_psi.clone().detach()
        torch.save(self.data, output_path / "mesh_params.pt")
        print("Done!", flush=True)

        print("Saving modified model (without hairs/beard)... ", end="", flush=True)
        save_obj(
            output_path / "mesh.obj",
            vertices=self._v,
            faces=self._f,
            uvcoords=self._v_uv,
            uvfaces=self._f_uv,
        )
        print("Done!", flush=True)

        print("Saving modified model (with hairs/beard)... ", end="", flush=True)
        create_displaced_mesh(
            input_path=output_path / "mesh.obj",
            img_path=output_path / "disp.png",
            output_path=output_path / "final.obj",
            strength=max(hair_s, beard_s),
            mid_level=0.0,
        )
        print("Done!", flush=True)

        with open(output_path / "slider_values.pkl", "wb") as f:
            values = {
                "eyes_dist": self.eyes_dist_slider.value,
                "eyes_depth": self.eyes_depth_slider.value,
                "head_width": self.head_width_slider.value,
                "head_depth": self.head_depth_slider.value,
                "beard_length": self.beard_slider.value,
                "hair_length": self.hair_slider.value,
            }
            pickle.dump(values, f)

    # Expression callbacks #############################################
    def _basis_slider_callback(self, w, _e):
        # Change and update the magnitude slider
        self.psi_slider.value = self.curr_psi[int(w.value)]

    def _psi_slider_callback_start(self, _w, _e):
        # Switch to the "expression" mesh
        self.remove(self._final_mesh)
        v, _ = self._flame(
            beta=self.curr_beta,
            psi=self.curr_psi,
            theta=self.theta,
        )
        self._expr_mesh.vertices = v.numpy(force=True)
        self.add(self._expr_mesh)

    def _psi_slider_callback(self, w, _e):
        # Update "expression" mesh
        basis_idx = int(self.basis_slider.value)
        self.curr_psi[basis_idx] = w.value
        v, _ = self._flame(
            beta=self.curr_beta,
            psi=self.curr_psi,
            theta=self.theta,
        )
        self._expr_mesh.vertices = v.numpy(force=True)

    def _psi_slider_callback_end(self, _w, _e):
        # Switch back to the final mesh and update it
        self.remove(self._expr_mesh)
        self.add(self._final_mesh)
        self._update_mesh(self.curr_beta, self.curr_psi)
    ####################################################################

    def _update_mesh(self, beta: Tensor = None, psi: Tensor = None):
        if mesh_changed := (beta is not None or psi is not None):
            # Recompute new vertices
            v, _ = self._flame(beta=beta, psi=psi, theta=self.theta)
            # Densify mesh (3 times)
            meshes = Meshes(v[None], self._flame.faces[None])
            for _ in range(3):
                meshes = SubdivideMeshes(meshes).forward(meshes)
            self._v = meshes.verts_packed()

        # Rescale displacements
        beard_s = self.beard_slider.value / 100
        hair_s = self.hair_slider.value / 100
        c = beard_s * self.beard_mask + hair_s * (1 - self.beard_mask)
        tex = c * self.tex
        # Rescale between 0 and 1 (and preventing division by zero)
        tex /= max(tex.max(), torch.finfo(tex.dtype).eps)
        # Apply the displacement
        v = apply_displacement(
            self._v, self._f,
            self._v_uv, self._f_uv,
            texture=tex,
            midlevel=0.0,
            strength=max(beard_s, hair_s),
            use_precomputed=not mesh_changed
        )
        self._final_mesh.vertices = v.numpy(force=True)

    def _get_local_info(self, beta: Tensor) -> tuple[Namespace, Namespace]:
        v, j = self._flame(beta=beta, theta=self.theta)
        # Compute global-to-local transformation matrix
        local_rot = self.theta.view(-1, 3)
        global_pos = j[1, :, None]
        global_rot = torch.matmul(
            axis_angle_to_matrix(local_rot[0]),
            axis_angle_to_matrix(local_rot[1])
        )
        l2g = transform_mat(R=global_rot, T=global_pos)
        g2l = torch.inverse(l2g).T
        # Eyes info
        j = pad(j, pad=[0, 1], value=1)
        j = j @ g2l
        eyes_info = Namespace(
            width=torch.abs(j[-2, 0] - j[-1, 0]),
            depth=j[[-2, -1], 2].mean()
        )
        # Head info
        v = pad(v[self._head_mask], pad=[0, 1], value=1)
        v = torch.matmul(v, g2l)
        head_info = Namespace(
            width=v[:, 0].std(dim=0, **torch_var_opts),
            depth=v[:, 2].std(dim=0, **torch_var_opts),
        )
        return eyes_info, head_info


class EditorApp(Qt.QWidget):
    _title = "Editor"

    def __init__(
        self,
        results_path: Path,
        parent: Qt.QWidget = None,
        f: QtFlags.Widget = QtFlags.Widget
    ):
        super(EditorApp, self).__init__(parent, f)
        # Don't override parent taskbar icon
        if self.parentWidget() is None:
            self.setWindowIcon(QIcon(str(_cwd / "other/editor.ico")))
        self.setWindowTitle(f"{EditorApp._title} [{results_path}]")
        self.resize(1000, 1000)

        windowLayout = Qt.QVBoxLayout()

        editorWidget = QVTKRenderWindowInteractor(self)
        self.editor = Editor(results_path, qt_widget=editorWidget).show()
        windowLayout.addWidget(editorWidget)

        buttons_layout = Qt.QHBoxLayout()
        buttons_layout.setMargin(0)
        reset_button = Qt.QPushButton("Reset expression")
        reset_button.clicked.connect(lambda: self.editor.reset(self.editor.RESET_EXPR))
        buttons_layout.addWidget(reset_button)
        reset_button = Qt.QPushButton("Reset face shape")
        reset_button.clicked.connect(lambda: self.editor.reset(self.editor.RESET_FACE))
        buttons_layout.addWidget(reset_button)
        reset_button = Qt.QPushButton("Reset beard/hair")
        reset_button.clicked.connect(lambda: self.editor.reset(self.editor.RESET_BH))
        buttons_layout.addWidget(reset_button)
        reset_button = Qt.QPushButton("Reset all")
        reset_button.setStyleSheet("background-color: rgb(73, 0, 0)")
        reset_button.clicked.connect(lambda: self.editor.reset(self.editor.RESET_ALL))
        buttons_layout.addWidget(reset_button)

        buttons_layout.addStretch()
        save_button = Qt.QPushButton("Save")
        save_button.setStyleSheet("background-color: rgb(0, 73, 0)")
        save_button.clicked.connect(self.editor.save)
        buttons_layout.addWidget(save_button)
        windowLayout.addLayout(buttons_layout)

        self.setLayout(windowLayout)

    def closeEvent(self, e):
        self.editor.close()
        e.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Editor",
        description="Edit pipeline results."
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to the results directory. The folder must contain "
             "mesh_params.pt, disp.png and wp.pkl.",
        metavar="RESULTS_PATH",
        nargs='?'
    )
    path = parser.parse_args().results_path

    app = Qt.QApplication()
    app.setPalette(DarkPalette())
    if path is None:
        path = Qt.QFileDialog.getExistingDirectory()
        if path == "":
            exit(0)
        path = Path(path).resolve()
    window = EditorApp(results_path=path)
    window.show()
    exit(app.exec_())
