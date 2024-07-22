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
import os
import pickle
import sys
import traceback
from argparse import Namespace
from pathlib import Path

# Workaround for opencv
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix
from torch.nn.functional import normalize, pad
from torch.optim import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Libraries path
_cwd = Path(__file__).parent.resolve()
sys.path.insert(0, str(_cwd))

from utils.common import torch_var_opts
from utils.flame import Flame, features, beard_uv_mask
from utils.geometry import (
    find_plane, project_on_plane, umeyama, wrap_curve_xy
)
from utils.io import create_displaced_mesh, load_points_pp, save_obj
from utils.math import minmax
from utils.metrics import adaptive_distance, euclidean_dist
from utils.plot import create_line, create_mesh, create_points, plot_objects
from utils.raster import (
    iqr_remove, masked_avg_pool2d, prepare_to_pytorch3d, pix_to_uv,
    simple_rasterization_settings, smooth_borders,
)
from utils.torch3dext import SubdivideMeshesUV, mesh_laplacian_smoothing2
from utils.training import Trainer

# Rasterization settings
img_size = 3000
ker_size = 301
subdivisions = 3
# Texture settings
h, w = 512, 512
blur_settings = {
    "ksize": (11, 11),
    "sigma": 0,
    "steps": 3,
}


def one_coin_pipeline(wp: Namespace):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the output folder
    wp.output.mkdir(exist_ok=True, parents=True)

    # Points alignment wrt the morphable model #########################################################################
    # Loading the morphable model and its features
    flame = Flame(wp.gender).to(device=dev)
    key_lines = features["key_lines"]
    key_points = features["key_points"]
    key_areas = features["key_areas"]
    mm_lmks = [
        key_points[k]
        for k in ["gnathion", "stomion", "subnasale", "pronasale", "glabella"]
    ]
    mm_contour = key_lines["profile"]
    mm_eyes = [key_points["left_eye"], key_points["right_eye"]]
    # Initializing the morphable model
    sign = {"left": -1, "right": 1}[wp.direction]
    gp = torch.tensor([0, sign * np.pi / 2, 0], device=dev)
    v, _ = flame(theta=gp)
    f = flame.faces
    t_mm = pad(-v[mm_contour, 2:].mean(dim=0), pad=[2, 0])
    v, _ = flame(theta=gp, t=t_mm)

    # Preprocessing
    print("(0/8) Preprocessing points...", flush=True)
    pc_points = load_points_pp(wp.points, device=dev)
    t, z_vec = find_plane(pc_points[wp.contour])
    pc_points = project_on_plane(pc_points, pos=t, normal=z_vec)
    pc_points -= t
    x_tmp = pc_points[wp.eye]
    z_vec *= torch.sign(torch.dot(torch.cross(pc_points[wp.lmks[1]], x_tmp), z_vec))
    y_vec = normalize(torch.cross(z_vec, x_tmp), dim=0)
    x_vec = normalize(torch.cross(y_vec, z_vec), dim=0)
    R = torch.stack([sign * x_vec, y_vec, sign * z_vec])
    pc_points @= R.T
    M = Transform3d(device=dev).translate(-t[None]).rotate(R.T)
    # Plot
    if wp.interactive:
        plot_objects(
            create_points(pc_points, scale=50, c="lime"),
            title="Preprocessing result"
        )

    print("(1/8) Points pre-alignment...", flush=True)
    # Pre-alignment using landmarks (evaluated on XY plane)
    t, r, s = umeyama(pc_points[wp.lmks, :2], v[mm_lmks, :2])
    R = torch.eye(3, device=dev)
    R[:2, :2] = r
    t = pad(t, pad=[0, 1])[None]
    pc_points = s * pc_points @ R + t
    # Save transformations for later
    M = M.scale(s).rotate(R).translate(t)
    # Plot
    if wp.interactive:
        plot_objects(
            create_mesh(v, f),
            create_line(v[mm_contour], c="red"),
            create_points(v[mm_lmks], scale=54, c="yellow"),
            create_points(pc_points[wp.contour], scale=50, c="lime"),
            create_points(pc_points[wp.lmks], scale=54, c="blue"),
            title="Pre-alignment result",
        )

    # Final alignment (evaluated on XY plane only)
    t_xy = torch.zeros(2, device=dev)
    r_z = torch.zeros(1, device=dev)
    s = torch.tensor(1.0, device=dev)
    trainer = Trainer(
        optimizer_cls=Adagrad,
        optimizer_params={"params": [t_xy, r_z, s], "lr": 0.1},
        scheduler_cls=ReduceLROnPlateau,
        scheduler_params={"patience": 200},
        epochs=1000,
        plot_losses=wp.interactive,
        desc="(2/8) Points final alignment",
    )
    for _ in trainer:
        R = axis_angle_to_matrix(pad(r_z, pad=[2, 0]))
        t = pad(t_xy, pad=[0, 1])[None]
        lmks_tmp = s * pc_points @ R + t
        trainer.loss = (
            adaptive_distance(v[mm_contour, :2], lmks_tmp[wp.face, :2])
            + 0.1 * euclidean_dist(v[mm_lmks, :2], lmks_tmp[wp.lmks, :2])
        )
    R = axis_angle_to_matrix(pad(r_z, pad=[2, 0]))
    t = pad(t_xy, pad=[0, 1])[None]
    pc_points = s * pc_points @ R + t
    # Save transformations for later
    M = M.scale(s).rotate(R).translate(t)
    # Plot
    if wp.interactive:
        plot_objects(
            create_mesh(v, f),
            create_line(v[mm_contour], c="red"),
            create_points(v[mm_lmks], scale=54, c="yellow"),
            create_points(pc_points[wp.contour], scale=50, c="lime"),
            create_points(pc_points[wp.lmks], scale=54, c="blue"),
            title="Final alignment",
        )

    # Morphing #########################################################################################################
    mesh_params = []
    h_atol = wp.tolerance * torch.linalg.vector_norm(
        pc_points[wp.contour].std(dim=0, **torch_var_opts)
    )
    for atol, desc, plot_title in [
        (0.0, "(3/8) Grown model generation", "Grown model result"),
        (h_atol, "(4/8) Optimal model generation", "Optimal model result"),
    ]:
        # Initialize the morphable model parameters
        beta = torch.zeros(250, device=dev)
        psi = torch.zeros(100, device=dev)
        neck_pose_x = torch.zeros(1, device=dev)
        trainer = Trainer(
            optimizer_cls=Adagrad,
            optimizer_params={"params": [beta, neck_pose_x], "lr": 0.1},
            scheduler_cls=ReduceLROnPlateau,
            scheduler_params={"patience": 1000},
            epochs=3000,
            plot_losses=wp.interactive,
            desc=desc,
        )
        for _ in trainer:
            neck_pose = pad(neck_pose_x, pad=[0, 2])
            new_t_mm, new_gp = flame.move_neck(
                new_beta=beta, new_np=neck_pose, old_t=t_mm, old_gp=gp
            )
            theta = torch.cat([new_gp, neck_pose])
            v, j = flame(beta=beta, theta=theta, t=new_t_mm)
            contour = v[mm_contour]
            trainer.loss = torch.stack([
                # Positions of the eyes (evaluated on XY only)
                wp.eye_w * euclidean_dist(v[mm_eyes, :2], pc_points[[wp.eye, wp.eye], :2]),
                # Profile shape terms
                wp.front_neck_w * adaptive_distance(contour, pc_points[wp.front_neck]),
                wp.jaw_w * adaptive_distance(contour, pc_points[wp.jaw]),
                wp.upper_lip_w * adaptive_distance(contour, pc_points[wp.upper_lip]),
                wp.nose_w * adaptive_distance(contour, pc_points[wp.nose]),
                wp.hair_w * adaptive_distance(contour, pc_points[wp.hair], atol=atol),
                wp.back_neck_w * adaptive_distance(contour, pc_points[wp.back_neck]),
            ]).sum()
            # Smoothing term
            if atol > 0:
                meshes = Meshes(v[None], f[None])
                trainer.loss += wp.smoothing_w * mesh_laplacian_smoothing2(meshes)

        # Get the last values
        neck_pose = pad(neck_pose_x, pad=[0, 2])
        new_t_mm, new_gp = flame.move_neck(
            new_beta=beta, new_np=neck_pose, old_t=t_mm, old_gp=gp
        )
        theta = torch.cat([new_gp, neck_pose])
        params = {
            "beta": beta.clone().detach(),
            "psi": psi.clone().detach(),
            "theta": theta.clone().detach(),
            "t": new_t_mm.clone().detach(),
        }
        # Plot
        if wp.interactive:
            v, _ = flame(**params)
            plot_objects(
                create_mesh(v, f),
                create_points(pc_points[wp.contour], scale=50, c="lime"),
                create_points(pc_points[wp.eye], scale=50, c="lime"),
                create_line(v[mm_contour], c="red"),
                create_points(v[mm_eyes], scale=50, c="red"),
                title=plot_title,
            )
        mesh_params.append(params)

    # Write morphable model parameters to disk
    torch.save(mesh_params, wp.output / "mesh_params.pt")

    # Hair and beard generation ########################################################################################
    with torch.no_grad():
        # Adjust coin pose (use preprocessing transformations)
        v_coin, f_coin_info, _ = load_obj(wp.scan, load_textures=False, device=dev)
        f_coin = f_coin_info.verts_idx
        v_coin = M.transform_points(v_coin)

        # Move hair landmarks in order to "enclose" the morphable model
        v, _ = flame(**mesh_params[0])
        v_uv = flame.uvcoords
        f_uv = flame.uvfaces
        vn = Meshes(v[None], f[None]).verts_normals_packed()
        pc_points[wp.hair] = wrap_curve_xy(
            polyline=v[mm_contour],
            pl_normals=vn[mm_contour],
            points=pc_points[wp.hair]
        )
        # Fit beard and hair regions to the NDC area
        roi = [i for reg in wp.beard_hair for i in reg]
        pcmin, pcmax = torch.aminmax(pc_points[roi, :2], dim=0)
        t = pad((pcmax + pcmin) / 2, pad=[0, 1])
        s = 2 / torch.max((pcmax - pcmin) * 1.01)  # A little bit smaller than 2
        pc_points, v_coin, v = [(x - t) * s for x in [pc_points, v_coin, v]]
        # Plot
        if wp.interactive:
            plot_objects(
                create_mesh(v, f),
                create_mesh(v_coin, f_coin, c="#bbbbbb"),
                create_points(pc_points, scale=50, c="lime"),
                title="Model alignment wrt the coin",
            )

        print("(5/8) Generating the uv map...", flush=True)
        # Coin mask generation
        coin_mask = np.zeros([img_size, img_size], dtype=np.uint8)
        pc_points[:, :2] = (pc_points[:, :2] + 1) / 2 * (img_size - 1)
        pc_points[:, 1] = (img_size - 1) - pc_points[:, 1]
        areas = [
            pc_points[reg, :2].numpy(force=True).astype(np.int32)
            for reg in wp.beard_hair
        ]
        coin_mask = cv2.drawContours(coin_mask, areas, -1, 255, -1)
        coin_mask = torch.tensor(coin_mask, device=dev).bool()
        # Prepare to Pytorch3D rasterization
        v_coin, v = [prepare_to_pytorch3d(x) for x in [v_coin, v]]
        # Mesh mask (and pixel-to-uv mapping)
        pix_to_uvcoords, mesh_mask = pix_to_uv(
            v, f, v_uv, f_uv, image_size=coin_mask.shape
        )
        # Exclusion mask
        settings = {**simple_rasterization_settings, "image_size": coin_mask.shape}
        pt3d_mm_mesh = Meshes(v[None], f[None])
        pix_to_face, _, _, _ = rasterize_meshes(meshes=pt3d_mm_mesh, **settings)
        excluded_verts = torch.cat([
            torch.tensor(key_areas[k], dtype=torch.long, device=dev)
            for k in ["left_ear", "right_ear", "lips"]
        ])
        excluded_faces = torch.where(torch.isin(f, excluded_verts).sum(dim=-1) == 3)[0]
        exclusion_mask = torch.isin(pix_to_face.squeeze(), excluded_faces, invert=True)
        # Final mask
        mask = mesh_mask & coin_mask & exclusion_mask
        # UV coordinates to draw
        uvpixels = pix_to_uvcoords[mask]
        # From UV coordinates to IJ indices
        uvpixels[:, 1] = 1 - uvpixels[:, 1]
        uvpixels *= torch.tensor([w - 1, h - 1]).to(uvpixels)
        uvpixels = uvpixels.long()

        print("(6/8) Generating the displacement map...", flush=True)
        # Rasterize the coin (zbuffer)
        coin_mesh = Meshes(v_coin[None], f_coin[None])
        _, depth, _, _ = rasterize_meshes(meshes=coin_mesh, **settings)
        # Remove outliers
        depth = iqr_remove(depth.squeeze(), mask=coin_mask)
        # Remove coin curvature (with values flip)
        avg, _ = masked_avg_pool2d(depth, mask, kernel_size=ker_size)
        depth = avg[mask] - depth[mask]
        tex = torch.zeros(h, w, device=dev)
        # Draw the displacement map (symmetrically)
        tex[uvpixels[:, 1], uvpixels[:, 0]] = depth
        tex[uvpixels[:, 1], (w - 1) - uvpixels[:, 0]] = depth
        # Linearly rescale between 0 and 1
        tex_mask = torch.zeros_like(tex).bool()
        tex_mask[uvpixels[:, 1], uvpixels[:, 0]] = True
        tex_mask[uvpixels[:, 1], (w - 1) - uvpixels[:, 0]] = True
        # FIXME: A new cleaning is necessary. Why are there outliers?
        tex = iqr_remove(tex, mask=tex_mask)
        tex[tex_mask] = minmax(tex[tex_mask])
        # Adapt displacement between beard and hair
        beard_mask = beard_uv_mask.to(device=dev)
        tex *= wp.beard_length * beard_mask + wp.hair_length * (1 - beard_mask)
        tex /= max(tex.max(), torch.finfo(tex.dtype).eps)
        # Smoothing the displacement and save it to disk
        tex = smooth_borders(tex.numpy(force=True), **blur_settings).clip(0, 1)
        tex = (tex * 255).astype(np.uint8)
        cv2.imwrite(str(wp.output / "disp.png"), tex)

        print("(7/8) Saving the optimal model (without hair/beard)...", flush=True)
        v, _ = flame(**mesh_params[1])
        # Densify the mesh
        subdivider = SubdivideMeshesUV(v[None], f[None], v_uv[None], f_uv[None])
        mm_meshes, mm_uvmaps = subdivider.subdivide(subdivisions)
        v, f = mm_meshes.get_mesh_verts_faces(0)
        v_uv, f_uv = mm_uvmaps.get_uvmap_verts_faces(0)
        save_obj(wp.output / "mesh.obj", v, f, uvcoords=v_uv, uvfaces=f_uv)

        print("(8/8) Saving the optimal model (with hair/beard)...", flush=True)
        create_displaced_mesh(
            input_path=wp.output / "mesh.obj",
            img_path=wp.output / "disp.png",
            output_path=wp.output / "final.obj",
            strength=max(wp.beard_length, wp.hair_length),
            mid_level=0.0,
        )

        print("Generation completed!", flush=True)
        # FIXME: VTK (on which Vedo is based) does not natively support
        #  WSL, causing a segmentation fault at the end of the pipeline,
        #  so a flag must be saved if the process is successful
        (wp.output / "success_flag").touch(exist_ok=True)


def two_coins_pipeline(wp: Namespace):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the output folder
    wp.output.mkdir(exist_ok=True, parents=True)

    # MM alignment wrt front landmarks #################################################################################
    # Load the morphable model and its features
    flame = Flame(wp.gender).to(device=dev)
    key_lines = features["key_lines"]
    key_points = features["key_points"]
    key_areas = features["key_areas"]

    print("(0/11) Front landmarks preprocessing...", flush=True)
    fc_points = load_points_pp(wp.front_points, device=dev)
    t, z_vec = find_plane(fc_points[wp.front_contour])
    fc_points = project_on_plane(fc_points, pos=t, normal=z_vec)
    fc_points -= t
    x_vec = normalize(
        # From right eye to left eye
        fc_points[wp.front_lmks[1]] - fc_points[wp.front_lmks[0]], dim=0
    )
    y_vec = normalize(
        # From chin to right eye
        fc_points[wp.front_lmks[0]] - fc_points[wp.front_lmks[-2]], dim=0
    )
    # Fix Z direction
    z_vec *= torch.sign(torch.dot(torch.cross(x_vec, y_vec), z_vec))
    # Orthogonalization
    y_vec = normalize(torch.cross(z_vec, x_vec), dim=0)
    x_vec = normalize(torch.cross(y_vec, z_vec), dim=0)
    R = torch.stack([x_vec, y_vec, z_vec])
    fc_points @= R.T
    # Mf = Transform3d(device=dev).translate(-t[None]).rotate(R.T)  # TODO: to be used
    # Plot
    if wp.interactive:
        plot_objects(
            create_points(fc_points, scale=50, c="lime"),
            title="Front landmarks preprocessing",
        )

    # Pre-alignment
    print("(1/11) Model pre-alignment wrt the front landmarks", flush=True)
    v, _ = flame()
    f = flame.faces
    mm_lmks = [
        key_points[key]
        for key in ["left_eye", "right_eye", "subnasale", "stomion", "menton"]
    ]
    r_mm = torch.zeros(3, device=dev)
    mm_fcontour = flame.find_contour(v, r_mm)
    s_mm = (
        torch.linalg.vector_norm(fc_points[wp.front_contour, :2].std(dim=0, **torch_var_opts))
        / torch.linalg.vector_norm(v[mm_fcontour, :2].std(dim=0, **torch_var_opts))
    )
    t_mm = fc_points[wp.front_contour].mean(dim=0) - (s_mm * v[mm_fcontour]).mean(dim=0)
    # Plot
    if wp.interactive:
        v, _ = flame(theta=r_mm, t=t_mm, s=s_mm)
        mm_fcontour = flame.find_contour(v, r_mm)
        plot_objects(
            create_mesh(v, f),
            create_points(v[mm_fcontour], scale=50, c="red"),
            create_points(v[mm_lmks], scale=50, c="yellow"),
            create_points(fc_points[wp.front_contour], scale=50, c="lime"),
            create_points(fc_points[wp.front_lmks], scale=50, c="blue"),
            title="Morphable model pre-alignment",
        )

    # Final alignment
    # rz_mm is 0 since we have aligned the front lmks using the eyes line
    tx, ty, tz = t_mm.split(split_size=1)
    rx, ry, rz = r_mm.split(split_size=1)
    trainer = Trainer(
        optimizer_cls=Adagrad,
        optimizer_params={"params": [tx, ty, ry, s_mm], "lr": 0.1},
        scheduler_cls=ReduceLROnPlateau,
        scheduler_params={"patience": 200},
        epochs=1000,
        plot_losses=wp.interactive,
        desc="(2/11) Final model alignment",
    )
    for _ in trainer:
        t_mm = torch.cat([tx, ty, tz])
        r_mm = torch.cat([rx, ry, rz])
        v, _ = flame(theta=r_mm, t=t_mm, s=s_mm)
        mm_fcontour = flame.find_contour(v, r_mm)
        trainer.loss = (
            adaptive_distance(fc_points[wp.front_contour, :2], v[mm_fcontour, :2])
            + 0.1 * euclidean_dist(v[mm_lmks, :2], fc_points[wp.front_lmks, :2])
        )
    r_mm = torch.cat([rx, ry, rz])
    t_mm = torch.cat([tx, ty, tz])
    # Plot
    if wp.interactive:
        v, _ = flame(theta=r_mm, t=t_mm, s=s_mm)
        mm_fcontour = flame.find_contour(v, r_mm)
        plot_objects(
            create_mesh(v, f),
            create_points(v[mm_fcontour], scale=50, c="red"),
            create_points(v[mm_lmks], scale=50, c="yellow"),
            create_points(fc_points[wp.front_contour], scale=50, c="lime"),
            create_points(fc_points[wp.front_lmks], scale=50, c="blue"),
            title="Morphable model final alignment",
        )

    # Profile landmarks alignment ######################################################################################
    # Change MM references
    v, _ = flame(theta=r_mm, t=t_mm, s=s_mm)
    mm_lmks = [
        key_points[key]
        for key in ["gnathion", "stomion", "subnasale", "pronasale", "glabella"]
    ]
    mm_pcontour = key_lines["profile"]
    mm_eyes = [key_points["left_eye"], key_points["right_eye"]]

    # Profile lmks preprocessing
    print("(3/11) Profile points preprocessing...", flush=True)
    pc_points = load_points_pp(wp.points, device=dev)
    c, normal = find_plane(pc_points[wp.contour])
    pc_points = project_on_plane(pc_points, pos=c, normal=z_vec)
    # Plot
    if wp.interactive:
        plot_objects(
            create_points(pc_points, scale=50, c="lime"),
            title="Profile points preprocessing",
        )

    # Pre-alignment
    print("(4/11) Pre-alignment of the profile points...", flush=True)
    t, R, s = umeyama(pc_points[wp.lmks], v[mm_lmks])
    pc_points = s * pc_points @ R + t
    Mp = Transform3d(device=dev).scale(s).rotate(R).translate(t[None])
    # Plot
    if wp.interactive:
        plot_objects(
            create_mesh(v, f),
            create_line(v[mm_pcontour], c="red"),
            create_points(v[mm_lmks], scale=54, c="yellow"),
            create_points(pc_points[wp.contour], scale=50, c="lime"),
            create_points(pc_points[wp.lmks], scale=54, c="blue"),
            title="Profile points pre-alignment",
        )

    # Fine-alignment
    t = torch.zeros(1, 3, device=dev)
    r = torch.zeros(3, device=dev)
    s = torch.tensor(1.0, device=dev)
    trainer = Trainer(
        optimizer_cls=Adagrad,
        optimizer_params={"params": [t, r, s], "lr": 0.1},
        scheduler_cls=ReduceLROnPlateau,
        scheduler_params={"patience": 200},
        epochs=1000,
        plot_losses=wp.interactive,
        desc="(5/11) Profile points final alignment",
    )
    for _ in trainer:
        R = axis_angle_to_matrix(r)
        lmks_tmp = s * pc_points @ R + t
        trainer.loss = (
            adaptive_distance(v[mm_pcontour], lmks_tmp[wp.face])
            + 0.1 * euclidean_dist(v[mm_lmks], lmks_tmp[wp.lmks])
        )
    R = axis_angle_to_matrix(r)
    pc_points = s * pc_points @ R + t
    Mp = Mp.scale(s).rotate(R).translate(t)
    # Plot
    if wp.interactive:
        plot_objects(
            create_mesh(v, f),
            create_line(v[mm_pcontour], c="red"),
            create_points(v[mm_lmks], scale=54, c="yellow"),
            create_points(pc_points[wp.contour], scale=50, c="lime"),
            create_points(pc_points[wp.lmks], scale=54, c="blue"),
            title="Profile points final alignment",
        )

    # Morphing #########################################################################################################
    profile_o, profile_n = find_plane(pc_points[wp.contour])
    # Morphing
    mesh_params = []
    h_atol = wp.tolerance * torch.linalg.vector_norm(pc_points[wp.contour].std(dim=0, **torch_var_opts))
    for atol, desc, plot_title in [
        (0.0, "(3/11) Grown model generation", "Grown model result"),
        (h_atol, "(4/11) Optimal model generation", "Optimal model result"),
    ]:
        # Initialize model parameters
        beta = torch.zeros(250, device=dev)
        psi = torch.zeros(100, device=dev)
        neck_pose_x = torch.zeros(1, device=dev)
        trainer = Trainer(
            optimizer_cls=Adagrad,
            optimizer_params={"params": [beta, neck_pose_x], "lr": 0.1},
            scheduler_cls=ReduceLROnPlateau,
            scheduler_params={"patience": 1000},
            epochs=3000,
            plot_losses=wp.interactive,
            desc=desc,
        )
        for _ in trainer:
            neck_pose = pad(neck_pose_x, pad=[0, 2])
            new_t_mm, new_gp = flame.move_neck(
                new_beta=beta, new_np=neck_pose, old_t=t_mm, old_gp=r_mm, s=s_mm
            )
            theta = torch.cat([new_gp, neck_pose])
            # Save current flame parameters
            v, j = flame(beta=beta, theta=theta, t=new_t_mm, s=s_mm)
            v_eyes = project_on_plane(v[mm_eyes], pos=profile_o, normal=profile_n)
            trainer.loss = torch.stack([
                # Eyes position
                wp.eye_w * euclidean_dist(v_eyes, pc_points[[wp.eye, wp.eye]]),
                # Profile shape
                wp.front_neck_w * adaptive_distance(v[mm_pcontour], pc_points[wp.front_neck]),
                wp.jaw_w * adaptive_distance(v[mm_pcontour], pc_points[wp.jaw]),
                wp.upper_lip_w * adaptive_distance(v[mm_pcontour], pc_points[wp.upper_lip]),
                wp.nose_w * adaptive_distance(v[mm_pcontour], pc_points[wp.nose]),
                wp.hair_w * adaptive_distance(v[mm_pcontour], pc_points[wp.hair], atol=atol),
                wp.back_neck_w * adaptive_distance(v[mm_pcontour], pc_points[wp.back_neck]),
            ]).sum()
            # Optimizations
            if atol > 0:
                mm_fcontour = flame.find_contour(v, theta)
                trainer.loss += torch.stack([
                    wp.front_contour_w * adaptive_distance(fc_points[wp.front_contour, :2], v[mm_fcontour, :2], atol=atol),
                    # TODO: stack for future works
                ]).sum()
                meshes = Meshes(v[None], f[None])
                trainer.loss += wp.smoothing_w * mesh_laplacian_smoothing2(meshes)

        # Get final values
        neck_pose = pad(neck_pose_x, pad=[0, 2])
        new_t_mm, new_gp = flame.move_neck(
            new_beta=beta, new_np=neck_pose, old_t=t_mm, old_gp=r_mm, s=s_mm
        )
        theta = torch.cat([new_gp, neck_pose])
        params = {
            "beta": beta.clone().detach(),
            "psi": psi.clone().detach(),
            "theta": theta.clone().detach(),
            "t": new_t_mm.clone().detach(),
            "s": s_mm.clone().detach(),
        }
        # Plot
        if wp.interactive:
            v, _ = flame(**params)
            plot_objects(
                create_mesh(v, f),
                create_points(pc_points[wp.contour], scale=50, c="lime"),
                create_points(pc_points[wp.eye], scale=50, c="lime"),
                create_line(fc_points[wp.front_contour], c="lime"),
                create_line(v[mm_pcontour], c="red"),
                create_points(v[mm_eyes], scale=50, c="red"),
                title=plot_title,
            )
        mesh_params.append(params)

    # Save mm deformation histories
    torch.save(mesh_params, wp.output / "mesh_params.pt")

    # Hair/Beard generation ############################################################################################
    with torch.no_grad():
        # Adjust coin pose (use preprocessing transformations)
        v_coin, f_coin_info, _ = load_obj(wp.scan, load_textures=False, device=dev)
        f_coin = f_coin_info.verts_idx
        v_coin = Mp.transform_points(v_coin)

        # Bring profile coin, profile points and MM on XY plane
        sign = {"left": -1, "right": 1}[wp.direction]
        pc_points, v_coin, v = [x - profile_o for x in [pc_points, v_coin, v]]
        x_tmp = pc_points[wp.eye]
        profile_n *= torch.sign(
            torch.dot(torch.cross(pc_points[wp.lmks[1]], x_tmp), profile_n)
        )
        y_vec = normalize(torch.cross(profile_n, x_tmp), dim=0)
        x_vec = normalize(torch.cross(y_vec, profile_n), dim=0)
        R = torch.stack([sign * x_vec, y_vec, sign * profile_n])
        pc_points, v_coin, v = [x @ R.T for x in [pc_points, v_coin, v]]

        # Move hair landmarks in order to "enclose" the morphable model
        v, _ = flame(**mesh_params[0])
        v_uv = flame.uvcoords
        f_uv = flame.uvfaces
        vn = Meshes(v[None], f[None]).verts_normals_packed()
        pc_points[wp.hair] = wrap_curve_xy(
            polyline=v[mm_pcontour],
            pl_normals=vn[mm_pcontour],
            points=pc_points[wp.hair]
        )

        # Fit beard and hair regions to the NDC area
        roi = [i for reg in wp.beard_hair for i in reg]
        pcmin, pcmax = torch.aminmax(pc_points[roi, :2], dim=0)
        t = pad((pcmax + pcmin) / 2, pad=[0, 1])
        s = 2 / torch.max((pcmax - pcmin) * 1.01)  # A little bit smaller than 2
        pc_points, v_coin, v = [(x - t) * s for x in [pc_points, v_coin, v]]
        # Plot
        if wp.interactive:
            plot_objects(
                create_mesh(v, f),
                create_mesh(v_coin, f_coin, c="#bbbbbb"),
                create_points(pc_points, scale=50, c="lime"),
                title="Model alignment wrt the coin",
            )

        print("(8/11) Generating the uv map...", flush=True)
        # Coin mask generation
        coin_mask = np.zeros([img_size, img_size], dtype=np.uint8)
        pc_points[:, :2] = (pc_points[:, :2] + 1) / 2 * (img_size - 1)
        pc_points[:, 1] = (img_size - 1) - pc_points[:, 1]
        areas = [
            pc_points[reg, :2].numpy(force=True).astype(np.int32)
            for reg in wp.beard_hair
        ]
        coin_mask = cv2.drawContours(coin_mask, areas, -1, 255, -1)
        coin_mask = torch.tensor(coin_mask, device=dev).bool()
        # Prepare to Pytorch3D rasterization
        v_coin, v = [prepare_to_pytorch3d(x) for x in [v_coin, v]]
        # Mesh mask (and pixel-to-uv mapping)
        pix_to_uvcoords, mesh_mask = pix_to_uv(
            v, f, v_uv, f_uv, image_size=coin_mask.shape
        )
        # Exclusion mask
        settings = {**simple_rasterization_settings, "image_size": coin_mask.shape}
        pt3d_mm_mesh = Meshes(v[None], f[None])
        pix_to_face, _, _, _ = rasterize_meshes(meshes=pt3d_mm_mesh, **settings)
        excluded_verts = torch.cat([
            torch.tensor(key_areas[k], dtype=torch.long, device=dev)
            for k in ["left_ear", "right_ear", "lips"]
        ])
        excluded_faces = torch.where(torch.isin(f, excluded_verts).sum(dim=-1) == 3)[0]
        exclusion_mask = torch.isin(pix_to_face.squeeze(), excluded_faces, invert=True)
        # Final mask
        mask = mesh_mask & coin_mask & exclusion_mask
        # UV coordinates to draw
        uvpixels = pix_to_uvcoords[mask]
        # From UV coordinates to IJ indices
        uvpixels[:, 1] = 1 - uvpixels[:, 1]
        uvpixels *= torch.tensor([w - 1, h - 1]).to(uvpixels)
        uvpixels = uvpixels.long()

        print("(9/11) Generating the displacement map...", flush=True)
        # Rasterize the coin (zbuffer)
        coin_mesh = Meshes(v_coin[None], f_coin[None])
        _, depth, _, _ = rasterize_meshes(meshes=coin_mesh, **settings)
        # Remove outliers
        depth = iqr_remove(depth.squeeze(), mask=coin_mask)
        # Remove coin curvature (with values flip)
        avg, _ = masked_avg_pool2d(depth, mask, kernel_size=ker_size)
        depth = avg[mask] - depth[mask]
        tex = torch.zeros(h, w, device=dev)
        # Draw the displacement map (symmetrically)
        tex[uvpixels[:, 1], uvpixels[:, 0]] = depth
        tex[uvpixels[:, 1], (w - 1) - uvpixels[:, 0]] = depth
        # Linearly rescale between 0 and 1
        tex_mask = torch.zeros_like(tex).bool()
        tex_mask[uvpixels[:, 1], uvpixels[:, 0]] = True
        tex_mask[uvpixels[:, 1], (w - 1) - uvpixels[:, 0]] = True
        # FIXME: A new cleaning is necessary. Why are there outliers?
        tex = iqr_remove(tex, mask=tex_mask)
        tex[tex_mask] = minmax(tex[tex_mask])
        # Adapt displacement between beard and hair
        beard_mask = beard_uv_mask.to(device=dev)
        tex *= wp.beard_length * beard_mask + wp.hair_length * (1 - beard_mask)
        tex /= max(tex.max(), torch.finfo(tex.dtype).eps)
        # Smoothing the displacement and save it to disk
        tex = smooth_borders(tex.numpy(force=True), **blur_settings).clip(0, 1)
        tex = (tex * 255).astype(np.uint8)
        cv2.imwrite(str(wp.output / "disp.png"), tex)

        print("(10/11) Saving the optimal model (without hair/beard)...", flush=True)
        v, _ = flame(**mesh_params[1])
        # Densify the mesh
        subdivider = SubdivideMeshesUV(v[None], f[None], v_uv[None], f_uv[None])
        mm_meshes, mm_uvmaps = subdivider.subdivide(subdivisions)
        v, f = mm_meshes.get_mesh_verts_faces(0)
        v_uv, f_uv = mm_uvmaps.get_uvmap_verts_faces(0)
        save_obj(wp.output / "mesh.obj", v, f, uvcoords=v_uv, uvfaces=f_uv)

        print("(11/11) Saving the optimal model (with hair/beard)...", flush=True)
        create_displaced_mesh(
            input_path=wp.output / "mesh.obj",
            img_path=wp.output / "disp.png",
            output_path=wp.output / "final.obj",
            strength=max(wp.beard_length, wp.hair_length),
            mid_level=0.0,
        )

        print("Generation completed!", flush=True)
        # FIXME: VTK (on which Vedo is based) does not natively support
        #  WSL, causing a segmentation fault at the end of the pipeline,
        #  so a flag must be saved if the process is successful
        (wp.output / "success_flag").touch(exist_ok=True)


if __name__ == "__main__":
    try:
        parse = argparse.ArgumentParser()
        parse.add_argument("input", type=Path)
        args = parse.parse_args()
        with open(args.input, "rb") as file:
            work_pkg = pickle.load(file)
        if work_pkg.front_enabled:
            two_coins_pipeline(work_pkg)
        else:
            one_coin_pipeline(work_pkg)
    except:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(
            "\nAn error occurred during the generation process. Please "
            "see log files for more details.",
            flush=True
        )
    else:
        print("\nGeneration completed!", flush=True)
