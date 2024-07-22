from pathlib import Path
from xml.etree.ElementTree import parse

import bpy
import numpy as np
import torch
from numpy import ndarray
from torch import LongTensor, Tensor

from .common import output_redirected, wrong_shape_msg as ws_msg


def load_pp(filepath: str | Path) -> ndarray:
    r"""
    Load 3D points from a MeshLab PickedPoints file.

    :param filepath: The file path.
    :type filepath: Path | str
    :return: A NumPy array containing the coordinates of the points.
    :rtype: ndarray
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".pp":
        raise ValueError("Unknown file type.")
    root = parse(filepath.resolve()).getroot()
    points = [
        [float(p.attrib[a]) for a in "xyz"]
        for p in root.findall("point")
    ]
    return np.array(points).astype(np.float32)


def load_points_pp(filepath: str | Path, **kwargs) -> Tensor:
    r"""
    Load 3D points from a MeshLab PickedPoints file.

    :param filepath: The file path.
    :type filepath: Path | str
    :param kwargs: PyTorch Tensor parameters.
    :return: A tensor containing the coordinates of the points.
    :rtype: Tensor
    """
    points = load_pp(filepath)
    return torch.tensor(points, **kwargs)


def save_obj(
    filepath: str | Path,
    vertices: Tensor,
    faces: LongTensor,
    *,
    uvcoords: Tensor = None,
    uvfaces: LongTensor = None,
    material_name: str = None
):
    r"""
    Save the mesh to Wavefront OBJ file.

    :param filepath: The output file path.
    :type filepath: str | Path
    :param vertices: The mesh vertices. Its shape must be (V, 3).
    :type vertices: Tensor
    :param faces: The mesh faces. Its shape must be (F, 3).
    :type faces: LongTensor
    :param uvcoords: The UV map coordinates. Its shape must be (T, 2).
    :type uvcoords: Tensor, optional
    :param uvfaces: The UV map faces. Its shape must be (F, 3).
    :type uvfaces: LongTensor, optional
    :param material_name: Material name for the texture.
    :type material_name: str, optional
    """
    filepath = Path(filepath)
    if filepath.suffix != ".obj":
        raise ValueError("Unknown file extension.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(ws_msg(vertices, "vertices", "(V, 3)"))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(ws_msg(faces, "faces", "(F, 3)"))
    if torch.any(faces.lt(0)) or torch.any(faces.ge(vertices.shape[0])):
        raise IndexError(
            f"Some faces indices are out of the range [0, {vertices.shape[0]})."
        )
    write_uv = uvcoords is not None and uvfaces is not None
    if write_uv:
        if uvcoords.ndim != 2 or uvcoords.shape[1] != 2:
            raise ValueError(ws_msg(uvcoords, "uvcoords", "(T, 2)"))
        if uvfaces.ndim != 2 or uvfaces.shape[1] != 3:
            raise ValueError(ws_msg(vertices, "uvfaces", "(F, 3)"))
        if torch.any(uvfaces.lt(0)) or torch.any(uvfaces.ge(uvcoords.shape[0])):
            raise IndexError(
                f"Some face indices are outside the range [0, {uvcoords.shape[0]})."
            )
        if faces.shape[0] != uvfaces.shape[0]:
            raise ValueError(
                f"The size of faces {faces.shape[0]} does not match the "
                f"size of uvfaces {uvfaces.shape[0]}."
            )
    with open(filepath, "w") as file:
        if write_uv and material_name is not None:
            file.writelines([
                f"mtllib {filepath.with_suffix('.mtl').name}\n",
                f"usemtl {material_name}\n"
            ])
        # Mesh vertices
        file.writelines([
            f"v {v[0]} {v[1]} {v[2]}\n"
            for v in vertices.cpu()
        ])
        if write_uv:
            # UV coordinates
            file.writelines([
                f"vt {v[0]} {v[1]}\n"
                for v in uvcoords.cpu()
            ])
            # Mesh faces alongside UV-faces
            faces = torch.cat([faces, uvfaces], dim=1) + 1
            file.writelines([
                f"f {f[0]}/{f[3]} {f[1]}/{f[4]} {f[2]}/{f[5]}\n"
                for f in faces.cpu()
            ])
        else:
            # Mesh faces only
            faces += 1
            file.writelines([
                f"f {f[0]} {f[1]} {f[2]}\n"
                for f in faces.cpu()
            ])


def create_displaced_mesh(
    input_path: str | Path,
    img_path: str | Path,
    *,
    output_path: str | Path = None,
    strength: float = 1.0,
    mid_level: float = 0.5,
):
    r"""
    Apply a displacement to an input mesh (OBJ) and save it to file.

    :param input_path: The input mesh path.
    :type input_path: str | Path
    :param img_path: The input texture path.
    :type img_path: str | Path
    :param output_path: The output mesh path. By default, it's the same
        as the input mesh, but with path stem "final".
    :type output_path: str | Path, optional
    :param strength: Amount to displace geometry. Defaults to 1.0.
    :type strength:  float, optional
    :param mid_level: Material value that gives no displacement.
        Defaults to 0.5.
    :type mid_level: float, optional
    """
    # Importing the mesh
    input_path = Path(input_path).resolve()
    with output_redirected():
        # Cleaning blender default scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.obj_import(
            filepath=str(input_path),
            directory=str(input_path.parent),
            files=[{"name": input_path.name}],
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh = bpy.context.object
        if strength > 0:  # Preventing error on strength == 0
            # Importing the displacement map
            img_path = Path(img_path).resolve()
            texture = bpy.data.textures.new(name="Texture", type="IMAGE")
            texture.image = bpy.data.images.load(str(img_path))
            # Applying displacement
            modifier = mesh.modifiers.new(name="Displace", type="DISPLACE")
            modifier.texture = texture
            modifier.texture_coords = "UV"
            modifier.uv_layer = "UVMap"
            modifier.direction = "NORMAL"
            modifier.strength = strength
            modifier.mid_level = mid_level
            bpy.ops.object.modifier_apply(modifier="Displace")
        # Save the final mesh
        if output_path is None:
            output_path = input_path.with_stem("final")
        else:
            output_path = Path(output_path).resolve()
        bpy.ops.wm.obj_export(
            filepath=str(output_path),
            export_materials=False
        )
