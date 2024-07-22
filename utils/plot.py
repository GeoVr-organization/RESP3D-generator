from typing import Any

import vedo
from torch import Tensor
from vedo import Line, Mesh, Points, Text2D
from vedo.pyplot import plot

from .common import wrong_shape_msg as ws_msg


def create_mesh(
    vertices: Tensor,
    faces: Tensor,
    *,
    c: Any = "white",
    alpha: float = 1.0,
    **kwargs
) -> Mesh:
    r"""
    Generate a Vedo Mesh from given vertices and faces.

    :param vertices: The vertices of the mesh. Its shape must be (N, 3).
    :type vertices: Tensor
    :param faces: The faces of the mesh. Its shape must be (M, 3).
    :type faces: Tensor
    :param c: Face color. RGB format, hex, symbol or name. Defaults to
        "white".
    :param alpha: Alpha coefficient. Defaults to 1.0.
    :type alpha: float, optional
    :param kwargs: Other Vedo Mesh parameters.
    :return: The correspondent Vedo Mesh.
    :rtype: Mesh
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(ws_msg(vertices, "vertices", "(N, 3)"))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(ws_msg(faces, "faces", "(M, 3)"))
    vertices, faces = vertices.numpy(force=True), faces.numpy(force=True)
    mesh = Mesh([vertices, faces], c=c, alpha=alpha)
    mesh.backcolor(kwargs.get("bc"))
    mesh.linecolor(kwargs.get("lc"))
    mesh.linewidth(kwargs.get("lw"))
    return mesh


def create_points(
    points: Tensor,
    *,
    scale: float | Tensor = 1.0,
    **kwargs
) -> Points:
    r"""
    Create a Vedo Points mesh for the given input points.

    :param points: The input (set of) point(s). Its shape must be
        (..., 3).
    :type points: Tensor
    :param scale:  Relative scale of the points. Defaults to 1.0.
    :type scale: float | Tensor, optional
    :return: The corresponding Vedo Points mesh representing the input
        set of points.
    :rtype: Points
    """
    if points.shape[-1] != 3:
        raise ValueError(ws_msg(points, "points", "(..., 3)"))
    scale = scale.item() if isinstance(scale, Tensor) else scale
    points = points.view(-1, 3)
    kwargs = {"r": scale * 0.1, **kwargs}
    return Points(points.numpy(force=True), **kwargs)


def create_line(
    points: Tensor,
    *,
    c: Any = "red",
    **kwargs
) -> Line:
    r"""
    Create a Vedo Line mesh which connects the given input points.

    :param points: The input ordered set of points.
    :type points: Tensor
    :param c: Line color. Defaults to "red".
    :param kwargs: Other Vedo Line parameters.
    :return: Correspondent Vedo Line mesh connecting the given points.
    :rtype: Line
    """

    return Line(points.numpy(force=True), c=c, **kwargs)


def plot_objects(
    *actors: Any,
    ortho: bool = True,
    axes: int = 8,
    bg: Any = "#555555",
    viewup: str = "y",
    size: tuple[int, int] = (800, 800),
    msg: str = "Close to continue",
    **kwargs
):
    r"""
    Plot Vedo objects.

    :param actors: The Vedo actors objects to be plotted.
    :type actors: Mesh
    :param ortho: Use orthographic view. Defaults to True.
    :type ortho: bool, optional
    :param axes: Vedo Plotter axes style. Defaults to 8.
    :param bg: Plot background color. This can be RGB format, hex,
        symbol or a color name. Defaults to "#555555".
    :type bg: Any, optional
    :param viewup: Camera up-vector. This can be "x", "y" or "z".
        Defaults to "y".
    :type viewup: str, optional
    :param size: Plot window size. Defaults to (800, 800).
    :type size: tuple[int, int]
    :param msg: Plot message. Defaults to "Close to continue".
    :type msg: str, optional
    :param kwargs: Vedo Plotter parameters.
    """
    (
        vedo.Plotter(axes=axes, bg=bg, **kwargs)
        .parallel_projection(ortho)
        .look_at("xy")
        .add(actors, Text2D(msg, pos="top-mid"))
        .show(viewup=viewup, size=size)
        .close()
    )


def plot_values(
    x: Tensor,
    y: Tensor,
    *,
    size: tuple[int, int] = (900, 650),
    fmt: str = "-r",
    lw: int = 1,
    **kwargs
):
    r"""
    Plot a list of values.

    :param x: The x-axis values. The shape must be (N,).
    :type x: Tensor
    :param y: The y-axis values. The shape must be (N,).
    :type y: Tensor
    :param size: Plot window size. Defaults to (900, 650).
    :type size: tuple[int, int], optional
    :param fmt: Line format. Defaults to "-r" (red solid line).
    :type fmt: str, optional
    :param lw: The line width in pts. Defaults to 1.
    :type lw: int, optional
    :param kwargs: Other Vedo Plotter parameters.
    """
    x, y = x.numpy(force=True), y.numpy(force=True)
    (
        plot(x, y, fmt, lw=lw, **kwargs)
        .show(size=size, zoom="tight")
        .close()
    )
