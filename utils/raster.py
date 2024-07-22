import cv2
import numpy as np
import torch
from numpy import ndarray
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes
from torch import BoolTensor, LongTensor, Tensor
from torch.nn.functional import avg_pool2d

from .common import wrong_shape_msg as ws_msg

simple_rasterization_settings = {
    "image_size": (256, 256),
    "blur_radius": 0.0,
    "faces_per_pixel": 1,
    "bin_size": None,
    "max_faces_per_bin": None,
    "perspective_correct": False
}


def prepare_to_pytorch3d(points: Tensor, tz: float = 10.0) -> Tensor:
    r"""
    Convert spatial coordinates from the opengl axis convention
    (X-right, Y-up, Z-back), to the PyTorch3D convention (X-left, Y-up,
    Z-forward), then shift everything along the Z-axis.

    :param points: The input points from an opengl reference system. Its
        shape must be (N, 3).
    :type points:  Tensor
    :param tz: The amount of translation along the Z-axis. Defaults to 10.0.
    :type tz: float, optional
    :return: The corresponding points in a PyTorch3D reference system.
    :rtype: Tensor

    For more details, see https://pytorch3d.org/docs/renderer_getting_started
    """
    if points.ndim != 2 and points.shape[1] != 3:
        raise ValueError(ws_msg(points, "points", "(N, 3)"))
    # Flip x,z-axes
    points[:, [0, 2]] *= -1
    points[:, 2] += tz
    return points


def masked_avg_pool2d(
    img: Tensor,
    mask: BoolTensor,
    kernel_size: int
) -> tuple[Tensor, Tensor]:
    r"""
    A masked version of the avg_pool_2D function. In this version, the
    average takes into account only valid values within the kernel
    (i.e., filtered by the mask). The input is a single grayscale image.
    To get the same shape as the input, the stride value is set to 1 and
    the zero-padding is half the size of the kernel.

    :param img: The input image. Its shape must be (H, W).
    :type img: Tensor
    :param mask: The input binary mask. Its shape must be identical to
        the zbuff parameter.
    :type mask: BoolTensor
    :param kernel_size: Size of the squared kernel. It must be an odd
        value.
    :type kernel_size: int
    :return: A tuple containing, the avg-image and a reliability value
        for each pixel of the avg (number of valid values within the
        kernel wrt its area).
    :rtype: tuple[Tensor, Tensor]
    """
    if img.ndim != 2:
        raise ValueError(ws_msg(img, "img", "(N, M)"))
    if mask.ndim != 2:
        raise ValueError(ws_msg(mask, "mask", "(N, M)"))
    if img.shape != mask.shape:
        raise ValueError(
            f"img and mask should have the same shape, but the current"
            f" shapes are {tuple(img.shape)} and {tuple(mask.shape)}"
            f" respectively.")

    # Remove values outside the mask
    img = img.masked_fill(~mask, 0)
    # "sum" pool layer settings
    settings = {
        "kernel_size": kernel_size,
        "padding": kernel_size // 2,
        "stride": 1,
        "divisor_override": 1,
    }
    # Get the local sums of valid values
    img_sums = avg_pool2d(img[None, None], **settings).squeeze()
    # Get the local count of valid values
    img_counts = avg_pool2d(mask[None, None].float(), **settings).squeeze()
    # Compute the average and clean values outside the mask
    avg = (img_sums / img_counts).masked_fill(~mask, 0)
    reliability = (img_counts / kernel_size ** 2).masked_fill(~mask, 0)
    return avg, reliability


def pix_to_uv(
    verts: Tensor,
    faces: LongTensor,
    uv_verts: Tensor,
    uv_faces: LongTensor,
    *,
    image_size: int | tuple[int, int] = 256
) -> tuple[Tensor, Tensor]:
    r"""
    Rasterize a mesh and retrieve, for each pixel, the correspondent UV
    texture coordinates, where they are available.

    :param verts: Vertices of the mesh. Its shape must be (V, 3).
    :type verts: Tensor
    :param faces: Faces of the mesh. Its shape must be (F, 3).
    :type faces: LongTensor
    :param uv_verts: UV coordinates. Its shape must be (T, 2).
    :type uv_verts: Tensor
    :param uv_faces: UV faces. Its shape must be (F, 3).
    :type uv_faces: LongTensor
    :param image_size: The size of the desired image. Defaults to 256.
    :type image_size: int | tuple[int, int], optional
    :return: A tuple containing the image with the uv coordinates for
        each pixel, and a boolean mask which filter the pixels that
        haven"t any uv coordinates.
    :rtype: tuple[Tensor, BoolTensor]
    """
    mesh = Meshes(verts[None], faces[None])
    settings = {
        **simple_rasterization_settings,
        "image_size": image_size
    }
    pix_to_face, _, bary, _ = rasterize_meshes(mesh, **settings)  # (1, h, w, 1), (1, h, w, 1, 3)
    pix_to_face, bary = pix_to_face.squeeze(), bary.squeeze()  # (h, w), (h, w, 3)
    # Visibility mask
    visibility = pix_to_face > -1  # (h, w)
    pix_to_face.masked_fill_(~visibility, 0)
    h, w = pix_to_face.shape
    idx = pix_to_face.view(h * w, 1, 1).expand(h * w, 3, 2)
    # For each face retrieve uv vertex coords
    pix_to_face_uvcoords = uv_verts[uv_faces].gather(dim=0, index=idx).view(h, w, 3, 2)
    # Interpolate the information about the vertices of the faces to
    # achieve information about specific *points* on the mesh
    pixel_to_uvcoords = (bary[..., None] * pix_to_face_uvcoords).sum(dim=-2)  # (h, w, 2)
    mask = ~visibility[..., None].expand(*pixel_to_uvcoords.shape)
    pixel_to_uvcoords = pixel_to_uvcoords.masked_fill(mask, -1)
    return pixel_to_uvcoords, visibility


def iqr_remove(img: Tensor, *, radius: int = 3, mask: BoolTensor = None) -> Tensor:
    r"""
    Use IQR score to remove outliers from an input image. The outlier
    values will be replaced by the averages of their surroundings.

    :param img: The input image. Its shape must be (H, W).
    :type img: Tensor
    :param radius: The radius of the avg window. Defaults to 3.
    :type radius: int, optional
    :param mask: Mask of the interested pixels. Its shape must be the
        same as `img`. If not mask is given, the whole image will be
        taken into account.
    :type mask: BoolTensor, optional
    :return: The input image without outliers.
    :rtype: Tensor

    For more details, see https://en.wikipedia.org/wiki/Interquartile_range
    """
    if img.ndim != 2:
        raise ValueError(ws_msg(img, "img", "(N, M)"))
    if mask is None:
        # Use the entire image
        mask = torch.ones_like(img, dtype=torch.bool)
    elif mask.shape != img.shape:
        raise ValueError(ws_msg(mask, "mask", img.shape))
    img_masked = img.masked_fill(~mask, 0)
    # Find anomalies within the area of the mask
    q1, q3 = img[mask].quantile(0.25), img[mask].quantile(0.75)
    iqr = q3 - q1
    outliers_mask = (img < q1 - 1.5 * iqr) | (img > q3 + 1.5 * iqr)
    # For each pixel of the image compute the avg of its valid
    # surroundings (center excluded)
    settings = {
        "kernel_size": radius * 2 + 1,
        "padding": radius,
        "stride": 1,
        "divisor_override": 1,
    }
    sums = avg_pool2d(img_masked[None, None], **settings).squeeze()
    counts = avg_pool2d(mask[None, None].float(), **settings).squeeze()
    avg = (sums - img_masked) / (counts - 1)
    # Remove outliers from the original image
    # noinspection PyTypeChecker
    return torch.where(mask & outliers_mask, avg, img)


def smooth_borders(
    img: ndarray,
    *,
    ksize: tuple[int, int],
    sigma: float,
    steps: int = 3,
) -> ndarray:
    r"""
    Smooth the borders of a displacement map. 0 values are assumed
    background.

    :param img: The input image.
    :type img: ndarray
    :param ksize: Size of the Gaussian blurring filter. Must be
        a tuple of odd values.
    :type ksize: tuple[int, int], optional
    :param sigma: Sigma value of the Gaussian blurring filter. Must
        be odd. Defaults to 0 (esteemed automatically).
    :type sigma: float, optional
    :param steps: Number of smoothing steps. Defaults to 2.
    :type steps: int, optional
    :return: The processed image.
    :rtype: ndarray
    """
    dt = img.dtype
    if dt == np.uint8:
        img = img.astype(np.float64) / 255
    elif dt == np.float32 and np.all((0 <= img) & (img <= 1)):
        img = img.astype(np.float64)
    elif dt == np.float64 and np.all((0 <= img) & (img <= 1)):
        pass
    else:
        raise ValueError("Invalid image values")

    def blur(x: ndarray) -> ndarray:
        return cv2.GaussianBlur(x, ksize, sigma, cv2.BORDER_CONSTANT)

    for _ in range(steps):
        # Mask of drawn values
        mask = (img > 0).astype(np.float64)
        # Mask of not blurred values
        mask = np.isclose(blur(mask), 1.0)
        # Blur borders only
        img = np.where(mask, img, blur(img))
    img = img.clip(0, 1)
    if dt == np.uint8:
        img *= 255
    return img.astype(dt)
