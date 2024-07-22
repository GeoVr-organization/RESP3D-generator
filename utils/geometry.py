import torch
from pytorch3d.structures import Meshes
from torch import LongTensor, Tensor
from torch.nn.functional import normalize, pad

from .common import torch_var_opts, wrong_shape_msg as ws_msg

# apply_displacement precomputed vars
_vn = None
_verts_on_uv = None


def transform_mat(R: Tensor, T: Tensor) -> Tensor:
    r"""
    Creates a batch of affine transformation matrices from batches
    of rotations and translations.

    :param R: Batch of rotation matrices. Its shape must be (..., 3, 3).
    :type R: Tensor
    :param T: Batch of translation vectors. Its shape must be
        (..., 3, 1).
    :type T: Tensor
    :return: The correspondent batch of transformation matrices.
    :rtype: Tensor
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(ws_msg(R, "R", "(..., 3, 3)"))
    if T.shape[-2:] != (3, 1):
        raise ValueError(ws_msg(T, "T", "(..., 3, 1)"))
    if R.shape[:-2] != T.shape[:-2]:
        raise ValueError(
            f"R and t must have the same batch shape, but the current "
            f"shapes are {tuple(R.shape)} and {tuple(T.shape)} "
            f"respectively."
        )

    # Pad R on the bottom with zeros
    R_padded = pad(R, pad=[0, 0, 0, 1])
    # Pad T on the bottom with a single one
    t_padded = pad(T, pad=[0, 0, 0, 1], value=1.0)
    return torch.cat([R_padded, t_padded], dim=-1)


def project_on_plane(points: Tensor, pos: Tensor, normal: Tensor) -> Tensor:
    r"""
    Project a set of points onto a given plane.

    :param points: The points that must be projected. Its shape must be
        (N, 3).
    :type points: Tensor
    :param pos: A "position" of the plane (a point that belongs to it).
    :type pos: Tensor
    :param normal: Plane normal.
    :type normal: Tensor
    :return: The projection of the points onto the given plane.
    :rtype: Tensor

    For more details, see: https://stackoverflow.com/a/9605695
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(ws_msg(points, "points", "(N, 3)"))
    if pos.shape != (3,):
        raise ValueError(ws_msg(pos, "pos", "(3,)"))
    if normal.shape != (3,):
        raise ValueError(ws_msg(normal, "normal", "(3,)"))

    n = normalize(normal, dim=0).expand(points.shape[0], 3)
    # Projections are give by: p_proj[i] = p[i] + ((c - p[i]) dot n) * n
    return points + torch.linalg.vecdot((pos - points), n)[:, None] * n


def find_plane(pts: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Find plane 'position' and normal from a given set of points.

    :param pts: The input set of points. Its shape must be (M, 3), with
        M >= 3.
    :type pts: Tensor
    :return: Plane "position" and normal.
    :rtype: tuple[Tensor, Tensor]
    """
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        raise ValueError(ws_msg(pts, "pts", "(N, 3)", "N >= 3"))

    position = pts.mean(dim=0)
    centered_pts = pts - position
    U, _, _ = torch.linalg.svd(centered_pts.T)
    normal = normalize(U[:, -1], dim=0)
    return position, normal


def umeyama(points: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform Kabsch-Umeyama algorithm: given two paired sets of :math:`N`
    points, :math:`\mathbf{X}` and :math:`\mathbf{Y}`, this function
    computes the scaling, rotation, and translation values suche that
    they define the rigid transformation which minimizes the mean
    squared error:

    .. math:: \mathit{MSE}(s, \mathbf{R}, \mathbf{t}) =
        \frac{1}{N}\sum_{i=0}^{N} \Vert \mathbf{Y}_i -
        (s\mathbf{X}_i\mathbf{R} - \mathbf{t}) \Vert^2

    :param points: The initial points set. Its shape must be (N, M).
    :type points: Tensor
    :param targets: The target points set. Its shape must be (N, M).
    :type targets: Tensor
    :return: A tuple of tensors, (t, R, s), containing the translation
        vector, the rotation matrix and the scaling factor which
        minimize the mean squared error.
    :rtype: tuple[Tensor, Tensor, Tensor]

    For more details, see: https://web.stanford.edu/class/cs273/refs/umeyama.pdf
    """
    if points.ndim != 2:
        raise ValueError(ws_msg(points, "points", "(N, M)"))
    if targets.ndim != 2:
        raise ValueError(ws_msg(targets, "targets", "(N, M)"))
    if points.shape != targets.shape:
        raise ValueError(
            f"points and targets should have the same shape, but the "
            f"current shapes are {tuple(points.shape)} and "
            f"{tuple(targets.shape)} respectively."
        )
    n = points.shape[0]
    mu_p, mu_t = points.mean(dim=0), targets.mean(dim=0)
    centered_p, centered_t = points - mu_p, targets - mu_t
    C = torch.mm(centered_p.T, centered_t) / n
    V, S, W = torch.linalg.svd(C)
    if torch.linalg.det(V) * torch.linalg.det(W) < 0.0:
        S[-1] *= -1
        V[:, -1] *= -1
    # By default, the Bessel correction for the variance is active, so
    # we need to disable it.
    s = torch.sum(S) / points.var(dim=0, **torch_var_opts).sum()
    R = torch.mm(V, W)
    t = mu_t - s * mu_p @ R
    return t, R, s


def wrap_curve_xy(
    polyline: Tensor,
    pl_normals: Tensor,
    points: Tensor
) -> Tensor:
    r"""
    Given a polyline and its points normals, re-project internal point
    on the other side. This procedure is done only wrt the XY plane.

    :param polyline: The input polyline, expressed as a set of points.
        Its shape must be (N, D).
    :type polyline: Tensor
    :param pl_normals: The normals of the polyline edges. Its shape must
        be (N, D).
    :type pl_normals:
    :param points: The set of points to be transformed. Its shape must
        be (M, D).
    :type points: Tensor
    :return: The transformed points.
    :rtype: Tensor
    """
    n, m, d = polyline.shape[0], points.shape[0], points.shape[1]

    ## Points-curve's vertices distances
    curve_rep = polyline[:, None].expand(n, m, d)
    points_rep = points[None].expand(n, m, d)
    dist_pv = torch.linalg.vector_norm(curve_rep - points_rep, dim=-1)  # (n, m)

    ## Points-curve's edges distances
    starts, ends = polyline[:-1], polyline[1:]
    edge_vecs = ends - starts  # (n - 1, d)
    e = edge_vecs.shape[0]  # e = n - 1
    # Edge lengths (maximum limit for the inner products)
    edge_lengths = torch.linalg.vector_norm(edge_vecs, dim=-1, keepdim=True)  # (e, 1)
    # Edge start-to-end directions (unit vectors)
    edge_dirs = edge_vecs / edge_lengths  # (e, d)
    edge_dirs = edge_dirs[:, None].expand(e, m, d)
    # points projection distances wrt edge's starts
    points_rep = points[None].expand(e, m, d)
    starts_rep = starts[:, None].expand(e, m, d)
    inners = torch.linalg.vecdot(points_rep - starts_rep, edge_dirs)  # (e, m)
    # Project every lmk on each edge and compute the distances
    points_proj = starts_rep + inners[..., None] * edge_dirs  # (e, m, d)
    dist_pe = torch.linalg.vector_norm(points_rep - points_proj, dim=-1)  # (e, m)
    # filter invalid distances (projections outside the edges bounds)
    valid = (0 < inners) & (inners < edge_lengths)  # (e, m)
    dist_pe = dist_pe.masked_fill(~valid, torch.inf)

    ## Closest projections of the points
    indices = torch.cat([dist_pv, dist_pe]).min(dim=0).indices
    indices_rep = indices[None, :, None].expand(1, m, 3)  # (1, M, 3)
    projections = (
        torch.cat([curve_rep, points_proj]).gather(dim=0, index=indices_rep).squeeze()
    )  # (M, 3)
    # projection-to-point vectors (XY only)
    proj_to_point = points - projections  # (M, 3)
    # Use the normals of the curve nodes to verify if lmks are inside the curve
    outer = torch.linalg.cross(pl_normals[:-1], edge_dirs[:, 0])
    edge_normals = normalize(torch.linalg.cross(edge_dirs[:, 0], outer), dim=-1)
    pl_normals = torch.cat([pl_normals, edge_normals], dim=0)
    is_inside = torch.linalg.vecdot(
        pl_normals[indices, :2], proj_to_point[:, :2]
    ) < 0
    # Invert the directions
    points[is_inside, :2] = projections[is_inside, :2] - proj_to_point[is_inside, :2]
    return points


# FIXME: works but returns a rough surface
def apply_displacement(
    v: Tensor,
    f: LongTensor,
    uv_v: Tensor,
    uv_f: LongTensor,
    texture: Tensor,
    *,
    midlevel: float = 0.5,
    strength: float = 1.0,
    use_precomputed: bool = False
) -> Tensor:
    r"""
    Apply a displacement map to the surface.

    :param v: Mesh vertices. Its shape must be (V, 3).
    :type v: Tensor
    :param f: Mesh faces. Its shape must be (F, 3).
    :type f: LongTensor
    :param uv_v: UV coordinates. Its shape must be (T, 2).
    :type uv_v: Tensor
    :param uv_f: UV faces. Its shape must be (F, 3).
    :type uv_f: LongTensor
    :param texture: The displacement map (texture). Its shape must be (H, W).
    :type texture: Tensor
    :param midlevel: The texture value which will be treated as no
        displacement by the modifier. Texture values below this
        threshold will result in negative displacement along the
        selected direction, while texture values above it will result in
        positive displacement. Its value must be between 0 and 1.
    :type midlevel:
    :param strength: The strength of the displacement. After offsetting
        by the `midlevel` value, the displacement will be multiplied by
        the strength value to give the final vertex offset. A negative
        strength can be used to invert the effect of the modifier.
    :type strength: float
    :param use_precomputed: If the same mesh is provided, enable this to
        use previously computed normals and vertices mapping to avoid
        waste of computation and speed up the processing.
    :return: The displaced vertices.
    :rtype: Tensor
    """
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(ws_msg(v, "v", "(V, 3)"))
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(ws_msg(f, "f", "(F, 3)"))
    if uv_v.ndim != 2 or uv_v.shape[1] != 2:
        raise ValueError(ws_msg(uv_v, "uv_v", "(T, 2)"))
    if uv_f.ndim != 2 or uv_f.shape[1] != 3:
        raise ValueError(ws_msg(uv_f, "uv_f", "(F, 3)"))
    if f.shape[0] != uv_f.shape[0]:
        raise ValueError("f and uv_f must have the same number of elements.")

    global _vn, _verts_on_uv
    if not use_precomputed or _vn is None or _verts_on_uv is None:
        # Compute verts normals
        _vn = Meshes(v[None], f[None]).verts_normals_packed()
        # Mapping the vertices of the mesh onto the texture uniquely.
        uv_faces_verts = uv_v[uv_f]  # (F, 3, 2)
        _, idx, counts = torch.unique(
            f.ravel(),
            sorted=True,
            return_inverse=True,
            return_counts=True
        )
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(dim=0)
        cum_sum = pad(cum_sum[:-1], pad=[1, 0])
        first_indices = ind_sorted[cum_sum]
        filtered_f = f.ravel()[first_indices]
        filtered_uv_faces_verts = uv_faces_verts.view(-1, 2)[first_indices]
        _verts_on_uv = torch.zeros(v.shape[0], 2).to(v)
        _verts_on_uv[filtered_f] = filtered_uv_faces_verts
        h, w = texture.shape
        _verts_on_uv[:, 1] = 1 - _verts_on_uv[:, 1]
        _verts_on_uv *= torch.tensor([w - 1, h - 1]).to(_verts_on_uv)
        _verts_on_uv = _verts_on_uv.long()

    texture_vals = texture[_verts_on_uv[:, 1], _verts_on_uv[:, 0]]  # (V,)
    displacements = strength * (texture_vals - midlevel)  # (V,)

    # Apply displacements to the vertices
    v = v + _vn * displacements[:, None]
    return v
