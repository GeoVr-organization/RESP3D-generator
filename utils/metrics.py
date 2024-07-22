import torch
from torch import Tensor

from .common import wrong_shape_msg


def euclidean_dist(
    points: Tensor,
    targets: Tensor,
    *,
    use_mean: bool = True,
) -> Tensor:
    r"""
    Compute the sum/mean of the squared distances between two paired
    sets of points.

    :param points: The first set of points. Its shape must be (N, D).
    :type points: Tensor
    :param targets: The second set of points. Its shape must be (N, D).
    :type targets: Tensor
    :param use_mean: Return the mean instead of sum. Defaults to True.
    :type use_mean: bool, optional
    :return: The sum/mean of the Euclidean distances.
    :rtype: Tensor
    """
    if points.ndim != 2:
        raise ValueError(wrong_shape_msg(points, "points", "(N, D)"))
    if targets.ndim != 2:
        raise ValueError(wrong_shape_msg(targets, "targets", "(N, D)"))
    if points.shape != targets.shape:
        raise ValueError(
            f"points and targets must have the same shape, but their "
            f"current shapes are {points.shape} and {targets.shape} "
            f"respectively."
        )
    result = torch.pairwise_distance(points, targets, eps=0.0)
    return (result ** 2).mean() if use_mean else (result ** 2).sum()


def adaptive_distance(
    curve: Tensor,
    points: Tensor,
    *,
    atol: float | Tensor = 0.0,
    use_mean: bool = True
) -> Tensor:
    r"""
    Compute the mean/sum of the shortest-squared distances between a set
    of points and a polyline (represented as an ordered set of points).

    :param curve: The ordered set of points representing the target
        polyline. Its shape must be (M, D).
    :type curve: Tensor
    :param points: The set of points. Its shape must be (N, D).
    :type points: Tensor
    :param atol: Absolute tolerance - the minimum value below which the
        point-curve distance is considered 0. Defaults to 0.0 (no
        tolerance).
    :type atol: float | Tensor, optional
    :param use_mean: Return the mean instead of summing the distances.
        Defaults to True.
    :type use_mean: bool, optional
    :return: The mean/sum of the minimum squared distances between the
        points and the line.
    :rtype: Tensor
    """
    if curve.ndim != 2:
        raise ValueError(wrong_shape_msg(curve, "curve", "(N, D)"))
    if points.ndim != 2:
        raise ValueError(wrong_shape_msg(points, "points", "(M, D)"))
    if curve.shape[1] != points.shape[1]:
        raise ValueError(
            f"curve and points should have the same (last) dimension D, "
            f"but their current shape are {curve.shape[1]} and "
            f"{points.shape[1]} respectively."
        )

    n, m, d = curve.shape[0], points.shape[0], points.shape[1]

    ## Points-curve's vertices distances
    curve_rep = curve[:, None].expand(n, m, d)
    points_rep = points[None].expand(n, m, d)
    dist_pv = torch.linalg.vector_norm(curve_rep - points_rep, dim=-1)  # (n, m)

    ## Points-curve's edges distances
    starts, ends = curve[:-1], curve[1:]
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
    # Filter invalid distances (projections outside the edges bounds)
    valid = (0 < inners) & (inners < edge_lengths)  # (e, m)
    dist_pe = dist_pe.masked_fill(~valid, torch.inf)

    ## Minimum distances
    min_dists = torch.cat([dist_pv, dist_pe]).min(dim=0).values
    result = (min_dists - atol).clamp(min=0)
    return (result ** 2).mean() if use_mean else (result ** 2).sum()
