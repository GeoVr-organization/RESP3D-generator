from __future__ import annotations

import torch
from pytorch3d.ops import SubdivideMeshes, cot_laplacian
from pytorch3d.structures import Meshes
from torch import BoolTensor, LongTensor, Tensor
from torch.nn.functional import pad

from .common import wrong_shape_msg as ws_msg


def mesh_laplacian_smoothing2(meshes: Meshes, method: str = "uniform") -> Tensor:
    r"""
    Note: The squared version of pytorch3d.loss.mesh_laplacian_smoothing().
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing, namely
    with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").
    For more details read [1, 2].

    :param meshes: Meshes object with a batch of meshes.
    :type meshes: Meshes
    :param method: str specifying the method for the laplacian.
    :returns: Average laplacian smoothing loss across the batch.
        Returns 0 if `meshes` contains no meshes or all empty meshes.
    :rtype: Tensor

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of
    shape Mx3. The Laplacian matrix L is a NxN tensor such that LV gives
    a tensor of vectors: for a uniform Laplacian, LuV[i] points to the
    centroid of its neighboring vertices, a cotangent Laplacian LcV[i]
    is known to be an approximation of the surface normal, while the
    curvature variant LckV[i] scales the normals by the discrete mean
    curvature. For a vertex i, assume S[i] is the set of neighboring
    vertices to i, a_ij and b_ij are the "outside" angles in the two
    triangles connecting vertex v_i and its neighboring vertex v_j for
    j in S[i], as seen in the diagram below.

    .. Code-block:: python

               A_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

    The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
    For the uniform variant, w_ij = 1 / |S[i]|
    For the cotangent variant,
        w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
    For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
    where A[i] is the sum of the areas of all triangles containing
    vertex v_i.

    There is a nice trigonometry identity to compute cotangents.
    Consider a triangle with side lengths A, B, C and angles a, b, c.

    .. Code-block:: python

               C
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

    Then cot a = (B^2 + C^2 - A^2) / 4 * area
    We know that area = CH/2, and by the law of cosines we have

    A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

    Putting these together, we get:

    B^2 + C^2 - A^2   2BC cos a
    _______________ = _________ = (B/H) cos a = cos a / sin a = cot a
       4 * area         2CH


    [1] Desbrun et al., "Implicit fairing of irregular meshes using
    diffusion and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al., "Laplacian Mesh Optimization", Graphite 2006.
    """
    if meshes.isempty():
        return torch.zeros(
            1, dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don"t want to backprop through the computation of the Laplacian;
    # treat it as a magic constant matrix used to transform verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    else:
        raise ValueError("Method should be one of {uniform, cot, cotcurv}")
    loss = loss.norm(dim=1) ** 2  # CHANGED HERE

    loss = loss * weights
    return loss.sum() / N


class UVMaps:
    def __init__(
        self,
        uv_verts: Tensor | list[Tensor],
        uv_faces: LongTensor | list[LongTensor]
    ):
        if isinstance(uv_verts, Tensor):
            if uv_verts.shape[-1] != 2:
                raise ValueError(ws_msg(uv_verts, "uv_verts", "(B, N, 2)"))
            else:
                # Add a "third" dimension
                uv_verts = pad(uv_verts, pad=[0, 1])
        elif isinstance(uv_verts, list):
            if any(not isinstance(x, Tensor) or x.shape[-1] != 2 for x in uv_verts):
                raise ValueError("uv_verts must be list of Tensor of size (N, 2).")
            else:
                # Add a "third" dimension
                uv_verts = [pad(el, pad=[0, 1]) for el in uv_verts]
        else:
            raise TypeError("Unknown type for uv_verts")

        self._uv_mesh = Meshes(uv_verts, uv_faces)

    def __len__(self) -> int:
        return self._uv_mesh.__len__()

    def __getitem__(self, index: int | list[int] | slice | BoolTensor | LongTensor) -> UVMaps:
        uv_mesh_subset = self._uv_mesh.__getitem__(index)
        verts_padded = uv_mesh_subset.verts_padded()
        faces_padded = uv_mesh_subset.faces_padded()
        return UVMaps(verts_padded[..., :2], faces_padded)

    def verts_list(self) -> list[Tensor]:
        r"""
        Get the list representation of the vertices.

        :returns: List of tensors of vertices of shape (V_n, 3).
        :rtype: list[Tensor]
        """
        return [v[..., :2] for v in self._uv_mesh.verts_list()]

    def verts_padded(self) -> Tensor:
        r"""
        Get the padded representation of the vertices.

        :returns: Tensor of vertices of shape (N, max(V_n), 2).
        :rtype: Tensor
        """
        return self._uv_mesh.verts_padded()[..., :2]

    def verts_packed(self) -> Tensor:
        r"""
        Get the packed representation of the vertices.

        :returns: Tensor of vertices of shape (sum(V_n), 2).
        :rtype: Tensor
        """
        return self._uv_mesh.verts_packed()[..., :2]

    def verts_packed_to_uvmap_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor with the same first dimension as verts_packed.
        verts_packed_to_uvmap_idx[i] gives the index of the UV-map which
        contains verts_packed[i].

        :returns: 1D tensor of indices.
        :rtype: LongTensor
        """
        return self._uv_mesh.verts_packed_to_mesh_idx()

    def uvmap_to_verts_packed_first_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        such that the first vertex of the ith UV-map is verts_packed[x[i]].

        :returns: 1D tensor of indices of first items.
        :rtype: LongTensor
        """
        return self._uv_mesh.mesh_to_verts_packed_first_idx()

    def num_verts_per_uvmap(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        giving the number of vertices in each mesh.

        :returns: 1D tensor of sizes.
        :rtype: LongTensor
        """
        return self._uv_mesh.num_verts_per_mesh()

    def faces_list(self) -> list[LongTensor]:
        r"""
        Get the list representation of the vertices.

        :returns: List of tensors of vertices of shape (F_n, 3).
        :rtype: list[Tensor]
        """
        return self._uv_mesh.faces_list()

    def faces_padded(self) -> LongTensor:
        r"""
        Get the padded representation of the faces.

        :returns: Tensor of faces of shape (N, max(F_n), 3).
        :rtype:  LongTensor
        """
        return self._uv_mesh.faces_padded()

    def faces_packed(self) -> LongTensor:
        r"""
        Get the packed representation of the vertices.

        :returns: Tensor of vertices of shape (sum(F_n), 3).
        :rtype: LongTensor
        """
        return self._uv_mesh.faces_packed()

    def faces_packed_to_uvmap_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor with the same first dimension as faces_packed.
        faces_packed_to_uvmap_idx[i] gives the index of the UV-map which
        contains faces_packed[i].

        :returns: 1D tensor of indices.
        :rtype: LongTensor
        """
        return self._uv_mesh.faces_packed_to_mesh_idx()

    def uvmap_to_faces_packed_first_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        such that the first face of the ith UV-map is faces_packed[x[i]].

        :returns: 1D tensor of indices of first items.
        :rtype: LongTensor
        """
        return self._uv_mesh.mesh_to_faces_packed_first_idx()

    def num_faces_per_uvamp(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        giving the number of faces in each UV-map.

        :returns: 1D tensor of sizes.
        :rtype: LongTensor
        """
        return self._uv_mesh.num_faces_per_mesh()

    def edges_packed(self) -> LongTensor:
        r"""
        Get the packed representation of the edges.

        :returns: Tensor of vertices of shape (sum(E_n), 2).
        :rtype: LongTensor
        """
        return self._uv_mesh.edges_packed()

    def edges_packed_to_uvmap_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor with the same first dimension as edges_packed.
        edges_packed_to_uvmap_idx[i] gives the index of the UV-map which
        contains edges_packed[i].

        :returns: 1D tensor of indices.
        :rtype: LongTensor
        """
        return self._uv_mesh.edges_packed_to_mesh_idx()

    def uvmap_to_edges_packed_first_idx(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        such that the first edge of the ith UV-map is edges_packed[x[i]].

        :returns: 1D tensor of indices of first items.
        :rtype: LongTensor
        """
        return self._uv_mesh.mesh_to_faces_packed_first_idx()

    def num_edges_per_uvmap(self) -> LongTensor:
        r"""
        Return a 1D tensor x with length equal to the number of UV-maps
        giving the number of edges in each UV-map.

        :returns: 1D tensor of sizes.
        :rtype: LongTensor
        """
        return self._uv_mesh.num_edges_per_mesh()

    def faces_packed_to_edges_packed(self) -> LongTensor:
        r"""
        Get the packed representation of the faces in terms of edges.
        Faces are given by the indices of the three edges in the packed
        representation of the edges.

        :returns: Tensor of faces of shape (sum(F_n), 3).
        :rtype: LongTensor
        """
        return self._uv_mesh.faces_packed_to_edges_packed()

    def faces_area_packed(self) -> LongTensor:
        r"""
        Get the packed representation of the face areas.

        :returns: Tensor of areas of shape (sum(F_n),).
        :rtype: LongTensor
        """
        return self._uv_mesh.faces_areas_packed()

    def get_uvmap_verts_faces(self, index: int) -> tuple[Tensor, LongTensor]:
        r"""
        Get tensors for a single UV-map from the list representation.

        :param index: Index in the range [0, N).
        :type index: int
        :return: A tuple the vertices Tensor of shape (V, 2), and the
            faces LongTensor of shape (F, 3).
        :rtype: tuple[Tensor, LongTensor]
        """
        verts, faces = self._uv_mesh.get_mesh_verts_faces(index)
        return verts[..., :2], faces

    def to_meshes(self) -> Meshes:
        r"""
        Get a Meshes representation of the UV-map.

        :return: The correspondent Meshes representation.
        :rtype: Meshes
        """
        return self._uv_mesh.clone()


class SubdivideMeshesUV:
    r"""
    This class provides a simple tool to subdivide the faces of the
    meshes and their respective uv-maps. For faces/verts, it supports
    only Tensor containing a single mesh or a batch of meshes (of the
    same size). Mesh and UV-map will be treated independently, no
    checks will be attempted on their indices/data.
    """

    def __init__(
        self,
        verts: Tensor,
        faces: LongTensor,
        uv_verts: Tensor,
        uv_faces: LongTensor
    ):
        if verts.shape[0] != uv_verts.shape[0] or faces.shape[0] != uv_faces.shape[0]:
            raise ValueError("Incompatible batch sizes!")
        self.meshes = Meshes(verts, faces)
        self.uvmaps = UVMaps(uv_verts, uv_faces)

    def subdivide(self, n: int = 1) -> tuple[Meshes, UVMaps]:
        r"""
        Subdivide the faces of the meshes and their uv-maps (N times).

        :param n: Number of subdivisions. Default to 1.
        :type n: int, optional
        :return: A tuple contained the subdivided mesh and its UV-map.
        :rtype: tuple[Meshes, UVMaps]
        """
        meshes, uvmaps = self.meshes, self.uvmaps
        for _ in range(n):
            uvmeshes = uvmaps.to_meshes()
            # subdivide the meshes and their uv-maps
            meshes = SubdivideMeshes(meshes).subdivide_homogeneous(meshes)
            uvmeshes = SubdivideMeshes(uvmeshes).subdivide_homogeneous(uvmeshes)
            uvmaps = UVMaps(uvmeshes.verts_padded()[..., :2], uvmeshes.faces_padded())
        return meshes, uvmaps
