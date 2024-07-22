from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch import LongTensor, Tensor
from torch.nn import Module
from torch.nn.functional import normalize, pad

from .common import wrong_shape_msg as ws_msg
from .geometry import transform_mat

_flame_data = Path(__file__).parent.resolve() / "flame_binaries"
features = torch.load(_flame_data / "features.pt")
beard_uv_mask = torch.load(_flame_data / "beard_uv_mask.pt")


class Flame(Module):
    r"""
    Given some Flame parameters, this module provides a differentiable
    function that outputs the Flame mesh and its posed joints.
    """

    def __init__(self, gender: str = "male"):
        r"""
        Create a Flame module.

        :param gender: The gender of the flame model. Can be "male" or
            "female". Defaults to male.
        :type gender: str, optional
        """
        super(Flame, self).__init__()
        if gender in ("male", "female"):
            model_path = _flame_data / f"{gender}_model.pt"
        else:
            raise ValueError(f"Gender must be male or female. Got {gender}")

        # Load Flame data
        flame_model = torch.load(model_path)
        for key in flame_model:
            self.register_buffer(key, flame_model[key])

        # Fixed parameters
        self.register_buffer("fixed_beta", torch.zeros(300))
        self.register_buffer("fixed_psi", torch.zeros(100))
        self.register_buffer("fixed_theta", torch.zeros(15))
        self.register_buffer("fixed_t", torch.zeros(3))
        self.register_buffer("fixed_s", torch.tensor(1.0))

    def forward(
        self,
        beta: Tensor = None,
        psi: Tensor = None,
        theta: Tensor = None,
        s: Tensor = None,
        t: Tensor = None
    ) -> tuple[Tensor, Tensor]:
        r"""
        Apply Linear Blend Skinning and global transformations.

        :param beta: Shape blendshape coefficients. Its shape can span
            from (0,) to (300,).
        :type beta: Tensor, optional
        :param psi: Expression blendshape coefficients. Its shape can
            span from (0,) to (100,).
        :type psi: Tensor, optional
        :param theta: Pose blendshape coefficients. Its shape can span
            from (0,) to (15,).
        :type theta: Tensor, optional
        :param t: Global translation. Its shape must be (3,).
        :type t: Tensor, optional
        :param s: Global scale factor. It must be a scalar.
        :type s: Tensor, optional
        :return: A tuple containing the modeled vertices and the posed
            joints.
        :rtype: tuple[Tensor, Tensor]
        """
        beta = (
            pad(beta, pad=[0, 300 - beta.shape[-1]])
            if beta is not None
            else self.fixed_beta
        )
        if beta.shape[-1] != 300:
            raise ValueError(ws_msg(beta, "beta", "(..., 300)"))

        psi = (
            pad(psi, pad=[0, 100 - psi.shape[-1]])
            if psi is not None
            else self.fixed_psi
        )
        if psi.shape[-1] != 100:
            raise ValueError(ws_msg(psi, "psi", "(..., 100)"))

        theta = (
            pad(theta, pad=[0, 15 - theta.shape[-1]])
            if theta is not None
            else self.fixed_theta
        )
        if theta.shape[-1] != 15:
            raise ValueError(ws_msg(theta, "theta", "(..., 15)"))

        t = t if t is not None else self.fixed_t
        if t.shape[-1] != 3:
            raise ValueError(ws_msg(t, "t", "(..., 3)"))

        s = s if s is not None else self.fixed_s
        if not beta.shape[:-1] == psi.shape[:-1] == theta.shape[:-1] == t.shape[:-1] == s.shape:
            raise ValueError(
                "beta, psi, theta, t, s should have the same batch shape"
            )

        # Batch reshaping
        batch_shape = beta.shape[:-1]
        beta = beta.view(-1, 300)
        psi = psi.view(-1, 100)
        theta = theta.view(-1, 15)
        t = t.view(-1, 3)
        s = s.view(-1)

        # Create the final mesh through Linear Blend Skinning
        vertices, joints = self._lbs(beta=beta, psi=psi, theta=theta)

        # Applying global rescaling and translation
        vertices = (s * vertices + t).view(*batch_shape, -1, 3)
        joints = (s * joints + t).view(*batch_shape, -1, 3)

        return vertices, joints

    def _lbs(
        self,
        beta: Tensor,
        psi: Tensor,
        theta: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Perform Linear Blend Skinning:

        .. math:: M(\mathbf{\beta},\mathbf{\psi},\mathbf{\theta}) =
            W(T_p(\mathbf{\beta},\mathbf{\psi},\mathbf{\theta}), J(
            \mathbf{\beta}), \mathbf{\theta}, \mathbf{W})
        where

        .. math:: T_p(\mathbf{\beta},\mathbf{\psi},\mathbf{\theta}) =
            \mathbf{T} + B_s(\mathbf{\beta}; \mathbf{S}) +
            B_e(\mathbf{\psi}; \mathbf{E}) + B_p(\mathbf{\theta};
            \mathbf{P})

        :param beta: Shape blend-shape coefficients. Its shape must be
            (300,).
        :type beta: Tensor
        :param psi: Expression blend-shape coefficients. Its shape must
            be (100,).
        :type psi: Tensor
        :param theta: Pose blend-shape coefficients. Its shape must be
            (15,).
        :type theta: Tensor
        :return: A tuple containing the final transformed vertices of
            the mesh, and the new positions of the joints.
        :rtype: Tensor
        """
        if beta.ndim != 2 or beta.shape[-1] != 300:
            raise ValueError(ws_msg(beta, "beta", "(B, 300)"))
        if psi.ndim != 2 or psi.shape[-1] != 100:
            raise ValueError(ws_msg(psi, "psi", "(B, 100)"))
        if theta.ndim != 2 or theta.shape[-1] != 15:
            raise ValueError(ws_msg(theta, "theta", "(B, 15)"))
        if not beta.shape[0] == psi.shape[0] == theta.shape[0]:
            raise ValueError(
                "beta, psi, theta should have the same batch size"
            )

        b, n, k = beta.shape[0], self.T.shape[0], self.parents.shape[0]
        # Get rotation matrices for Theta and rest poses
        rot_mats = axis_angle_to_matrix(theta.view(b, k, 3))  # (b, k, 3, 3)
        I = torch.eye(3, device=theta.device).expand(b, k, 3, 3)
        theta = (rot_mats[:, 1:] - I[:, 1:]).view(b, 36)

        # Compute blendshape displacements
        Bs = (self.S * beta[:, None, None]).sum(dim=-1)     # (b, n, 3)
        Be = (self.E * psi[:, None, None]).sum(dim=-1)      # (b, n, 3)
        Bp = (self.P * theta[:, None, None]).sum(dim=-1)    # (b, n, 3)

        # Applying shape displacement to the template
        T_shaped = self.T + Bs  # (b, n, 3)

        # New global locations of the joints from shaped template
        joints = torch.matmul(self.J, T_shaped)  # (b, k, 3)

        # Applying expression and pose displacements
        Tp = T_shaped + Be + Bp  # (b, n, 3)

        # Apply rotations to the joints and get their new global locations
        joints_posed, rel_transforms = Flame._batch_rigid_transform(
            rot_mats=rot_mats, joints=joints, parents=self.parents
        )  # (b, k, 3), (b, k, 4, 4)

        # Applying linear blend skinning
        rel_transforms = rel_transforms.view(b, k, 16)
        vertices_transforms = torch.matmul(self.W, rel_transforms).view(b, n, 4, 4)
        Tp_homo = pad(Tp, pad=[0, 1, 0, 0, 0, 0], value=1.0)[..., None]  # (b, n, 4, 1)
        vertices_homo = torch.matmul(vertices_transforms, Tp_homo)  # (b, n, 4, 1)
        vertices = vertices_homo[:, :, :3, 0]  # (b, n, 3)

        return vertices, joints_posed

    @staticmethod
    def _batch_rigid_transform(
        rot_mats: Tensor,
        joints: Tensor,
        parents: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""
        Apply a chain of rigid transformations to the joints, along the
        given kinetic tree.

        :param rot_mats: The rotation matrices of the joints. Its shape
            must be (B, N, 3, 3).
        :type rot_mats: Tensor
        :param joints: Joints locations. Its shape must be (B, N, 3).
        :type joints: Tensor
        :param parents: The kinematic tree of each object (parent index
            of each joint). Its shape must be (N,).
        :type parents: Tensor
        :return: A tuple containing the new locations of the joints and
            their relative rigid transformations (wrt the rest pose).
        :rtype: tuple[Tensor, Tensor]
        """
        if rot_mats.ndim != 4 or rot_mats.shape[-2:] != (3, 3):
            raise ValueError(ws_msg(rot_mats, "rot_mats", "(B, N, 3, 3)"))
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(ws_msg(joints, "joints", "(B, N, 3)"))
        if parents.ndim != 1:
            raise ValueError(ws_msg(parents, "parents", "(N,)"))
        if not rot_mats.shape[1] == joints.shape[1] == parents.shape[0]:
            raise ValueError(
                "rot_mats, joints and parents sizes must match the "
                "number of joints N."
            )
        if not rot_mats.shape[0] == joints.shape[0]:
            raise ValueError(
                "rot_mats and joints must have the same batch size."
            )

        joints = joints[..., None]  # (b, n, 3, 1)
        # All joints are initially in rest pose (R = I). So, we can
        # easily compute each joint location wrt its parent
        rel_joints = joints.clone()  # (b, n, 3, 1)
        rel_joints[:, 1:] -= rel_joints[:, parents[1:]]  # (b, n, 3, 1)

        # Composing local transformation matrices (rot_mats represents
        # the local rotations of the joints)
        transforms_mat = transform_mat(rot_mats, rel_joints)  # (b, n, 4, 4)

        # Apply to each joint its transformation chain
        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            transform_chain.append(
                torch.bmm(
                    transform_chain[parents[i]], transforms_mat[:, i]
                )
            )
        transforms = torch.stack(transform_chain, dim=1)  # (b, n, 4, 4)

        # The last columns of the transformation matrices contain the
        # new global locations of the joints
        joints_posed = transforms[:, :, :3, 3]  # (b, n, 3)

        # Joint relative transformations from the rest pose. (see
        # https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf for
        # more details. This procedure is a shorthand for the eq. 3)
        joints_homo = pad(joints, pad=[0, 0, 0, 1])  # (b, n, 4, 1)
        rel_transforms = transforms - pad(
            torch.matmul(transforms, joints_homo), pad=[3, 0]
        )  # (b, n, 4, 4)

        return joints_posed, rel_transforms

    def move_neck(
        self,
        new_beta: Tensor,
        new_np: Tensor,
        s: Tensor = None,
        old_t: Tensor = None,
        old_beta: Tensor = None,
        old_gp: Tensor = None,
        old_np: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        The global pose of the neck joint is equal to:

        .. math:: G_1 = T_g S T_0 R_0 T_1 R_1

        where :math:`T_0` is the external translation (`old_t`),
        :math:`T_0, T_1` are the local translations of the global and
        neck joints (retrieved from `old_beta`), :math:`R_0, R_1`
        are the local rotations (`old_gp`, `old_np`) and S is the scale
        matrix (from the scale factor `s`). Since this, by varying the
        beta parameters and the local neck pose, it's possible to
        compute a new global pose and external translation such that
        :math:`G_1` remains fixed.

        :param new_beta: The new beta parameters to be applied to the
            model. Its shape must be between (0,) and (300,)
        :type new_beta: Tensor
        :param new_np: Axis-angle representation of the new local pose
            of the neck joint. Its shape must be (3,).
        :type new_np: Tensor
        :param s: The scale factor. It must be a scalar.
        :type s: Tensor, optional
        :param old_t: The old external translation. Its shape muse
            be (3,).
        :type old_t: Tensor, optional
        :param old_beta: The old beta parameters. Its shape must be
            between (0,) and (300,).
        :type old_beta: Tensor, optional
        :param old_gp: Axis-angle representation of the old local pose
            of the global joint. Its shape must be (3,).
        :type old_gp: Tensor, optional
        :param old_np: Axis-angle representation of the old local pose
            of the neck joint. Its shape must be (3,).
        :type old_np: Tensor, optional
        :return: A tuple containing the axis-angle representation of the
            new local pose of the global joint and the new external
            translation.
        :rtype: tuple[Tensor, Tensor]
        """
        new_beta = pad(new_beta, pad=[0, 300 - new_beta.shape[0]])
        if new_beta.shape != (300,):
            raise ValueError(ws_msg(new_beta, "new_beta", "(300,)"))
        new_np = pad(new_np, pad=[0, 3 - new_np.shape[0]])
        if new_np.shape != (3,):
            raise ValueError(ws_msg(new_np, "new_np", "(3,)"))
        s = s if s is not None else self.fixed_s
        if s.shape != tuple():
            raise ValueError(
                f"s should be a scalar, but its current shape is "
                f"{tuple(s.shape)}."
            )
        old_t = (
            pad(old_t, pad=[0, 3 - old_t.shape[0]])
            if old_t is not None
            else self.fixed_t[:3]
        )
        if old_t.shape != (3,):
            raise ValueError(ws_msg(old_t, "old_t", "(3,)"))
        old_beta = (
            pad(old_beta, pad=[0, 300 - old_beta.shape[0]])
            if old_beta is not None
            else self.fixed_beta
        )
        if old_beta.shape != (300,):
            raise ValueError(ws_msg(old_beta, "old_beta", "(300,)"))
        old_gp = (
            pad(old_gp, pad=[0, 3 - old_gp.shape[0]])
            if old_gp is not None
            else self.fixed_theta[:3]
        )
        if old_gp.shape != (3,):
            raise ValueError(ws_msg(old_gp, "old_gp", "(3,)"))
        old_np = (
            pad(old_np, pad=[0, 3 - old_np.shape[0]])
            if old_np is not None
            else self.fixed_theta[3:6]
        )
        if old_np.shape != (3,):
            raise ValueError(ws_msg(old_np, "old_np", "(3,)"))

        zeros = torch.zeros_like(old_gp)

        ## Previous context
        Bs = (self.S * old_beta).sum(dim=-1)  # (N, 3)
        joints = self.J @ (self.T + Bs)
        joints[1:] -= joints[self.parents[1:]]
        t = torch.stack([old_t, joints[0], joints[1]])[..., None]  # (3, 3, 1)
        r = axis_angle_to_matrix(torch.stack([zeros, old_gp, old_np]))  # (3, 3, 3)
        # Local poses
        Lg, L0, L1 = transform_mat(r, t).unbind()  # 3x (4, 4)
        S = torch.eye(4, device=old_t.device) * s
        S[-1, -1] = 1
        # Global neck pose G1 = Tg S T0 R0 T1 R1
        G1 = torch.chain_matmul(Lg, S, L0, L1)

        ## New context
        Bs = (self.S * new_beta).sum(dim=-1)
        joints = self.J @ (self.T + Bs)
        joints[1:] -= joints[self.parents[1:]]
        t = torch.stack([joints[0], joints[1]])[..., None]
        r = axis_angle_to_matrix(torch.stack([zeros, new_np]))
        T0, L1 = transform_mat(r, t).unbind()  # 2x (4, 4)
        ST0 = S @ T0
        G1 @= torch.linalg.inv(L1)
        # G1 L1^-1 = Tg S T0 R => tg = (G1 L1^-1)[:3, 3] - (S T0)[:3, 3]
        # R doesn't influence the last column
        tg = G1[:3, 3] - ST0[:3, 3]
        Tg = transform_mat(torch.eye(3, device=tg.device), tg[:, None])
        R0 = (torch.linalg.inv(Tg @ ST0) @ G1)[:3, :3]
        r0 = matrix_to_axis_angle(R0)

        return tg, r0

    @staticmethod
    def find_contour(
        vertices: Tensor,
        theta: Tensor,
        *,
        steps: int = 15
    ) -> LongTensor:
        r"""
        Get the indices of the vertices on the "contour" of the
        morphable model, wrt the XY plane. The contour is generated
        from the region of vertices which goes from the higher vertex
        to the chin height (vertex id: 3399).

        :param vertices: The final vertices. Its shape must be (N, 3).
        :type vertices: Tensor
        :param theta: The theta parameters used to obtain `vertices`.
            Its shape should be between (0,) and (15,).
        :type theta: Tensor
        :param steps: The number of points that will be returned for
            each side of the morphable model. The total number of
            indices will be 2`steps` + 1. Defaults to 15.
        :type steps: int, optional
        :return: The indices of the vertices which belong to the contour.
        :rtype: LongTensor
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(ws_msg(vertices, "vertices", "(N, 3)"))
        theta = pad(theta, pad=[0, 15 - theta.shape[0]])
        if theta.shape != (15,):
            raise ValueError(ws_msg(theta, "theta", "(15,)"))
        # Rotate vertices about the Z axis to align the Y axis of the neck wrt the world's one
        rot_chain = axis_angle_to_matrix(theta.view(-1, 3)[:2])
        y_vec = torch.tensor([0.0, 1.0, 0.0, 0.0], device=vertices.device)
        y_vec_neck = normalize((rot_chain[0] @ rot_chain[1])[1], dim=0)
        angle = torch.acos(torch.dot(y_vec[:2], y_vec_neck[:2])) * -torch.sign(y_vec_neck[0])
        Rx = axis_angle_to_matrix(pad(angle[None], pad=[2, 0]))
        vertices = vertices @ Rx.T

        # Vertices subdivision by height (from the higher vertex to chin height)
        max_y, max_y_idx = vertices[..., 1].max(dim=0)
        heights = torch.linspace(
            max_y.item(),
            vertices[3399, 1].item(),
            steps=steps + 1,
            device=vertices.device
        )[1:]
        lvl_y = heights.clone()
        delta = torch.abs(heights[1:] - heights[:-1]).mean() / 2
        h, v = heights.shape[0], vertices.shape[0]
        vertices = vertices[None].expand(h, v, 3)
        heights = heights[:, None].expand(h, v)
        levels_mask = (heights - delta < vertices[..., 1]) & (vertices[..., 1] < heights + delta)

        # Find two reference points (left and right) for each level
        lvl_x = vertices[..., 0].clone()
        lvl_x[~levels_mask] = torch.inf
        left = torch.stack([lvl_x.amin(dim=1), lvl_y], dim=1)[:, None].expand(h, v, 2)
        lvl_x[~levels_mask] = -torch.inf
        right = torch.stack([lvl_x.amax(dim=1), lvl_y], dim=1)[:, None].expand(h, v, 2)

        # Compute the distances
        left_dists = torch.linalg.norm(left - vertices[..., :2], dim=-1)  # (h, v)
        right_dists = torch.linalg.norm(right - vertices[..., :2], dim=-1)  # (h, v)
        # Remove distances which don't belong to the respective levels
        left_dists[~levels_mask] = right_dists[~levels_mask] = torch.inf

        # Find the minimum distance for each level
        left = left_dists.min(dim=1).indices
        right = right_dists.min(dim=1).indices

        # Return an ordered tensor of indices
        contour = torch.cat([left.flipud(), max_y_idx[None], right]).long()
        return contour
