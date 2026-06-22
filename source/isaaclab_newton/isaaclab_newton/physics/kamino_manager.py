# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.core.types import vec6f
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)

# Kamino joint DoF type for a 6-DoF free joint (a floating base's root joint).
_FREE_DOF_TYPE = wp.constant(int(JointDoFType.FREE))


@wp.kernel(enable_backward=False)
def _gather_base_state_from_joints(
    base_joint_index: wp.array(dtype=wp.int32),
    joint_dof_type: wp.array(dtype=wp.int32),
    joint_coords_offset: wp.array(dtype=wp.int32),
    joint_dofs_offset: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    # outputs
    base_q: wp.array(dtype=wp.transformf),
    base_u: wp.array(dtype=vec6f),
):
    """Gather per-world ``base_q`` / ``base_u`` from the free base joint's coords.

    A floating base's root pose/velocity lands in the free joint's ``joint_q[0:7]`` /
    ``joint_qd[0:6]`` head; worlds without a free base joint default to identity / zero.
    """
    wid = wp.tid()
    # Default: identity pose / zero twist (keeps fixed / anchored bases at their reference).
    base_q[wid] = wp.transform_identity()
    base_u[wid] = vec6f(0.0)

    base_jid = base_joint_index[wid]
    if base_jid < 0:
        return
    if joint_dof_type[base_jid] != _FREE_DOF_TYPE:
        return

    c = joint_coords_offset[base_jid]
    d = joint_dofs_offset[base_jid]
    base_q[wid] = wp.transformf(
        wp.vec3(joint_q[c + 0], joint_q[c + 1], joint_q[c + 2]),
        wp.quat(joint_q[c + 3], joint_q[c + 4], joint_q[c + 5], joint_q[c + 6]),
    )
    base_u[wid] = vec6f(
        joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2], joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5]
    )


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    # Contains closed-loop articulations
    _has_loop_joints: bool = False

    # True when any world's base joint is a 6-DoF free joint (a floating base).
    _has_floating_base: bool = False

    _base_q: wp.array | None = None
    """Per-world base poses fed to the reset, ``(num_worlds,)`` ``wp.transformf`` (floating base only)."""

    _base_u: wp.array | None = None
    """Per-world base twists fed to the reset, ``(num_worlds,)`` ``vec6f`` (floating base only)."""

    @classmethod
    def _forward_kamino(cls, world_mask: wp.array | None = None) -> None:
        """Reconcile reset-flagged worlds, dispatching on :attr:`KaminoSolverCfg.use_fk_solver`.

        * FK on: FK-solve bodies from joint coords (+ per-env root for a floating base), then
          ``eval_fk`` overwrites ``body_q`` (needed for closed-loop FK resets to train).
        * Floating base, FK off: rigidly transform the assembled reference to the per-env root.
        * Fixed base, FK off: reset to the model-default reference.

        Args:
            world_mask: Per-world reset mask ``(num_worlds,)`` ``wp.bool``; ``None`` reconciles all.
        """
        use_fk = bool(getattr(getattr(cls._solver, "_config", None), "use_fk_solver", False))

        # Gather the per-env root pose/twist for a floating base (None for fixed base).
        base_q = None
        base_u = None
        if cls._has_floating_base:
            model_kamino = cls._solver._model_kamino
            wp.launch(
                _gather_base_state_from_joints,
                dim=model_kamino.size.num_worlds,
                inputs=[
                    model_kamino.info.base_joint_index,
                    model_kamino.joints.dof_type,
                    model_kamino.joints.coords_offset,
                    model_kamino.joints.dofs_offset,
                    cls._state_0.joint_q,
                    cls._state_0.joint_qd,
                ],
                outputs=[
                    cls._base_q,
                    cls._base_u,
                ],
                device=model_kamino.device,
            )
            base_q = cls._base_q
            base_u = cls._base_u

        if use_fk:
            # FK-solve bodies from the joint targets (+ per-env root), then refresh body_q via
            # eval_fk for a frame-consistent state (needed for closed-loop FK resets to train).
            cls._solver.reset(
                cls._state_0,
                world_mask=world_mask,
                joint_q=cls._state_0.joint_q,
                joint_u=cls._state_0.joint_qd,
                base_q=base_q,
                base_u=base_u,
            )
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
        elif base_q is not None:
            # Rigidly transform the assembled reference to the per-env root (exact, honors a
            # randomized root).
            cls._solver.reset(cls._state_0, world_mask=world_mask, base_q=base_q, base_u=base_u)
        else:
            # Fixed base, no FK: reset to the model-default reference.
            cls._solver.reset(cls._state_0, world_mask=world_mask)

    @classmethod
    def forward(cls) -> None:
        """Reconcile reset-flagged worlds (no stepping) so ``body_q`` is consistent before reads.

        Used by the explicit-reset path (``env.reset()`` -> ``sim.forward()``). Tree assets get an
        extra ``eval_fk`` refresh; closed-loop body poses are owned by the solver.
        """
        cls._forward_kamino(world_mask=cls._world_reset_mask)
        if cls._world_reset_mask is not None:
            cls._world_reset_mask.zero_()
        if cls._fk_reset_mask is not None:
            cls._fk_reset_mask.zero_()
        # Closed-loop body poses are owned by the solver.
        if not cls._has_loop_joints:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Reconcile reset-flagged worlds (masked, so a no-op when no reset occurred).
        cls._forward_kamino(world_mask=cls._world_reset_mask)

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                NewtonManager._model_changes = set()

        # Lazy CUDA graph capture deferred from initialize_solver() when RTX was active: by the
        # first step() RTX is initialized and idle, giving a clean capture window.
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            NewtonManager._graph_capture_pending = False
            NewtonManager._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                # Replay once to pin the buffers StateKamino.from_newton() lazily allocated
                # during capture before any eager solver.reset() reads them.
                wp.capture_launch(cls._graph)
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Refresh body_q for the collision pipeline on dirtied (tree) articulations.
        if cls._needs_collision_pipeline and not cls._has_loop_joints:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)

        # Zero both masks after consumption
        NewtonManager._world_reset_mask.zero_()
        NewtonManager._fk_reset_mask.zero_()

        # Step simulation (graphed or not; _graph is None when capture is disabled or failed)
        if cfg is not None and cfg.use_cuda_graph and cls._graph is not None and "cuda" in device:  # type: ignore[union-attr]
            wp.capture_launch(cls._graph)
        else:
            with wp.ScopedDevice(device):
                cls._simulate_physics_only()
        if cls._usdrt_stage is not None:
            cls._mark_transforms_dirty()

        # Launch solver-specific debug logging after stepping.
        cls._log_solver_debug()

        PhysicsManager._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: KaminoSolverCfg) -> None:
        """Construct :class:`SolverKamino` and populate the base-class slots.

        Sets :attr:`NewtonManager._needs_collision_pipeline`, caps ``model.rigid_contact_max`` via
        :attr:`KaminoSolverCfg.max_contacts_per_world`, and detects closed-loop articulations
        (:attr:`_has_loop_joints`) and a floating base (:attr:`_has_floating_base`), allocating the
        ``base_q`` / ``base_u`` buffers when one is present.
        """
        if solver_cfg.max_contacts_per_world is not None:
            model.rigid_contact_max = int(solver_cfg.max_contacts_per_world) * model.world_count
            logger.info(
                "[KAMINO] Capping rigid_contact_max to %d (%d/world * %d worlds)",
                model.rigid_contact_max,
                solver_cfg.max_contacts_per_world,
                model.world_count,
            )
        NewtonManager._solver = SolverKamino(model, solver_cfg.to_solver_config())
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_collision_detector

        # Detect a floating base: any world whose base joint is a 6-DoF free joint. Only then does
        # the reset need the per-env root pose/twist (base_q / base_u).
        cls._has_floating_base = False
        model_kamino = NewtonManager._solver._model_kamino
        base_joint_index = getattr(model_kamino.info, "base_joint_index", None)
        joint_dof_type = getattr(model_kamino.joints, "dof_type", None)
        if base_joint_index is not None and joint_dof_type is not None:
            base_jid_np = base_joint_index.numpy()
            dof_type_np = joint_dof_type.numpy()
            valid = base_jid_np[base_jid_np >= 0]
            if valid.size > 0:
                cls._has_floating_base = bool((dof_type_np[valid] == int(JointDoFType.FREE)).any())

        # Per-world base pose/twist buffers fed to the reset (see _forward_kamino), only needed
        # for a floating base.
        if cls._has_floating_base:
            num_worlds = model_kamino.size.num_worlds
            cls._base_q = wp.zeros(num_worlds, dtype=wp.transformf, device=model_kamino.device)
            cls._base_u = wp.zeros(num_worlds, dtype=vec6f, device=model_kamino.device)
        else:
            cls._base_q = None
            cls._base_u = None

        # Detect closed-loop articulations
        cls._has_loop_joints = False
        _art_start = getattr(model, "articulation_start", None)
        _art_end = getattr(model, "articulation_end", None)
        if _art_start is not None and _art_end is not None:
            _art_start_np = _art_start.numpy()
            _art_end_np = _art_end.numpy()
            if _art_end_np.shape[0] > 0:
                cls._has_loop_joints = bool((_art_start_np[1:] > _art_end_np).any())
