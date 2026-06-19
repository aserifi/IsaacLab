# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import logging

import warp as wp
from newton import Model, eval_fk
from newton.solvers import SolverKamino

from isaaclab.physics import PhysicsManager

from .kamino_manager_cfg import KaminoSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Uses Newton's :class:`CollisionPipeline` unless
    :attr:`KaminoSolverCfg.use_collision_detector` is ``True``, in which case
    Kamino's internal collision detector handles contact generation.
    """

    # Contains closed-loop articulations
    _has_loop_joints: bool = False

    @classmethod
    def _forward_kamino(cls, world_mask: wp.array | None = None) -> None:
        """Reconcile body poses with the joint configuration via the solver reset.

        With ``use_fk_solver=True`` the written joint coordinates are passed so Kamino runs its
        FK solve (handling arbitrary joint targets); otherwise the worlds are reset to the
        assembled model-default reference. Masked by ``world_mask``, so it is a no-op when no
        world is flagged and safe to call every step.

        Args:
            world_mask: Per-world reset mask, shape ``(num_worlds,)``, dtype ``wp.bool``.
                If None, all worlds are reconciled.
        """
        use_fk = bool(getattr(getattr(cls._solver, "_config", None), "use_fk_solver", False))
        if use_fk:
            cls._solver.reset(
                cls._state_0,
                world_mask=world_mask,
                joint_q=cls._state_0.joint_q,
                joint_u=cls._state_0.joint_qd,
            )
        else:
            cls._solver.reset(cls._state_0, world_mask=world_mask)

    @classmethod
    def forward(cls) -> None:
        """Reconcile reset-flagged worlds without stepping physics.

        Runs the masked solver reset (a no-op when nothing is flagged) so the explicit-reset
        path (``env.reset()`` -> ``sim.forward()``) makes ``body_q`` consistent before
        observations are read. Tree assets additionally get a generic ``eval_fk`` refresh.
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

        Sets :attr:`NewtonManager._needs_collision_pipeline` from ``use_collision_detector``
        (Kamino's internal detector handles contacts otherwise), applies
        :attr:`KaminoSolverCfg.max_contacts_per_world` to bound ``model.rigid_contact_max``,
        and detects closed-loop articulations for :attr:`_has_loop_joints`.
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

        # Detect closed-loop articulations
        cls._has_loop_joints = False
        _art_start = getattr(model, "articulation_start", None)
        _art_end = getattr(model, "articulation_end", None)
        if _art_start is not None and _art_end is not None:
            _art_start_np = _art_start.numpy()
            _art_end_np = _art_end.numpy()
            if _art_end_np.shape[0] > 0:
                cls._has_loop_joints = bool((_art_start_np[1:] > _art_end_np).any())
