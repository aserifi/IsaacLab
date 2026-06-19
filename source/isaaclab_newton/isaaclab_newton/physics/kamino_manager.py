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
from isaaclab.utils.timer import Timer

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
        """Reconcile body poses with the current joint configuration via the solver reset.

        The reset mode is chosen from the solver's ``use_fk_solver`` config:

        * ``use_fk_solver=True``: pass the written ``joint_q`` / ``joint_u`` so Kamino runs its
          Gauss-Newton FK solve, producing a consistent ``body_q`` for arbitrary (e.g.
          randomized) joint reset targets. Efficient for open-chain (tree) assets.
        * ``use_fk_solver=False``: reset to the assembled model-default reference (no joint
          targets), which restores the spawn configuration regardless of the current pose.
          This is the scalable, FK-free path for assets pinned to the assembled
          configuration (e.g. closed-loop mechanisms like DR Legs).

        With an all-``False`` (or ``None``) ``world_mask`` the reset is a no-op, so this is safe
        to call unconditionally every step -- mirroring the masked ``eval_fk`` pattern used by
        the minimal-coordinate solvers.

        Args:
            world_mask: Per-world mask indicating which worlds to reconcile.
                Shape ``(num_worlds,)``, dtype ``wp.bool``. If None, reconciles all worlds.
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
        """Update kinematics without stepping physics.

        Reconciles the worlds flagged by :meth:`invalidate_fk` through the solver reset (a
        no-op when none are flagged), so the explicit-reset path (``env.reset()`` ->
        ``sim.forward()``) makes ``body_q`` consistent before observations are read. For
        tree (non-closed-loop) assets, Newton's generic ``eval_fk`` additionally refreshes
        body poses from the joint coordinates.
        """
        cls._forward_kamino(world_mask=cls._world_reset_mask)
        if cls._world_reset_mask is not None:
            cls._world_reset_mask.zero_()
        if cls._fk_reset_mask is not None:
            cls._fk_reset_mask.zero_()
        # Closed-loop body poses are owned by the solver
        if not cls._has_loop_joints:
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Reconcile any worlds flagged for reset through the solver. The reset is world-masked
        # and idempotent, so this is a no-op on steps where no reset occurred.
        cls._forward_kamino(world_mask=cls._world_reset_mask)

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._solver.notify_model_changed(change)
                NewtonManager._model_changes = set()

        # Lazy CUDA graph capture: deferred from initialize_solver() when RTX was active.
        # By the time step() is first called, RTX has fully initialized (all cudaImportExternalMemory
        # calls are done) and is idle between render frames — giving us a clean capture window.
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        if cls._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device:  # type: ignore[union-attr]
            NewtonManager._graph_capture_pending = False
            NewtonManager._graph = cls._capture_relaxed_graph(device)
            if cls._graph is not None:
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
                logger.info("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)")
            else:
                logger.warning("Newton deferred CUDA graph capture failed; using eager execution")

        # Ensure body_q is up-to-date before collision detection.
        # After env resets, joint_q is written but body_q (used by
        # broadphase/narrowphase) is stale until FK runs.
        # Only runs FK for dirtied articulations via the accumulated mask.
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

        Sets :attr:`NewtonManager._needs_collision_pipeline` to ``True`` only
        when ``use_collision_detector=False`` (Kamino's internal detector
        handles contacts otherwise).

        Applies :attr:`KaminoSolverCfg.max_contacts_per_world`, when set, by overriding
        ``model.rigid_contact_max`` before solver construction. This bounds GPU memory
        for contact-rich multi-env training that would otherwise over-allocate from
        ``geoms.world_minimum_contacts``.
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

    @classmethod
    def _capture_or_defer_cuda_graph(cls) -> None:
        """Capture the physics CUDA graph, or defer if RTX is initializing."""
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        use_cuda_graph = cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            if not use_cuda_graph:
                NewtonManager._graph = None
                return
            if cls._usdrt_stage is None:
                # No RTX active — use standard Warp capture (cudaStreamCaptureModeGlobal).
                with wp.ScopedCapture() as capture:
                    cls._simulate_physics_only()
                NewtonManager._graph = capture.graph
                logger.info("Newton CUDA graph captured (standard Warp mode)")

                # TODO: streamline this with base NewtonManager
                # Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
            else:
                # RTX is active during initialization — cudaImportExternalMemory and other
                # non-capturable RTX ops run on background CUDA streams right now.
                # Defer capture to the first step() call, after RTX is fully initialized
                # and idle between render frames (clean capture window).
                NewtonManager._graph = None
                NewtonManager._graph_capture_pending = True
                logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")
