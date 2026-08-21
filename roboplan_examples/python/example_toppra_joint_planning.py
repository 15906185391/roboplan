#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import tyro
import xacro
from pinocchio.visualize import ViserVisualizer

from common import get_home_configuration, get_model_data
from roboplan.core import JointPath, Scene, collapseContinuousJointPositions
from roboplan.example_models import get_package_share_dir
from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions
from roboplan.visualization import plotJointTrajectory, visualizeJointTrajectory


def _make_joint_waypoints(
    scene: Scene,
    group_name: str,
    q_home_full: np.ndarray,
    waypoint_count: int,
    path_span: float,
    curvature_scale: float,
) -> tuple[JointPath, np.ndarray]:
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2.")

    group_info = scene.getJointGroupInfo(group_name)
    q_start = collapseContinuousJointPositions(scene, group_name, q_home_full)
    q_lower, q_upper = scene.getPositionLimitVectors(group_name, collapsed=True)

    n_dof = len(q_start)
    if n_dof == 1:
        primary = np.ones(1)
        secondary = np.ones(1)
    else:
        primary = np.where(np.arange(n_dof) % 2 == 0, 1.0, -1.0)
        primary = primary / np.linalg.norm(primary)

        secondary = np.roll(primary, 1)
        secondary -= primary * float(secondary @ primary)
        secondary_norm = np.linalg.norm(secondary)
        if secondary_norm < 1e-9:
            secondary = np.roll(primary, 2)
            secondary -= primary * float(secondary @ primary)
            secondary_norm = np.linalg.norm(secondary)
        secondary = secondary / secondary_norm

    max_span = float(np.min(np.minimum(q_start - q_lower, q_upper - q_start)))
    if max_span <= 1e-9:
        raise RuntimeError(
            "Home configuration is too close to the joint limits to build a demo path."
        )

    base_span = min(path_span, 0.8 * max_span)
    curve_span = base_span * curvature_scale

    for scale in (1.0, 0.75, 0.55, 0.35, 0.2):
        span = base_span * scale
        curve = curve_span * scale
        path = JointPath()
        path.joint_names = list(group_info.joint_names)
        path.positions = []

        safe = True
        full_waypoints: list[np.ndarray] = []
        for idx in range(waypoint_count):
            s = float(idx) / float(waypoint_count - 1)
            q_group = q_start + s * span * primary + np.sin(np.pi * s) * curve * secondary
            q_group = np.clip(q_group, q_lower, q_upper)
            q_full = scene.toFullJointPositions(group_name, q_group)
            if scene.hasCollisions(q_full):
                safe = False
                break
            path.positions.append(q_group.copy())
            full_waypoints.append(q_full.copy())

        if safe and len(full_waypoints) == waypoint_count:
            return path, np.array(full_waypoints)

    path = JointPath()
    path.joint_names = list(group_info.joint_names)
    path.positions = []
    full_waypoints = []
    q_goal = np.clip(q_start + base_span * primary, q_lower, q_upper)
    for idx in range(waypoint_count):
        s = float(idx) / float(waypoint_count - 1)
        q_group = (1.0 - s) * q_start + s * q_goal
        q_full = scene.toFullJointPositions(group_name, q_group)
        path.positions.append(q_group.copy())
        full_waypoints.append(q_full.copy())
    return path, np.array(full_waypoints)


def _segment_samples_are_safe(scene: Scene, full_waypoints: np.ndarray, samples: int = 12) -> bool:
    for idx in range(len(full_waypoints) - 1):
        q_start = full_waypoints[idx]
        q_end = full_waypoints[idx + 1]
        for step in range(1, samples):
            fraction = float(step) / float(samples)
            q_interp = scene.interpolate(q_start, q_end, fraction)
            if scene.hasCollisions(q_interp):
                return False
    return True


def main(
    model: str = "ur5",
    toppra_mode: SplineFittingMode = SplineFittingMode.Adaptive,
    waypoint_count: int = 6,
    path_span: float = 0.45,
    curvature_scale: float = 0.25,
    dt: float = 0.01,
    velocity_scale: float = 1.0,
    acceleration_scale: float = 1.0,
    max_adaptive_iterations: int = 10,
    max_adaptive_step_size: float = 0.05,
    max_blend_deviation: float = 0.01,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Generate and replay a joint-space TOPPRA trajectory.

    Parameters:
        model: The name of the robot model to use.
        toppra_mode: The spline fitting mode used by TOPPRA.
        waypoint_count: Number of joint-space waypoints in the sparse path.
        path_span: Maximum joint-space excursion from the home pose.
        curvature_scale: Strength of the mid-path bend used to shape the sparse path.
        dt: Output trajectory sample period in seconds.
        velocity_scale: Scaling applied to the joint velocity limits.
        acceleration_scale: Scaling applied to the joint acceleration limits.
        max_adaptive_iterations: Maximum collision-check iterations for adaptive mode.
        max_adaptive_step_size: Maximum step size used to sample adaptive splines.
        max_blend_deviation: Maximum deviation for linear blend corner rounding.
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
    """
    model_data = get_model_data().get(model)
    if model_data is None:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()
    package_paths = [get_package_share_dir()]

    scene = Scene(
        "toppra_joint_space_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )
    group_name = model_data.default_joint_group

    q_home_full = get_home_configuration(scene, model_data)
    scene.setJointPositions(q_home_full)

    model = pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    viz = ViserVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True, loadModel=True, host=host, port=port)
    viz.display(q_home_full)
    time.sleep(0.1)

    path, full_waypoints = _make_joint_waypoints(
        scene,
        group_name,
        q_home_full,
        waypoint_count,
        path_span,
        curvature_scale,
    )
    if not _segment_samples_are_safe(scene, full_waypoints):
        print(
            "Warning: sampled path segments touch collision geometry; TOPPRA will still attempt to run."
        )

    toppra = PathParameterizerTOPPRA(scene, group_name)
    options = TOPPRAOptions(
        dt=dt,
        mode=toppra_mode,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        max_adaptive_iterations=max_adaptive_iterations,
        max_adaptive_step_size=max_adaptive_step_size,
        max_blend_deviation=max_blend_deviation,
    )

    print(f"Planning {len(path.positions)} joint-space waypoints with TOPPRA...")
    try:
        traj = toppra.generate(path, options)
    except Exception as exc:
        print(f"TOPPRA failed: {exc}")
        sys.exit(1)
    print(f"Trajectory duration: {traj.times[-1]:.3f} s")

    visualizeJointTrajectory(
        viz,
        scene,
        traj,
        model_data.ee_names,
        (0, 140, 220),
        "/toppra_joint_space/trajectory",
    )

    plt.figure()
    plt.ion()
    fig = plotJointTrajectory(
        traj,
        scene,
        group_name=group_name,
        title="TOPPRA Joint-Space Trajectory",
        positions=True,
        velocities=True,
        accelerations=True,
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

    print("Animating trajectory...")
    try:
        for q in traj.positions:
            q_full = scene.toFullJointPositions(group_name, q)
            scene.setJointPositions(q_full)
            viz.display(q_full)
            time.sleep(dt)
        print("Animation complete. Close the window or stop the process to exit.")
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
