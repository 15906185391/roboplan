#!/usr/bin/env python3

import sys
import queue
import time
from collections import deque
import tyro
import xacro

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer

from common import (
    get_home_configuration,
    get_model_data,
    get_octree,
    load_octree_from_ros2_point_cloud,
)
from preview_visualization import make_static_and_preview_visualizers
from roboplan.core import (
    CartesianConfiguration,
    JointConfiguration,
    PathShortcutter,
    PathShortcuttingOptions,
    Scene,
)
from roboplan.example_models import get_package_share_dir
from roboplan.rrt import RRTOptions, RRT, visualizeTree
from roboplan.simple_ik import SimpleIk, SimpleIkOptions
from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions
from roboplan.visualization import (
    visualizeJointTrajectory,
    visualizePath,
    plotJointTrajectory,
    visualizeOcTree,
)


def _pump_matplotlib(delay: float = 0.001) -> None:
    try:
        plt.pause(delay)
    except RuntimeError as exc:
        if "Event loop already running" not in str(exc):
            raise
        time.sleep(delay)


def _sample_collision_free_full_configuration(
    scene: Scene,
    model_data,
    tries: int = 32,
) -> np.ndarray:
    """Return a full collision-free configuration, falling back to the home pose."""
    q_home = get_home_configuration(scene, model_data)
    if not scene.hasCollisions(q_home):
        return q_home

    for _ in range(tries):
        q_full = scene.randomCollisionFreePositions()
        if q_full is not None:
            return q_full

    raise RuntimeError("Could not sample a collision-free initial configuration.")


def main(
    model: str = "ur5",
    max_connection_distance: float = 3.0,
    collision_check_step_size: float = 0.05,
    collision_check_use_bisection: bool = True,
    goal_biasing_probability: float = 0.15,
    max_nodes: int = 1000,
    max_planning_time: float = 2.0,
    rrt_connect: bool = False,
    rrt_star: bool = False,
    rewire_distance: float = 5.0,
    fast_return: bool = True,
    include_shortcutting: bool = False,
    max_shortcutting_iters: int = 100,
    toppra_mode: SplineFittingMode = SplineFittingMode.Adaptive,
    host: str = "localhost",
    port: str = "8000",
    rng_seed: int | None = None,
    include_obstacles: bool = False,
    include_octrees: bool = False,
    octree_pointcloud_topic: str = "/camera/depth/points",
    octree_voxel_resolution: float = 0.04,
    octree_timeout_s: float = 5.0,
    octree_max_points: int = 100000,
    octree_target_frame: str = "universe",
    octree_use_tf: bool = False,
    interactive_goal: bool = True,
):
    """
    Run the RRT example with the provided parameters.

    Parameters:
        model: The name of the model to use.
        max_connection_distance: Maximum connection distance between two search nodes.
        collision_check_step_size: Configuration-space step size for collision checking along edges.
        collision_check_use_bisection: If true, uses bisection instead of linear search for collision checking along edges.
            This can be helpful in collision-dense environments, but has a lower worst-case performance.
        goal_biasing_probability: Weighting of the goal node during random sampling.
        max_nodes: The maximum number of nodes to add to the search tree.
        max_planning_time: The maximum time (in seconds) to search for a path.
        rrt_connect: Whether or not to use RRT-Connect.
        rrt_star: Whether or not to use RRT*, which keeps optimizing until the node or time budget is exhausted and returns the lowest-cost path. Can be combined with `rrt_connect`.
        rewire_distance: The configuration-space radius used to find neighbors for RRT* rewiring (only used when `rrt_star` is true). Should generally be at least `max_connection_distance`.
        fast_return: If true, return on the first path found; if false, plan until the node or time budget is exhausted and return the lowest-cost path. Set to false to get RRT*'s asymptotically optimal behavior.
        include_shortcutting: Whether or not to include path shortcutting for found paths.
        max_shortcutting_iters: The maximum number of path shortcutting iterations.
        toppra_mode: The trajectory generation mode for TOPP-RA. Can be `Hermite`, `Cubic`, or `Adaptive` (default).
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
        rng_seed: The seed for selecting random start and end poses and solving RRT.
        include_obstacles: Whether or not to include additional obstacles in the scene. Don't use with `include_octrees` argument
        include_octrees: Whether or not to include an octree obstacle converted from a ROS 2 PointCloud2 topic.
        octree_pointcloud_topic: ROS 2 PointCloud2 topic used to build the octree obstacle.
        octree_voxel_resolution: Voxel resolution used while converting the point cloud to an octree.
        octree_timeout_s: Seconds to wait for the point cloud and its TF transform before falling back.
        octree_max_points: Maximum number of downsampled point-cloud points passed to the octree builder.
        octree_target_frame: Planning frame used for the octree obstacle.
        octree_use_tf: Whether to transform point cloud data into the planning frame through ROS 2 TF.
            If false, raw point coordinates are used directly in the planning frame.
        interactive_goal: If true, let the goal end-effector poses be set by dragging Viser
            transform controls and preview the resulting IK solution before planning.
    """
    model_data = get_model_data().get(model)
    if model_data is None:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    package_paths = [get_package_share_dir()]

    # Pre-process with xacro. This is not necessary for raw URDFs.
    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    # Specify argument names to distinguish overloaded Scene constructors from python.
    scene = Scene(
        "test_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )
    group_info = scene.getJointGroupInfo(model_data.default_joint_group)
    q_indices = group_info.q_indices

    # Create a redundant Pinocchio model just for visualization with mimic joints.
    # When Pinocchio 4.x releases nanobind bindings, we should be able to directly grab the model from the scene instead.
    model = pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    preview_collision_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    # Optionally add obstacles.
    # Again, until Pinocchio 4.x releases nanobind bindings, we need to add the obstacles separately
    # to the scene and to the Pinocchio models used for visualization.
    if include_obstacles:
        for obstacle in model_data.obstacles:
            obstacle.addToScene(scene)
            obstacle.addToPinocchioModels(model, collision_model, visual_model)

    fixed_viz, preview_viz = make_static_and_preview_visualizers(
        model,
        collision_model,
        visual_model,
        host,
        port,
        preview_collision_model=preview_collision_model,
    )

    if include_octrees:
        try:
            print(
                f"Loading octree obstacle from ROS 2 point cloud topic "
                f"{octree_pointcloud_topic}...",
                flush=True,
            )
            obstacle = load_octree_from_ros2_point_cloud(
                topic_name=octree_pointcloud_topic,
                voxel_resolution=octree_voxel_resolution,
                timeout_sec=octree_timeout_s,
                target_frame=octree_target_frame,
                max_points=octree_max_points,
                use_tf=octree_use_tf,
            )
            print(
                f"Loaded octree obstacle from {octree_pointcloud_topic} "
                f"at {octree_voxel_resolution:.3f} m resolution.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Warning: could not build octree from ROS 2 topic "
                f"{octree_pointcloud_topic}: {exc}",
                flush=True,
            )
            print("Falling back to the bundled example point cloud.", flush=True)
            obstacle = get_octree()
        obstacle.addToScene(scene)
        geom_obj = obstacle.createGeometryObject(model)
        visualizeOcTree(fixed_viz, geom_obj, fixed_viz.collisionRootNodeName)
        visualizeOcTree(fixed_viz, geom_obj, fixed_viz.visualRootNodeName)

    # Set up an RRT and perform path planning.
    options = RRTOptions(
        group_name=model_data.default_joint_group,
        max_nodes=max_nodes,
        max_connection_distance=max_connection_distance,
        collision_check_step_size=collision_check_step_size,
        collision_check_use_bisection=collision_check_use_bisection,
        goal_biasing_probability=goal_biasing_probability,
        max_planning_time=max_planning_time,
        rrt_connect=rrt_connect,
        rrt_star=rrt_star,
        rewire_distance=rewire_distance,
        fast_return=fast_return,
    )
    rrt = RRT(scene, options)

    toppra = PathParameterizerTOPPRA(scene, model_data.default_joint_group)
    traj_dt = 0.01

    if include_shortcutting:
        shortcutting_options = PathShortcuttingOptions(
            group_name=model_data.default_joint_group,
            max_step_size=options.collision_check_step_size,
            max_iters=max_shortcutting_iters,
        )
        shortcutter = PathShortcutter(scene, shortcutting_options)

    traj_queue = queue.Queue()
    cur_traj = None
    animate = False
    plan_stats = fixed_viz.viewer.gui.add_text(
        "Plan stats",
        "Waiting for the first plan.",
        disabled=True,
    )

    if rng_seed:
        rrt.setRngSeed(rng_seed)

    q_full = _sample_collision_free_full_configuration(scene, model_data)
    scene.setJointPositions(q_full)
    fixed_viz.display(q_full)
    preview_viz.display(q_full)
    time.sleep(0.1)

    if interactive_goal:
        goal_ik = SimpleIk(
            scene,
            SimpleIkOptions(
                group_name=model_data.default_joint_group,
                max_iters=100,
                step_size=1.0,
                max_linear_error_norm=0.001,
                max_angular_error_norm=0.001,
                check_collisions=True,
            ),
        )
        q_goal_full = _sample_collision_free_full_configuration(scene, model_data)
        q_goal_initial_full = q_goal_full.copy()
        q_goal_solution = JointConfiguration()
        q_goal_seed = JointConfiguration()
        q_goal_seed.positions = q_goal_full[q_indices].copy()
        goal_targets: list[CartesianConfiguration] = []
        goal_controls = []
        for ee_name in model_data.ee_names:
            target = CartesianConfiguration()
            target.base_frame = model_data.base_link
            target.tip_frame = ee_name
            goal_targets.append(target)

            pose = scene.forwardKinematics(q_goal_full, ee_name)
            controls = fixed_viz.viewer.scene.add_transform_controls(
                f"/rrt/goal/{ee_name}",
                depth_test=False,
                scale=0.2,
                disable_sliders=True,
                visible=True,
            )
            controls.position = pose[:3, 3].copy()
            controls.wxyz = pin.Quaternion(pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
            goal_controls.append(controls)

        preview_viz.display(q_goal_full)

        solve_times = deque(maxlen=30)
        solve_stats = fixed_viz.viewer.gui.add_text(
            "Solve stats",
            "Waiting for plan-time IK.",
            disabled=True,
        )
        preview_state = {"busy": False}

        def _update_goal_preview() -> bool:
            nonlocal q_goal_full
            if preview_state["busy"]:
                return False
            preview_state["busy"] = True
            try:
                world_T_base = scene.forwardKinematics(q_full, model_data.base_link)
                for target, controls in zip(goal_targets, goal_controls):
                    world_T_target = pin.SE3(
                        pin.Quaternion(controls.wxyz[[1, 2, 3, 0]]), controls.position
                    ).homogeneous
                    target.tform = np.linalg.inv(world_T_base) @ world_T_target

                q_goal_seed.positions = q_goal_full[q_indices].copy()
                t_start = time.perf_counter()
                result = goal_ik.solveIk(goal_targets, q_goal_seed, q_goal_solution)
                elapsed = time.perf_counter() - t_start
                solve_times.append(elapsed)
                avg = sum(solve_times) / len(solve_times)
                freq = 1.0 / avg if avg > 0.0 else 0.0
                stats = (
                    f"last {elapsed * 1000.0:.1f} ms | "
                    f"avg {avg * 1000.0:.1f} ms | "
                    f"{freq:.1f} Hz"
                )
                solve_stats.value = stats
                print(f"Solve stats: {stats}")

                if not result:
                    return False

                q_goal_full = scene.toFullJointPositions(
                    model_data.default_joint_group, q_goal_solution.positions
                )
                preview_viz.display(q_goal_full)
                q_goal_seed.positions = q_goal_solution.positions
                return True
            finally:
                preview_state["busy"] = False

        status_text = fixed_viz.viewer.gui.add_text(
            "Status",
            "Drag the goal markers, then plan the path. IK runs only when planning starts.",
            disabled=True,
        )
        plan_button = fixed_viz.viewer.gui.add_button("Plan path")
        reset_button = fixed_viz.viewer.gui.add_button("Reset goal")
        animate_button = fixed_viz.viewer.gui.add_button("Animate trajectory")
        animate_button.disabled = True

        @reset_button.on_click
        def reset_goal(_):
            nonlocal q_goal_full
            q_goal_full = q_goal_initial_full.copy()
            for ee_name, controls in zip(model_data.ee_names, goal_controls):
                pose = scene.forwardKinematics(q_goal_full, ee_name)
                controls.position = pose[:3, 3].copy()
                controls.wxyz = pin.Quaternion(pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
            scene.setJointPositions(q_full)
            fixed_viz.display(q_full)
            preview_viz.display(q_goal_full)
            status_text.value = "Goal reset to the initial sampled pose."

        @plan_button.on_click
        def plan_path(_):
            nonlocal animate
            animate = False
            plan_button.disabled = True
            animate_button.disabled = True

            if not _update_goal_preview():
                status_text.value = "Goal IK failed. Adjust the markers and try again."
                plan_button.disabled = False
                animate_button.disabled = False
                return

            start = JointConfiguration()
            start.positions = q_full[q_indices]
            assert start.positions is not None

            goal = JointConfiguration()
            goal.positions = q_goal_full[q_indices]
            assert goal.positions is not None

            print("\nPlanning...")
            plan_start = time.perf_counter()
            try:
                path = rrt.plan(start, goal)
            finally:
                plan_button.disabled = False
                animate_button.disabled = False
            plan_elapsed = time.perf_counter() - plan_start
            shortcut_elapsed = 0.0

            if include_shortcutting:
                print("Shortcutting path...")
                shortcut_start = time.perf_counter()
                shortened_path = shortcutter.shortcut(path)
                shortcut_elapsed = time.perf_counter() - shortcut_start

            print("Generating trajectory...")
            traj_start = time.perf_counter()
            traj = toppra.generate(
                shortened_path if include_shortcutting else path,
                TOPPRAOptions(dt=traj_dt, mode=toppra_mode),
            )
            traj_elapsed = time.perf_counter() - traj_start
            total_elapsed = plan_elapsed + shortcut_elapsed + traj_elapsed
            stats = (
                f"rrt {plan_elapsed:.3f} s | "
                f"shortcut {shortcut_elapsed:.3f} s | "
                f"toppra {traj_elapsed:.3f} s | "
                f"total {total_elapsed:.3f} s"
            )
            plan_stats.value = stats
            print(f"Plan stats: {stats}")

            fixed_viz.display(q_full)
            preview_viz.display(q_full)
            visualizeTree(fixed_viz, scene, rrt, model_data.ee_names, 0.05)

            q_start_full = scene.toFullJointPositions(
                model_data.default_joint_group, start.positions
            )
            q_goal_full_local = scene.toFullJointPositions(
                model_data.default_joint_group, goal.positions
            )
            for ee_name in model_data.ee_names:
                fixed_viz.viewer.scene.add_icosphere(
                    f"/rrt/start/{ee_name}",
                    radius=0.03,
                    color=(0, 200, 0),
                    position=scene.forwardKinematics(q_start_full, ee_name)[:3, 3],
                )
                fixed_viz.viewer.scene.add_icosphere(
                    f"/rrt/goal/{ee_name}",
                    radius=0.03,
                    color=(200, 0, 0),
                    position=scene.forwardKinematics(q_goal_full_local, ee_name)[:3, 3],
                )

            if include_shortcutting:
                visualizePath(
                    fixed_viz, scene, path, model_data.ee_names, 0.05, (100, 0, 0), "/rrt/path"
                )
                visualizeJointTrajectory(
                    preview_viz,
                    scene,
                    traj,
                    model_data.ee_names,
                    (0, 120, 255),
                    "/rrt/shortcut_path_preview",
                )
            else:
                visualizeJointTrajectory(
                    preview_viz,
                    scene,
                    traj,
                    model_data.ee_names,
                    (0, 120, 255),
                    "/rrt/path_preview",
                )

            preview_viz.display(q_full)
            for q in traj.positions:
                q_step_full = scene.toFullJointPositions(
                    model_data.default_joint_group, q
                )
                scene.setJointPositions(q_step_full)
                preview_viz.display(q_step_full)
                time.sleep(traj_dt)

            traj_queue.put(traj)
            status_text.value = "Planning complete. Use Animate trajectory to replay it."

        @animate_button.on_click
        def animate_trajectory(_):
            nonlocal animate
            plan_button.disabled = True
            animate_button.disabled = True
            animate = True

        status_text.value = "Ready. Drag the goal markers; IK runs when planning starts."

        plt.figure()
        plt.ion()
        plt.show(block=False)
        _pump_matplotlib()
        while True:
            if not traj_queue.empty():
                plt.clf()
                cur_traj = traj_queue.get()
                fig = plotJointTrajectory(
                    cur_traj, scene, group_name=model_data.default_joint_group
                )
                plt.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                _pump_matplotlib()
            elif animate and cur_traj is not None:
                print("Animating trajectory...")
                fixed_viz.display(q_full)
                for q in cur_traj.positions:
                    q_step_full = scene.toFullJointPositions(
                        model_data.default_joint_group, q
                    )
                    scene.setJointPositions(q_step_full)
                    fixed_viz.display(q_step_full)
                    time.sleep(traj_dt)
                    _pump_matplotlib()
                preview_viz.display(q_step_full)
                animate = False
                plan_button.disabled = False
                animate_button.disabled = False
                print("...done!")
            else:
                _pump_matplotlib(0.05)

        return

    # Create a path planning button.
    plan_button = fixed_viz.viewer.gui.add_button("Plan path")

    @plan_button.on_click
    def plan_path(_):
        nonlocal animate
        animate = False
        plan_button.disabled = True
        animate_button.disabled = True

        start = JointConfiguration()
        start.positions = q_full[q_indices]
        assert start.positions is not None

        goal = JointConfiguration()
        goal.positions = _sample_collision_free_full_configuration(
            scene, model_data
        )[q_indices]
        assert goal.positions is not None

        print("\nPlanning...")
        plan_start = time.perf_counter()
        try:
            path = rrt.plan(start, goal)
        finally:
            plan_button.disabled = False
            animate_button.disabled = False
        plan_elapsed = time.perf_counter() - plan_start
        shortcut_elapsed = 0.0

        # Optionally include path shortening
        if include_shortcutting:
            print("Shortcutting path...")
            shortcut_start = time.perf_counter()
            shortened_path = shortcutter.shortcut(path)
            shortcut_elapsed = time.perf_counter() - shortcut_start

        # Set up TOPP-RA to time-parameterize the path
        print("Generating trajectory...")
        traj_start = time.perf_counter()
        traj = toppra.generate(
            shortened_path if include_shortcutting else path,
            TOPPRAOptions(dt=traj_dt, mode=toppra_mode),
        )
        traj_elapsed = time.perf_counter() - traj_start
        total_elapsed = plan_elapsed + shortcut_elapsed + traj_elapsed
        stats = (
            f"rrt {plan_elapsed:.3f} s | "
            f"shortcut {shortcut_elapsed:.3f} s | "
            f"toppra {traj_elapsed:.3f} s | "
            f"total {total_elapsed:.3f} s"
        )
        plan_stats.value = stats
        print(f"Plan stats: {stats}")

        # Visualize the tree and path
        fixed_viz.display(q_full)
        visualizeTree(fixed_viz, scene, rrt, model_data.ee_names, 0.05)

        # Show the start (green) and goal (red) end-effector positions.
        q_start_full = scene.toFullJointPositions(
            model_data.default_joint_group, start.positions
        )
        q_goal_full = scene.toFullJointPositions(
            model_data.default_joint_group, goal.positions
        )
        for ee_name in model_data.ee_names:
            fixed_viz.viewer.scene.add_icosphere(
                f"/rrt/start/{ee_name}",
                radius=0.03,
                color=(0, 200, 0),
                position=scene.forwardKinematics(q_start_full, ee_name)[:3, 3],
            )
            fixed_viz.viewer.scene.add_icosphere(
                f"/rrt/goal/{ee_name}",
                radius=0.03,
                color=(200, 0, 0),
                position=scene.forwardKinematics(q_goal_full, ee_name)[:3, 3],
            )

        if include_shortcutting:
            visualizePath(
                fixed_viz, scene, path, model_data.ee_names, 0.05, (100, 0, 0), "/rrt/path"
            )
            visualizeJointTrajectory(
                preview_viz,
                scene,
                traj,
                model_data.ee_names,
                (0, 120, 255),
                "/rrt/shortcut_path_preview",
            )
        else:
            visualizeJointTrajectory(
                preview_viz,
                scene,
                traj,
                model_data.ee_names,
                (0, 120, 255),
                "/rrt/path_preview",
            )

        traj_queue.put(traj)
        plan_button.disabled = False
        animate_button.disabled = False

    # Create a trajectory animation button.
    animate_button = fixed_viz.viewer.gui.add_button("Animate trajectory")
    animate_button.disabled = True

    @animate_button.on_click
    def animate_trajectory(_):
        plan_button.disabled = True
        animate_button.disabled = True
        nonlocal animate
        animate = True

    # Main display and animation loop.
    plt.figure()
    plt.ion()
    plt.show(block=False)
    _pump_matplotlib()
    while True:
        if not traj_queue.empty():
            plt.clf()
            cur_traj = traj_queue.get()
            fig = plotJointTrajectory(
                cur_traj, scene, group_name=model_data.default_joint_group
            )
            plt.draw()
            fig.canvas.draw()
            fig.canvas.flush_events()
            _pump_matplotlib()
        elif animate and cur_traj is not None:
            print("Animating trajectory...")
            fixed_viz.display(q_full)
            for q in cur_traj.positions:
                q_step_full = scene.toFullJointPositions(
                    model_data.default_joint_group, q
                )
                scene.setJointPositions(q_step_full)
                fixed_viz.display(q_step_full)
                time.sleep(traj_dt)
                _pump_matplotlib()
            preview_viz.display(q_step_full)
            animate = False
            plan_button.disabled = False
            animate_button.disabled = False
            print("...done!")
        else:
            _pump_matplotlib(0.05)


if __name__ == "__main__":
    tyro.cli(main)
