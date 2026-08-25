#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform) -> np.ndarray:
    translation = np.array(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        dtype=np.float64,
    )
    rotation = _quat_to_matrix(
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w,
    )
    return points @ rotation.T + translation


def _point_cloud2_to_xyz(msg) -> np.ndarray:
    from sensor_msgs_py import point_cloud2

    raw_points = point_cloud2.read_points(
        msg, field_names=["x", "y", "z"], skip_nans=True
    )
    raw_dtype = getattr(raw_points, "dtype", None)
    if raw_dtype is not None and raw_dtype.names:
        points = np.column_stack([raw_points["x"], raw_points["y"], raw_points["z"]])
    else:
        points = np.asarray(raw_points).reshape(-1, 3)

    points = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def _voxel_downsample(points: np.ndarray, voxel_resolution: float) -> np.ndarray:
    if voxel_resolution <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_resolution).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def _stride_limit(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = int(np.ceil(len(points) / max_points))
    return points[::stride][:max_points]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/depth/points")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-frame", default="universe")
    parser.add_argument("--no-tf", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--voxel-resolution", type=float, default=0.04)
    parser.add_argument("--max-points", type=int, default=100000)
    args = parser.parse_args()

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    import tf2_ros

    rclpy.init(args=None)
    node = rclpy.create_node("roboplan_pointcloud_snapshot")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    buffer = tf2_ros.Buffer()
    _tf_listener = tf2_ros.TransformListener(buffer, node, spin_thread=False)
    latest: dict[str, PointCloud2] = {}

    def callback(msg: PointCloud2) -> None:
        if "msg" not in latest:
            latest["msg"] = msg

    subscription = node.create_subscription(
        PointCloud2, args.topic, callback, qos_profile_sensor_data
    )
    start = time.perf_counter()
    try:
        while "msg" not in latest:
            if time.perf_counter() - start > args.timeout:
                raise TimeoutError(f"Timed out waiting for PointCloud2 on {args.topic}")
            executor.spin_once(timeout_sec=0.1)

        msg = latest["msg"]
        source_frame = msg.header.frame_id or args.target_frame
        points = _point_cloud2_to_xyz(msg)
        raw_count = int(len(points))
        if raw_count == 0:
            raise ValueError(f"Point cloud topic {args.topic} did not contain finite XYZ points")

        if not args.no_tf and source_frame != args.target_frame:
            while True:
                try:
                    transform = buffer.lookup_transform(
                        args.target_frame, source_frame, Time()
                    )
                    points = _transform_points(points, transform)
                    break
                except Exception as exc:
                    if time.perf_counter() - start > args.timeout:
                        raise TimeoutError(
                            f"Timed out waiting for TF from {source_frame} "
                            f"to {args.target_frame}"
                        ) from exc
                    executor.spin_once(timeout_sec=0.1)

        points = _voxel_downsample(points, args.voxel_resolution)
        points = _stride_limit(points, args.max_points)
        if len(points) == 0:
            raise ValueError("Point cloud is empty after filtering/downsampling")

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.ascontiguousarray(points, dtype=np.float64))
        print(
            json.dumps(
                {
                    "topic": args.topic,
                    "source_frame": source_frame,
                    "target_frame": args.target_frame,
                    "raw_points": raw_count,
                    "points": int(len(points)),
                    "voxel_resolution": args.voxel_resolution,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    finally:
        node.destroy_subscription(subscription)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
