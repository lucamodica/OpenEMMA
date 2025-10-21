import os
import re
import argparse
from datetime import datetime
from math import atan2, cos, sin

import cv2
import numpy as np
import matplotlib.pyplot as plt

from nuscenes import NuScenes

from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)

from openemma.vlm.base_backbone import BaseOpenEMMA


OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


def parse_xy_list(text: str):
    """Parse "[x,y]" pairs from free-form text."""
    coords = re.findall(
        r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]",
        text or "",
    )
    return [[float(x), float(y)] for x, y in coords]


def world_to_ego_local(points_world: np.ndarray, origin_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    """
    Convert world XY points to ego local frame at origin_world with heading yaw_rad.
    Local frame convention: +y forward, +x left.
    """
    h = np.array([cos(yaw_rad), sin(yaw_rad)])  # forward (y)
    l = np.array([-sin(yaw_rad), cos(yaw_rad)])  # left (x)
    deltas = points_world[:, :2] - origin_world[:2]
    local_y = deltas @ h
    local_x = deltas @ l
    return np.stack([local_x, local_y], axis=1)


def ego_local_to_world(points_local: np.ndarray, origin_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    """
    Convert ego local XY (x left, y forward) to world XYZ at origin_world with heading yaw_rad.
    """
    h = np.array([cos(yaw_rad), sin(yaw_rad)])  # forward (y)
    l = np.array([-sin(yaw_rad), cos(yaw_rad)])  # left (x)
    world_xy = np.outer(points_local[:, 1], h) + np.outer(points_local[:, 0], l)
    world_xy += origin_world[:2]
    z = np.full((points_local.shape[0], 1), origin_world[2])
    return np.hstack([world_xy, z])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--api-key", type=str, default="", help="API key for GPT models if --model-id starts with gpt")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default="data/sets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--method", type=str, default="openemma")
    # Back-compat alias (main.py uses --model-path)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    if args.model_path and not args.model_id:
        args.model_id = args.model_path

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = f"{args.model_id}_results/{args.method}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize VLM backbone
    backbone = BaseOpenEMMA(args)

    # Load NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    for scene in scenes:
        token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        name = scene["name"]
        description = scene["description"]

        # Mirror main.py filtering
        if name not in ["scene-0103", "scene-1077"]:
            continue

        front_camera_images = []
        ego_poses = []
        camera_params = []

        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get("sample", curr_sample_token)
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"]) 

            # For parity with main.py, this call is kept (renders sample visualization)
            nusc.render_sample_data(cam_front_data["token"])  # noqa: F841

            front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data["filename"]))
            ego_poses.append(nusc.get("ego_pose", cam_front_data["ego_pose_token"]))
            camera_params.append(nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"]))

            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample["next"]

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")
        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        # Prepare arrays and diagnostics similar to main.py
        ego_poses_world = np.array([ego_poses[t]["translation"][:3] for t in range(scene_length)])

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_vel_norm = np.linalg.norm(ego_velocities, axis=1)
        est_pts = IntegrateCurvatureForPoints(
            ego_curvatures,
            ego_vel_norm,
            ego_poses_world[0],
            atan2(ego_velocities[0][1], ego_velocities[0][0]),
            scene_length,
        )

        if args.plot:
            plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")
            plt.plot(est_pts[:, 0], est_pts[:, 1], "g-", label="Reconstruction")
            plt.legend()
            plt.savefig(f"{out_dir}/{name}_interpolation.jpg")
            plt.close()

        cam_images_sequence = []
        ade1s_list, ade2s_list, ade3s_list = [], [], []

        for i in range(scene_length - TTL_LEN):
            obs_images = front_camera_images[i : i + OBS_LEN]
            obs_world = [ego_poses_world[t] for t in range(i, i + OBS_LEN)]
            fut_world = [ego_poses_world[t] for t in range(i + OBS_LEN, i + TTL_LEN)]
            obs_vel = ego_velocities[i : i + OBS_LEN]

            fut_start_world = np.array(obs_world[-1])
            yaw = atan2(obs_vel[-1][1], obs_vel[-1][0])

            # Convert to ego-local (x left, y forward)
            obs_local = world_to_ego_local(np.array(obs_world), fut_start_world, yaw)
            fut_local = world_to_ego_local(np.array(fut_world), fut_start_world, yaw)

            # Build diffs
            obs_diff = np.zeros_like(obs_local)
            obs_diff[1:] = obs_local[1:] - obs_local[:-1]
            obs_diff[0] = obs_diff[1]

            fut_diff = np.zeros_like(fut_local)
            fut_diff[1:] = fut_local[1:] - fut_local[:-1]
            fut_diff[0] = fut_diff[1]

            # High-level command from GT future
            command = backbone.compute_command(fut_local)

            data = {
                "gt_ego_fut_diff": fut_diff,
                "gt_ego_fut_trajs": fut_local,
                "gt_ego_his_diff": obs_diff,
                "gt_ego_his_trajs": obs_local,
            }

            curr_image_path = obs_images[-1]

            # Try up to 3 attempts to obtain parseable output
            prediction_text = None
            pred_local = []
            for _ in range(3):
                prediction_text = backbone.generate_waypoints(
                    command, curr_image_path, data=data, backbone=backbone, args=args
                )
                pred_local = parse_xy_list(prediction_text)
                if pred_local:
                    break

            if not pred_local:
                continue

            pred_local = np.array(pred_local[:FUT_LEN], dtype=float)
            pred_local = np.array(backbone.fix_traj(pred_local.tolist()))

            # Convert predicted local XY to world XYZ
            pred_traj_world = ego_local_to_world(pred_local, fut_start_world, yaw)

            # Visualization image
            with open(curr_image_path, "rb") as f:
                img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

            # Overlay predicted trajectory
            OverlayTrajectory(
                img,
                pred_traj_world.tolist(),
                camera_params[i + OBS_LEN - 1],
                ego_poses[i + OBS_LEN - 1],
                color=(255, 0, 0),
                args=args,
            )

            # Compute ADE metrics
            fut_world_arr = np.array(fut_world)
            pred_len = min(FUT_LEN, len(pred_traj_world))
            ade = np.mean(np.linalg.norm(fut_world_arr[:pred_len] - pred_traj_world[:pred_len], axis=1))

            pred1_len = min(pred_len, 2)
            ade1s = np.mean(
                np.linalg.norm(fut_world_arr[:pred1_len] - pred_traj_world[1 : pred1_len + 1], axis=1)
            )
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_world_arr[:pred2_len] - pred_traj_world[:pred2_len], axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_world_arr[:pred3_len] - pred_traj_world[:pred3_len], axis=1))
            ade3s_list.append(ade3s)

            # Save per-frame artifacts
            if args.plot:
                cv2.imwrite(f"{out_dir}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_world_arr[:, 0], fut_world_arr[:, 1], "r-", label="GT")
                plt.plot(pred_traj_world[:, 0], pred_traj_world[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade:.3f}")
                plt.savefig(f"{out_dir}/{name}_{i}_traj.jpg")
                plt.close()

                np.save(f"{out_dir}/{name}_{i}_pred_traj.npy", pred_traj_world)
                np.save(f"{out_dir}/{name}_{i}_pred_local.npy", pred_local)

                with open(f"{out_dir}/{name}_{i}_logs.txt", "w") as lf:
                    lf.write(f"High-level command: {command}\n")
                    lf.write(f"Raw VLM output: {prediction_text}\n")
                    lf.write(f"ADE: {ade}\n")

            # Keep sequence for video
            if args.plot:
                cam_images_sequence.append(img.copy())

        # Per-scene summary and video
        if ade1s_list and ade2s_list and ade3s_list:
            mean_ade1s = float(np.mean(ade1s_list))
            mean_ade2s = float(np.mean(ade2s_list))
            mean_ade3s = float(np.mean(ade3s_list))
            avg_ade = float(np.mean([mean_ade1s, mean_ade2s, mean_ade3s]))

            import json
            result = {
                "name": name,
                "token": token,
                "ade1s": mean_ade1s,
                "ade2s": mean_ade2s,
                "ade3s": mean_ade3s,
                "avgade": avg_ade,
            }
            with open(f"{out_dir}/ade_results.jsonl", "a") as f:
                f.write(json.dumps(result))
                f.write("\n")

            if args.plot and cam_images_sequence:
                WriteImageSequenceToVideo(cam_images_sequence, f"{out_dir}/{name}")


if __name__ == "__main__":
    main()
