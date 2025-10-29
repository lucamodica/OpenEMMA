import base64
import os.path
import re
import argparse
from datetime import datetime
from math import atan2
import time
import cv2
import numpy as np
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject


from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo, GenerateMotion


# check if the zod py package is available
if os.system("pip show zod") != 0:
    raise ImportError(
        "The zod package is not installed. Please install it by running 'pip install zod-python'."
    )

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='/datasets/zod/zodv2')
    parser.add_argument("--version", type=str, default='mini') # "mini" or "full"
    parser.add_argument("--method", type=str, default='openemma')
    args = parser.parse_args()    
    
    
    model = None
    processor = None
    tokenizer = None
    qwen25_loaded = False
    try:
        if "qwen" in args.model_path or "Qwen" in args.model_path:
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                
                print("model loaded to device:", model.device)
                print("device info: ", torch.cuda.get_device_name(model.device))
                
                processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
                tokenizer = None
                qwen25_loaded = True
            except Exception as e:
                print("Error while loading Qwen2.5-VL-3B-Instruct: ", e)
                print("Loading Qwen2-VL-7B-Instruct instead...")
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
                tokenizer = None
                qwen25_loaded = False
        else:
            if "llava" == args.model_path:    
                disable_torch_init()
                tokenizer, model, processor, context_len = load_pretrained_model("liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b")
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            elif "llava" in args.model_path:
                disable_torch_init()
                tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, "llava-v1.6-mistral-7b")
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                model = None
                processor = None
                tokenizer=None
    except Exception as e:
        print("Error while loading model: ", e)
        
    print("model loaded to device:", model.device)
    print("device info: ", torch.cuda.get_device_name(model.device))
    print(f"Model loaded: {model.__class__.__name__}")
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = args.model_path + f"_results/{args.method}/" + f"{timestamp}-zod_{args.version}"
    os.makedirs(timestamp, exist_ok=True)

    # Load the dataset
    zod = ZodSequences(dataset_root=args.dataroot, version=args.version)
    val_ids = zod.get_split(constants.VAL)
    
    draw_frames_every_nth = 5  # to match nuscenes recordig rate of frames in the scenes (2Hz instead of 10Hz)
    N_SEQUENCES = 150

    global_ade1s = []
    global_ade2s = []
    global_ade3s = []
    global_aveg_ades = []
    global_inference_times = []

    print(f"Number of sequences: {N_SEQUENCES}")
    for i in range(N_SEQUENCES):
        start_scene_time = time.time()
        
        seq = zod[val_ids[i]]
        seq_info = seq.info
        
        name = seq.id
        
        # Get all image and pose in this scene
        front_camera_images = []
        ego_poses = []
        camera_params = seq.calibration
        for j, frame in enumerate(seq_info.get_camera_frames()):
            if j % draw_frames_every_nth != 0:
                # Get the front camera image of the sample.
                front_camera_images.append(frame.read())

                # Get the ego pose of the sample.
                pose = seq.ego_motion.get_poses(frame.info.keyframe_time.timestamp())
                ego_poses.append(pose)
                    
        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")
        
        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ## Compute interpolated trajectory.
        # Get the velocities of the ego vehicle.
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # Get the curvature of the ego vehicle.
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],
                                                       atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)

        # Debug
        if args.plot:
            plt.figure(figsize=(10, 10))
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1],
                    color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.title(f"Ego Vehicle Trajectory Interpolation for Scene {name}")
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        # Get the waypoints of the ego vehicle.
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        for i in range(scene_length - TTL_LEN):
            # Get the raw image data.
            obs_images = front_camera_images[i:i+OBS_LEN]
            obs_ego_poses = ego_poses[i:i+OBS_LEN]
            obs_camera_params = camera_params[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

            # Get positions of the vehicle.
            obs_start_world = obs_ego_traj_world[0]
            fut_start_world = obs_ego_traj_world[-1]
            curr_image = obs_images[-1]

            # obs_images = [curr_image]

            # Allocate the images.
            with open(os.path.join(curr_image), "rb") as image_file:
                img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    
            # img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]

            for rho in range(3):
                # Assemble the prompt.
                if not "gpt" in args.model_path:
                    obs_images = curr_image
                (prediction,
                scene_description,
                object_description,
                updated_intent) = GenerateMotion(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                                obs_ego_curvatures, prev_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)

                # Process the output.
                prev_intent = updated_intent  # Stateful intent

                pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if not coordinates == []:
                    break
            if coordinates == []:
                print("**WARNING**: No valid coordinates found, skipping this frame.")
                continue
            
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # GT
            OverlayTrajectory(img, fut_ego_traj_world, obs_camera_params[-1], obs_ego_poses[-1], color=(255, 255, 0), args=args)

            # Pred
            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,
                                                                   pred_speeds,
                                                                   fut_start_world,
                                                                   atan2(obs_ego_velocities[-1][1],
                                                                         obs_ego_velocities[-1][0]), pred_len)

            # Overlay the trajectory.
            check_flag = OverlayTrajectory(img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args)

            # Compute ADE.
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
            
            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1] , axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len] , axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len] , axis=1))
            ade3s_list.append(ade3s)

            # Write to image.
            if args.plot == True:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                # Plot the trajectory.
                # plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                # plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                # plt.legend()
                # plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                # plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                # plt.close()

                # Save the trajectory
                # np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                # np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                # np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                # Save the descriptions
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Scene Description: {scene_description}\n")
                    f.write(f"Object Description: {object_description}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

            # break  # Timestep

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])
        inference_time = time.time() - start_scene_time

        global_ade1s.append(mean_ade1s)
        global_ade2s.append(mean_ade2s)
        global_ade3s.append(mean_ade3s)
        global_aveg_ades.append(aveg_ade)
        global_inference_times.append(inference_time)

        result = {
            "name": name,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
            "inference_time": inference_time
        }

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")
            
        print(f"Scene {name} done in {time.time() - start_scene_time} seconds. ADE1s: {mean_ade1s}, ADE2s: {mean_ade2s}, ADE3s: {mean_ade3s}, AvgADE: {aveg_ade}")

        break  # Scenes
        
    global_result = {
        "name": "OVERALL",
        "overall_ade1s": np.mean(global_ade1s),
        "overall_ade2s": np.mean(global_ade2s),
        "overall_ade3s": np.mean(global_ade3s),
        "overall_avgade": np.mean(global_aveg_ades),
        "overall_inference_time_per_scene": np.mean(global_inference_times)
    }
    with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(global_result))
            f.write("\n")


    print(f"Overall ADE1s: {np.mean(global_ade1s)}, ADE2s: {np.mean(global_ade2s)}, ADE3s: {np.mean(global_ade3s)}, AvgADE: {np.mean(global_aveg_ades)}")
    print(f"Overall Inference Time per Scene: {np.mean(global_inference_times)} seconds")



