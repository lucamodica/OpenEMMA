import base64
import requests
import time
import random
import io
import base64
from math import atan2
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import torch
import re
from PIL import Image
from qwen_vl_utils import process_vision_info
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates


random.seed(42)

KEY = "<your-api-key>"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4(question, api_key=None, image_path=None, proxy='openai', sys_message=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    if image_path is not None:
        base64_image = encode_image(image_path)
        if sys_message is not None:
            params = {
                "messages": [
                    {
                    "role": "system", 
                    "content": sys_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:

            params = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o-mini-2024-07-18',
                "temperature": 0.0
            }
    else:
        if sys_message is not None:
            params = {
                "messages": [

                    {
                        "role": "system", 
                        "content": sys_message
                    },
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:
            params = {
                "messages": [
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }


    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content


def PlotBase64Image(image: str):
    i = base64.b64decode(image)
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='JPG')

    plt.imshow(i, interpolation='nearest')
    plt.show()



def TransformPoint(point, transform):
    """ Transform a 3D point using a transformation matrix. """
    if isinstance(point, list):
        point = np.array(point)

    if point.shape[-1] == 3:
        point = np.append(point, 1)
    transformed_point = transform @ point
    return transformed_point[:3]

def FormTransformationMatrix(translation, rotation):
    """ Create a transformation matrix from translation and rotation (as a quaternion). """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def ProjectEgoToImage(points_3d: np.array, K):
    """ Project 3D points to 2D using camera intrinsic matrix K. """
    # Filter out points that are behind the camera
    points_3d = points_3d[points_3d[:, 2] > 0]

    # Project the remaining points
    points_2d = np.dot(K, points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2][:, np.newaxis]  # Normalize by depth
    return points_2d

def ProjectWorldToImage(points3d_world: list, cam_to_ego, ego_to_world):
    # Plot the waypoints.

    T_ego_global = FormTransformationMatrix(ego_to_world['translation'], Quaternion(ego_to_world['rotation']))
    T_cam_ego = FormTransformationMatrix(cam_to_ego['translation'], Quaternion(cam_to_ego['rotation']))
    T_cam_global = T_ego_global @ T_cam_ego
    T_global_cam = np.linalg.inv(T_cam_global)

    points3d_cam = [TransformPoint(point, T_global_cam) for point in points3d_world]

    points3d_img = ProjectEgoToImage(np.array(points3d_cam), cam_to_ego['camera_intrinsic'])

    return points3d_img


def OffsetTrajectory3D(points, offset_distance):
    """
    Offsets a 3D trajectory by a specified distance normal to the trajectory.

    Parameters:
        points (np.ndarray): n x 3 array representing the 3D trajectory (x, y, z).
        offset_distance (float): Distance to offset the trajectory.

    Returns:
        np.ndarray: Offset trajectory as an n x 3 array.
    """
    # Compute differences to find tangent vectors
    tangents = np.gradient(points, axis=0)  # Approximate tangents
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize tangents

    # Reference vector for normal plane computation (e.g., z-axis)
    reference_vector = np.array([0, 0, 1])

    # Compute normal vectors via cross product
    normals = np.cross(tangents, reference_vector)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize normals

    # Compute offset points
    offset_points = points + offset_distance * normals

    return offset_points

def OverlayTrajectory(img, points3d_world: list, cam_to_ego, ego_to_world, color=(0, 0, 255), args=None):

    # Construct left/right boundaries.
    points3d_left_world = OffsetTrajectory3D(np.array(points3d_world), -1.73 / 2)
    points3d_right_world = OffsetTrajectory3D(np.array(points3d_world), 1.73 / 2)

    # Project the waypoints to the image.
    points3d_img = ProjectWorldToImage(points3d_world, cam_to_ego, ego_to_world)
    points3d_left_img = ProjectWorldToImage(points3d_left_world.tolist(), cam_to_ego, ego_to_world)
    points3d_right_img = ProjectWorldToImage(points3d_right_world.tolist(), cam_to_ego, ego_to_world)

    if args.plot:
        # Overlay the waypoints on the image.
        for i in range(len(points3d_img) - 1):
            cv2.circle(img, tuple(points3d_img[i].astype(int)), radius=6, color=color, thickness=-1)

        # # Draw lines.
        # for i in range(len(points3d_img) - 1):
        #     cv2.line(img, tuple(points3d_img[i].astype(int)), tuple(points3d_img[i+1].astype(int)), color, 2)

    # Draw sweep area polygon between the boundaries.
    frame = np.zeros_like(img)
    polygon = np.vstack((np.array(points3d_left_img), np.array(points3d_right_img)[::-1])).astype(np.int32)
    check_flag = False
    if polygon.size == 0:
        check_flag = True
        return check_flag
    if args.plot:
        cv2.fillPoly(frame, [polygon], color=color)  # Green polygon
        mask = frame.astype(bool)
        img[mask] = cv2.addWeighted(img, 0.5, frame, 0.5, 0)[mask]
    return check_flag



def EstimateCurvatureFromTrajectory(traj):
    traj = traj[:, :2]
    curvature = np.zeros(len(traj))

    for i in range(1, len(traj) - 1):
        x1, y1 = traj[i - 1]
        x2, y2 = traj[i]
        x3, y3 = traj[i + 1]

        # Vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        # Lengths
        L1 = np.linalg.norm(v1)
        L2 = np.linalg.norm(v2)
        L3 = np.linalg.norm(np.array([x3 - x1, y3 - y1]))

        # Signed area (using cross product)
        area_signed = 0.5 * ((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))

        if L1 > 0 and L2 > 0 and L3 > 0:
            curvature[i] = 4 * area_signed / (L1 * L2 * L3)

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return curvature

def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span, time_span)  # Time vector

    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=0)
    theta += theta0  # 手动加上初始角度

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=0)
    y = cumulative_trapezoid(v_y, t, initial=0)
    x += x0  # 手动加上初始位置
    y += y0

    return np.stack((x, y), axis=1)

def WriteImageSequenceToVideo(cam_images_sequence: list, filename):
    assert len(cam_images_sequence) >= 1, "No images to write to video."
    # Save the image sequence as video
    # Define the codec and initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(f"{filename}.mp4", fourcc, fps=2,
                                   frameSize=(cam_images_sequence[0].shape[1], cam_images_sequence[0].shape[0]))

    for img in cam_images_sequence:
        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    
    
def getMessage(prompt, image=None, args=None):
    if "llama" in args.model_path or "Llama" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]   
    return message

def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None):
    out = None
    if "llama" in args.model_path or "Llama" in args.model_path:
        image = Image.open(images).convert('RGB')
        message = getMessage(text, args=args)
        input_text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=2048)

        output_text = processor.decode(output[0])
        if "llama" in args.model_path or "Llama" in args.model_path:
            output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
        out = output_text

    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = getMessage(text, image=images, args=args)
        text_prompt = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # print(f"number of generated tokens: {len(generated_ids_trimmed[0])}")
        out = output_text[0]
        
    elif "llava" in args.model_path:
        conv_mode = "mistral_instruct"
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in text:
            if model.config.mm_use_im_start_end:
                text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
            else:
                text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
        else:
            if model.config.mm_use_im_start_end:
                text = image_token_se + "\n" + text
            else:
                text = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(images).convert('RGB')
        image_tensor = process_images([image], processor, model.config)[0]
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True,
                pad_token_id = tokenizer.eos_token_id,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        out = outputs
                
    # elif "gpt" in args.model_path:
    #     PROMPT_MESSAGES = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 *map(lambda x: {"image": x, "resize": 768}, images),
    #                 text,
    #             ],
    #         },
    #     ]
    #     if sys_message is not None:
    #         sys_message_dict = {
    #             "role": "system",
    #             "content": sys_message
    #         }
    #         PROMPT_MESSAGES.append(sys_message_dict)
    #     params = {
    #         "model": "gpt-4o-2024-11-20",
    #         "messages": PROMPT_MESSAGES,
    #         "max_tokens": 400,
    #     }
    #     result = client.chat.completions.create(**params)
    #     return result.choices[0].message.content
    
    return out

def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    if "llava" in args.model_path:
        prompt = f"""You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):

    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result

def DescribeOrUpdateIntent(obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):

    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
        
    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

        if "llava" in args.model_path:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

    return result

def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None, args=None):
    # assert len(obs_images) == len(obs_waypoints)

    scene_description, object_description, intent_description = None, None, None

    if args.method == "openemma":
        scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
        print(f'Scene Description: {scene_description}')
        print(f'Object Description: {object_description}')
        print(f'Intent Description: {intent_description}')

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    
    # aka how much to turn the steering wheel. In the paper it's
    # the array K = {k_t}. It's multiplied by 100
    obs_curvatures = obs_curvatures * 100
    
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

    if args.method == "openemma":
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene_description}. 
        The identified critical objects are {object_description}. 
        The car's intent is {intent_description}. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    for rho in range(3):
        result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer, args=args)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    return result, scene_description, object_description, intent_description
