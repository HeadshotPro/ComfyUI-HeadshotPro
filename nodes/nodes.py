import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import subprocess
import time
import shutil
import os
import hashlib
import random
from diffusers.utils import load_image
from .utils import * 
import cv2
from facenet_pytorch import MTCNN
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import facer
from torchvision.transforms.functional import to_pil_image
import comfy

def download_weights_pget(url, dest, unpack=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if unpack:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class ReplaceTransparentByWhite:
    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "image": ("IMAGE",),
            },
        }
    

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_background"
    CATEGORY = "HeadshotPro"

    def replace_background(self, image):
        pil_image = images_to_pil_images(image)[0][0]
        pil_image = pil_image.convert("RGBA")

        # Create a new image with white background
        white_bg_image = Image.new("RGBA", pil_image.size, "WHITE")
        # Paste the original image onto the white background using its alpha channel as a mask
        white_bg_image.paste(pil_image, (0, 0), pil_image)
        # Convert back to RGB to drop the alpha channel
        final_image = white_bg_image.convert("RGB")
        
        tensor = torch.from_numpy(np.array(final_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (tensor,)

       

class RandomValueFromList:
    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "list_str": ("STRING", {"multiline": True, "default": ""}),               
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("INT",)
    OUTPUT_NODE = True
    RETURN_NAMES = ("int",)
    FUNCTION = "get_random_string"
    CATEGORY = "HeadshotPro"

    def get_random_string(self, list_str):
        # Turn string list of line breaks into array
        list_items = list_str.split(',')
        # Remove any empty strings that may exist due to splitting
        list_items = [item.strip() for item in list_items if item.strip()]  # Remove leading/trailing whitespace
        # Randomly select a string from the list
        choice = random.choice(list_items)
        # Check if the choice can be converted to an int or float, then return the appropriate type
        try:
            int_choice = int(choice)
            return (int_choice,)  # Return as int if possible
        except ValueError:
            try:
                float_choice = float(choice)
                return (float_choice,)  # Return as float if possible
            except ValueError:
                return (choice,)  # Return as string if it's neither int nor float

class DownloadFluxLora:
    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "url": ("STRING", {"multiline": False, "default": ""}),               
            },
            "optional": {
                "force_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    OUTPUT_NODE = True
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_lora"
    CATEGORY = "HeadshotPro"

    def download_lora(self, url, force_download=True):
        dest = os.path.join(os.path.dirname(__file__), "../../../models/loras")
        # Generate a unique ID from the URL
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        file_extension = url.split('?')[0].split('.')[-1]  # Extract file extension from URL

        if file_extension == "zip":
            file_path = f"{dest}/{url_hash}.zip"  # Use the hash in the zip file name
        elif file_extension == "tar":
            file_path = f"{dest}/{url_hash}.tar"  # Use the hash in the tar file name
        else:
            raise ValueError("Unsupported file format")
        
        dreambooth_model_dest = f"{dest}/{url_hash}"  # Use the hash in the model folder name

        if force_download or not os.path.exists(dreambooth_model_dest):
            if os.path.exists(dreambooth_model_dest):
                shutil.rmtree(dreambooth_model_dest)
            # Download the file to file_path instead of dest
            if os.path.exists(file_path):
                os.remove(file_path)
            download_weights_pget(url, file_path, unpack=True)  # Assuming you want to unpack
            # Ensure the directory exists
            subprocess.check_call(["mkdir", "-p", dreambooth_model_dest], close_fds=False)
            # Extract the file based on its format
            if file_extension == "zip":
                subprocess.check_call(
                    ["unzip", "-o", file_path, "-d", dreambooth_model_dest], close_fds=False
                )
                # Find the .safetensors file in the output/flux_train_replicate/ directory
                safetensors_file = None
                for root, dirs, files in os.walk(os.path.join(dreambooth_model_dest, "output/flux_train_replicate")):
                    for file in files:
                        if file.endswith(".safetensors"):
                            safetensors_file = os.path.join(root, file)                            
                            print(safetensors_file)
                            break
                    if safetensors_file:
                        break
                
                if safetensors_file:
                    # Move the .safetensors file up to the dest folder and rename it
                    shutil.move(safetensors_file, os.path.join(dest, f"{url_hash}.safetensors"))

                else:
                    print("No .safetensors file found in the output/flux_train_replicate/ directory")
                
            # Delete the .zip file and extraction folder
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(dreambooth_model_dest):
                shutil.rmtree(dreambooth_model_dest)
                
            elif file_extension == "tar":
                subprocess.check_call(
                    ["tar", "-xvf", file_path, "-C", dreambooth_model_dest], close_fds=False
                )
            print(f"File downloaded and extracted to {dreambooth_model_dest}")
        else:
            print(f"Model already exists at {dreambooth_model_dest}, skipping download.")
        return (f"{url_hash}.safetensors",)

class DownloadDreamboothCheckpointFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "url": ("STRING", {"multiline": False, "default": ""}),               
            },
            "optional": {
                "force_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    OUTPUT_NODE = True
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_file_and_unzip"
    CATEGORY = "HeadshotPro"

    def download_file_and_unzip(self, url, force_download=True):
        dest = os.path.join(os.path.dirname(__file__), "../../../models/diffusers")
        # Generate a unique ID from the URL
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
       
        if file_extension == "zip":
            file_path = f"{dest}/{url_hash}.zip"  # Use the hash in the zip file name
        elif file_extension == "tar":
            file_path = f"{dest}/{url_hash}.tar"  # Use the hash in the tar file name
        else:
            raise ValueError("Unsupported file format")
        
        dreambooth_model_dest = f"{dest}/{url_hash}"  # Use the hash in the model folder name

        if force_download or not os.path.exists(dreambooth_model_dest):
            if os.path.exists(dreambooth_model_dest):
                shutil.rmtree(dreambooth_model_dest)
            # Download the file to file_path instead of dest
            if os.path.exists(file_path):
                os.remove(file_path)
            download_weights_pget(url, file_path, unpack=True)  # Assuming you want to unpack
            # Ensure the directory exists
            subprocess.check_call(["mkdir", "-p", dreambooth_model_dest], close_fds=False)
            # Extract the file based on its format
            if file_extension == "zip":
                subprocess.check_call(
                    ["unzip", "-o", file_path, "-d", dreambooth_model_dest], close_fds=False
                )
            elif file_extension == "tar":
                subprocess.check_call(
                    ["tar", "-xvf", file_path, "-C", dreambooth_model_dest], close_fds=False
                )
            print(f"File downloaded and extracted to {dreambooth_model_dest}")
        else:
            print(f"Model already exists at {dreambooth_model_dest}, skipping download.")
        return (url_hash,)

class GetCannyFromPoseAndFace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "face_image": ("IMAGE",),
                "pose_image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "get_canny_from_pose_and_face"
    CATEGORY = "HeadshotPro"

    def images_to_pil_images(self, images: torch.Tensor) -> list[Image.Image]:
        pil_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)
        return (pil_images,)

    def get_canny_from_pose_and_face(self, face_image, pose_image):
        MODELS = [
            "mobilenet0.25_Final.pth",
            "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
        ]

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weights_dir = os.path.join(base_dir, "ComfyUI-HeadshotPro/weights")

        for model in MODELS:
            model_path = os.path.join(weights_dir, model)
            if not os.path.exists(model_path):
                # Call download_weights function here
                print(model, model_path)
                download_file(model, weights_dir)
                # Replace "URL_TO_DOWNLOAD_MODEL" with the actual URL to download the model  
        device = comfy.model_management.get_torch_device()
        print(device)
        mtcnn = MTCNN(image_size=320,keep_all=True, device=device)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        face_parser = facer.face_parser('farl/lapa/448', model_path=os.path.join(base_dir,"ComfyUI-HeadshotPro/weights/face_parsing.farl.lapa.main_ema_136500_jit191.pt"), device="cpu")
        face_detector_seg = facer.face_detector('retinaface/mobilenet', model_path=os.path.join(base_dir,"ComfyUI-HeadshotPro/weights/mobilenet0.25_Final.pth"), device="cpu")
        face_segment_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        face_segment_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to('cuda')
        face_segment_model.to('cuda')
        print("Setting up models done")
        try:

            # init_pose = load_image(pose_image)
            init_pose = self.images_to_pil_images(pose_image)[0][0]
            init_pose = init_pose.resize((512, 768), Image.BILINEAR)
            init_pose.save(os.path.join(base_dir,'ComfyUI-HeadshotPro/devfiles/init_pose.png'))

            canny_control_net = Image.new('RGB', (init_pose.width, init_pose.height), 'white')
            canny_hair_control_net = Image.new('RGB', (init_pose.width, init_pose.height), 'white')
            
            # 1. Load and resize both images
            print('# 1. Load and resize both images')
            if face_image is not None:
                # image = load_image(face_image)       
                image = self.images_to_pil_images(face_image)[0][0]
                image = image.resize((512, 512), Image.BILINEAR)
                image = image.convert('RGB')    

            # 2. Detect the faces in each image, calculate measurments 
            print(' # 2. Detect the faces in each image, calculate measurments ')
            if face_image is not None:
                p_boxes, _ = mtcnn.detect(init_pose)
                c_boxes, _ = mtcnn.detect(image)
                print("Finished detecting p and c")
                p_x, p_y, p_x2, p_y2 = p_boxes[0]
                c_x, c_y, c_x2, c_y2 = c_boxes[0]
                p_width, p_height = [p_x2 - p_x, p_y2 - p_y]
                c_width, c_height = [c_x2 - c_x, c_y2 - c_y]

                # 3. Calculate new dimensions
                scaling_factor_height = int(p_height) / int(c_height)
                scaling_factor_width = int(p_width) / int(c_width)
                scaling_factor = scaling_factor_height #(scaling_factor_height + scaling_factor_width) / 2
                # scaling_factor = (scaling_factor_height + scaling_factor_width) / 2
                new_width = int(image.width * scaling_factor)
                new_height = int(image.height * scaling_factor)

                # 4. Resize the canny image
                image = image.resize((new_width, new_height), Image.BILINEAR)
                image.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/temp/canny_input.png"))
            
                # 5. Get face segmentation
                print('# 5. Get face segmentation')
                mask = None      
                try:          
                    inputs = face_segment_processor(images=image, return_tensors="pt").to('cuda')
                    outputs = face_segment_model(**inputs)
                    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

                    # resize output to match input image dimensions
                    upsampled_logits = nn.functional.interpolate(logits,
                                    size=image.size[::-1], # H x W
                                    mode='bilinear',
                                    align_corners=False)
                    labels = upsampled_logits.argmax(dim=1)[0]
                    desired_label_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17]  # Example label indices to keep

                    # Create a mask where only the desired labels are True
                    print('# Create a mask where only the desired labels are True')
                    label_mask = torch.zeros_like(labels, dtype=torch.bool)
                    for label_index in desired_label_indices:
                        label_mask |= (labels == label_index)

                    # Apply the mask to the labels, setting all other labels to 0 (background)
                    filtered_labels = torch.where(label_mask, labels, torch.tensor(0, dtype=labels.dtype))

                    # Move to CPU and convert to numpy for visualization
                    filtered_labels_viz = filtered_labels.cpu().numpy()
                    binary_face_segmentation= filtered_labels_viz
                    # 6. Generate an alpha mask for the face segmentation, and make into a mask 
                    print(' # 6. Generate an alpha mask for the face segmentation, and make into a mask ')
                    bool_arr = binary_face_segmentation == 1
                    alpha = generate_alpha_mask_new(binary_face_segmentation, desired_label_indices, 0)
                    mask = Image.fromarray((alpha * 255).astype(np.uint8))
                    mask.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/devfiles/mask.png"))
                except:
                    binary_face_segmentation = get_face_segmentation(os.path.join(base_dir,"ComfyUI-HeadshotPro/temp/canny_input.png"),face_detector_seg,face_parser, binary=False, segments_of_interest=[1,2,3,4,5,6,7,8,9,10]).squeeze()
                    alpha = generate_alpha_mask(binary_face_segmentation, 0)
                    mask = Image.fromarray((alpha * 255).astype(np.uint8))
                    mask.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/devfiles/mask.png"))


                # 7. Make white and compound background
                print('# 7. Make white and compound background')
                white_background = Image.new('RGBA', image.size, (0, 0, 0, 255))
                composite_image = Image.composite(image, white_background, mask)
                white_background.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/devfiles/white.png"))
                composite_image.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/devfiles/comp.png"))

                # 8. Make padded image
                print(' # 8. Make padded image')
                padded_image = Image.new('RGBA', (512, 768), (0, 0, 0, 255))
                paste_x = p_x - (c_x*scaling_factor)
                paste_y = p_y - (c_y*scaling_factor)

                
                
                # 9. Paste padding back
                print(' # 9. Paste padding back')
                padded_image.paste(composite_image, (int(paste_x), int(paste_y)))
                

                # 10. Now get canny detection 
                print('  # 10. Now get canny detection ')
                # canny_image = np.array(padded_image)
                # low_threshold = 40
                # high_threshold = 100
                # canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
                # canny_image = canny_image[:, :, None]
                # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                # canny_control_net = Image.fromarray(canny_image)
                # canny_control_net.save(os.path.join(base_dir,"ComfyUI-HeadshotPro/devfiles/canny_control.png"))

                tensor = torch.from_numpy(np.array(padded_image).astype(np.float32) / 255.0).unsqueeze(0)

                # tensor = torch.from_numpy(np.array(canny_control_net).astype(np.float32) / 255.0).unsqueeze(0)

            # 12. Return all control nets
            print('# 12. Return all control nets')
            return (tensor, )
        except:
            # Return empty black image
            padded_image = Image.new('RGBA', (512, 768), (0, 0, 0, 255))
            tensor = torch.from_numpy(np.array(padded_image).astype(np.float32) / 255.0).unsqueeze(0)
            return (tensor, )




