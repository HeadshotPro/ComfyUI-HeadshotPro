from scipy.ndimage import distance_transform_edt
import numpy as np
import facer
import torch
from PIL import Image
import subprocess


def images_to_pil_images(images: torch.Tensor) -> list[Image.Image]:
    pil_images = []
    for image in images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        pil_images.append(img)
    return (pil_images,)

def generate_alpha_mask_new(face_segment, desired_label_indices, transition_width=10):
    """
    Given a face segmentation, this method creates an alpha mask including the desired labels,
    which will be used to blend the base image and the upscaled image.

    Args:
        face_segment (np.array): The face segmentation as a numpy array with label indices.
        desired_label_indices (list): List of label indices to include in the mask.
        transition_width (int): The width of the transition zone for the mask.

    Return:
        np.array: The alpha mask
    """

    # Create a mask where only the desired labels are True
    label_mask = np.zeros_like(face_segment, dtype=bool)
    for label_index in desired_label_indices:
        label_mask |= (face_segment == label_index)

    # Calculate distance of each face pixel to non-face (background) pixel
    distance_transform = distance_transform_edt(~label_mask)

    # Create a smooth transition from 0 to 1 based on distance
    if transition_width == 0:
        transitioned_mask = np.where(distance_transform <= 0, 1, 0)
    else:
        transitioned_mask = 1 / (1 + np.exp((distance_transform - 0) / transition_width))

    # Normalize the mask to ensure values between 0 and 1
    transitioned_mask = (transitioned_mask - np.min(transitioned_mask)) / (np.max(transitioned_mask) - np.min(transitioned_mask))
    # transitioned_mask = 1 - transitioned_mask

    # Return finalized alpha mask 
    return transitioned_mask

def get_face_segmentation(image_pth,face_detector_seg,face_parser,binary=True, segments_of_interest=[2,3,4,5,7,8,9,10]):
    """
    Returns binary mask of face segmentation for the given image path. 


    Args: 
        image_pth (str): The path to the image
        face_detector_seg (FaceDetector): The face detector
        face_parser (FaceParser): The face parser

    Return:
        np.array: The binary mask of the face segmentation
    """
    image = facer.hwc2bchw(facer.read_hwc(image_pth)).to(device="cuda")  # image: 1 x 3 x h x w
    face_detector_seg.to("cuda")
    face_parser.to("cuda")
    with torch.inference_mode():
        faces = face_detector_seg(image)

    with torch.inference_mode():
        faces = face_parser(image, faces)
    image.to("cpu")
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    vis_seg_probs = seg_probs.argmax(dim=1)
    vis_seg_probs = vis_seg_probs.cpu().numpy()

    
    # Just want binary 
    if binary:
        vis_seg_probs[vis_seg_probs > 0] = 1
        vis_seg_probs = vis_seg_probs[0]
    else:
        # Get just specific parts of the faces 
        vis_seg_copy = np.zeros_like(vis_seg_probs) 
        for soi in segments_of_interest:
            vis_seg_copy[vis_seg_probs == soi] = 1
        vis_seg_probs = vis_seg_copy



    return vis_seg_probs

def generate_alpha_mask(face_segment, transition_width=10):
    """
    Given a face segmentation, this method creates an elliptical alpha mask,
    which will be used to blend the base image and the upscaled image.

    Args:
        face_segment (Image): The face segmentation

    Return:
        np.array: The alpha mask
    """

    # 1. Convert face segmentation into boolean array
    bool_arr = face_segment == 1

    # 2. Calculate distance of each face pixel to non-face (background) pixel
    distance_transform = distance_transform_edt(bool_arr)

    # 3. Define the distance at which you want the transition to start, and the width
    start_distance = 0  # Adjust this value as needed
    # transition_width = transition_width  # Adjust this value as needed

    # 4. Create a smooth transition from 0 to 1 based on distance
    if transition_width == 0:
        transitioned_mask = np.where(distance_transform <= start_distance, 1, 0)
    else:
        transitioned_mask = 1 / (1 + np.exp((distance_transform - start_distance) / transition_width))

    # 5. Normalize the mask to ensure values between 0 and 1
    transitioned_mask = (transitioned_mask - np.min(transitioned_mask)) / (np.max(transitioned_mask) - np.min(transitioned_mask))
    transitioned_mask = 1 - transitioned_mask
    alpha = transitioned_mask

    # 6. Return finalized alpha mask 
    return alpha     

def tensor_to_pil(tensor):
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.cpu().detach()
    # Convert to numpy array
    array = tensor.numpy()
    # If it's a 4D batch tensor, take the first image. Adjust as necessary.
    if array.ndim == 4:
        array = array[0]
    # Convert from (C, H, W) to (H, W, C) if necessary
    if array.shape[0] < array.shape[2]:
        array = array.transpose(1, 2, 0)
    # Handle grayscale images (2D arrays)
    if array.ndim == 2:
        mode = "L"
    else:
        mode = "RGB"
    # Convert to PIL Image
    pil_image = Image.fromarray(array.astype("uint8"), mode=mode)
    return pil_image

def download_file(path, dest):
    try:
        url = f"https://pub-c6593539e5e54a94b671581e33081130.r2.dev/custom_nodes/ComfyUI-HeadshotPro/{path}.tar"
        print(url)
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        print(f"Downloaded {url} to {dest} from custom endpoint")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download {path} from both primary and custom hosting.") from e

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)