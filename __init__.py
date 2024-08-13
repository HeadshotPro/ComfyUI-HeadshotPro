from .nodes.nodes import *

NODE_CLASS_MAPPINGS = { 
    "[HSP] Download Dreambooth Checkpoint": DownloadDreamboothCheckpointFromUrl,
    "[HSP] Get Random Value From List": RandomValueFromList,
    "[HSP] Load Canny Pose Face": GetCannyFromPoseAndFace,
    "[HSP] Transparent to White Background": ReplaceTransparentByWhite,
    "[HSP] Download Flux Lora": DownloadFluxLora
}
    
print("\033[34mComfyUI HeadshotPro Nodes: \033[92mLoaded\033[0m")