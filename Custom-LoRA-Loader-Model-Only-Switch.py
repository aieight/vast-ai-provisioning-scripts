import torch
import comfy.sd
import comfy.utils
from comfy.model_patcher import ModelPatcher
import folder_paths

class CustomLoRALoader:
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative."
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative."
                }),
                "use_lora_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, the LoRA will be applied to the diffusion model."
                }),
                "use_lora_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, the LoRA will be applied to the CLIP model."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a LoRA model with an option to apply modifications to both the diffusion "
        "and CLIP models, or disable modifications to either."
    )

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, use_lora_model, use_lora_clip):
        if (not use_lora_model and not use_lora_clip) or (strength_model == 0 and strength_clip == 0):
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model if use_lora_model else None, 
            clip if use_lora_clip else None, 
            lora, 
            strength_model if use_lora_model else 0, 
            strength_clip if use_lora_clip else 0
        )

        return (model_lora if use_lora_model else model, clip_lora if use_lora_clip else clip)

# Register the node for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CustomLoRALoader": CustomLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomLoRALoader": "LoRA Loader (Enable/Disable Model & Clip)[custom]",
}
