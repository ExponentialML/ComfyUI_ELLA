import os
import torch
import comfy

import folder_paths
from folder_paths import supported_pt_extensions, models_dir, folder_names_and_paths
from .ella_model.model import ELLA, T5TextEmbedder

folder_names_and_paths["ella"] = ([os.path.join(models_dir, "ella")], supported_pt_extensions)
folder_names_and_paths["t5_model"] = ([os.path.join(models_dir, "t5_model")],[])

for i, f_path in enumerate([folder_names_and_paths["t5_model"], folder_names_and_paths["ella"]]):
    f_path = f_path[0][0]

    if i == 0:
        f_path = f_path + "/flan-t5-xl"
        
    if not os.path.exists(f_path):
        os.makedirs(f_path, exist_ok=True)

class LoadElla:
    def __init__(self):
        self.device = comfy.model_management.text_encoder_device()
        self.dtype = comfy.model_management.text_encoder_dtype()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ella_model": (folder_paths.get_filename_list("ella"),),
                              "t5_model": (os.listdir(folder_names_and_paths["t5_model"][0][0]),),
                              }}

    RETURN_TYPES = ("ELLA",)
    FUNCTION = "load_ella"

    CATEGORY = "ella/loaders"

    def load_ella(self, ella_model, t5_model):
        t5_path = os.path.join(models_dir, 't5_model', t5_model)
        ella_path = os.path.join(models_dir, 'ella', ella_model)
        t5_model = T5TextEmbedder(t5_path).to(self.device, self.dtype)
        ella = ELLA().to(self.device, self.dtype)

        ella_state_dict = comfy.utils.load_torch_file(ella_path)

        ella.load_state_dict(ella_state_dict)

        return ({"ELLA": ella, "T5": t5_model}, )

class ELLATextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}), 
                "manual_timestep": ("INT", {"default": 0}, ),
                "ella": ("ELLA", ),
            },
            "optional": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, text, ella: dict, manual_timestep, sigmas=None, model=None):
        ella_dict = ella
        ella: ELLA = ella_dict.get("ELLA")
        t5: T5TextEmbedder = ella_dict.get("T5")
        cond = t5(text)

        if model is None and sigmas is not None:
            raise ValueError("ELLATextEncode: model must be provided if sigmas are provided")
        if model is not None:
            if sigmas is None:
                raise ValueError("ELLATextEncode: sigmas must be provided if model is provided")
            ella_conds = []
            num_sigmas = len(sigmas)
            print(f"creating ELLA conds for {num_sigmas} sigmas")

            for i, sigma in enumerate(sigmas):
                timestep =  model.model.model_sampling.timestep(sigma)
                cond_ella = ella(cond, timestep)

                # Calculate start and end percentages based on the position of sigma in the batch
                start = (i / num_sigmas) # Start percentage is calculated based on the index
                end = ((i + 1) / num_sigmas) # End percentage is calculated based on the next index

                cond_ella_dict = {
                    "start_percent": start,
                    "end_percent": end
                }
                ella_conds.append([cond_ella, cond_ella_dict])
    
            return(ella_conds,)
        
        else:
            print("No model and sigmas provided, using manual timestep for ELLA conditioning")
            manual_timestep = torch.tensor(manual_timestep)
            cond_ella = ella(cond, manual_timestep)
        return ([[cond_ella, {"pooled_output": cond_ella}]], ) # Output twice as we don't use pooled output
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadElla": LoadElla,
    "ELLATextEncode": ELLATextEncode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadElla": "Load ELLA Model",
    "ELLATextEncode": "ELLA Text Encode (Prompt)",
}
