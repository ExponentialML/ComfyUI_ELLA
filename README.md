# ComfyUI_ELLA
ComfyUI Implementaion of ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment

![image](https://github.com/ExponentialML/ComfyUI_ELLA/assets/59846140/dd7ce18d-5cb3-458e-b32f-18d39baf4629)


> [!NOTE]  
> As per the ELLA developers / researchers, only the SD 1.5 checkpoint is released.

# Quick Start Guide

## Models

These models must be placed in the corresponding directories under `models`.

Example: `ComfyUI/models/ella/model_file`

1. Place the ELLA Model under a new folder `ella`: https://huggingface.co/QQGYLab/ELLA/blob/main/ella-sd1.5-tsc-t5xl.safetensors

2. Create a folder called `t5_model`. Navigate to that folder (you must be in that directory), and `git clone https://huggingface.co/google/flan-t5-xl` to download the t5 model.

3. If you don't wish to use git, you can dowload each indvididually file manually by creating a folder `t5_model/flan-t5-xl`, then download every file from [here](https://huggingface.co/google/flan-t5-xl/tree/main), although I recommend `git` as it's easier.

In summary, you should have the following model directory structure:

- `ComfyUI/models/ella/ella-sd1.5-tsc-t5xl.safetensors`
- `ComfyUI/models/t5_model/flan-t5-xl/all_downloaded_t5_models`

# Installation

To install, simply navigate to `custom_nodes`, and inside that directory, do `git clone https://github.com/ExponentialML/ComfyUI_ELLA.git`

To get started quickly, a workflow is provided in the workflow directory.

# Usage

## ELLA Loader
![image](https://github.com/ExponentialML/ComfyUI_ELLA/assets/59846140/c137008d-64ff-4252-902b-77c43754d70d)

- **ella_model**: The path to the ella checkpoint **file**.
- **t5_model**: The path to the t5 model **folder**.

## ELLA Text Encode

![image](https://github.com/ExponentialML/ComfyUI_ELLA/assets/59846140/685221ac-b6b9-49c0-81cd-255ed32addc2)

- **ella**: The loaded model using the ELLA Loader.
- **text**: Conditioning prompt. All weighting and such should be 1:1 with all condiioning nodes.
- **sigma**: The required sigma for the prompt. It must be the same as the KSampler settings. Without the workflow, initially this will be a float. You can simply right click the node, `convert sigma to input`, then use the `Get Sigma` node.

# Support

All conditioning nodes should be supported, as well as prompt weighting and ControlNet. 

![image](https://github.com/ExponentialML/ComfyUI_ELLA/assets/59846140/18bb28e4-b886-4c24-9a72-e1ba7dc46998)

# Attribution

Thanks to the following for open sourcing. Please follow their respective licensing.

- ELLA: https://github.com/TencentQQGYLab/ELLA

- Diffusers (borrowed timestep modules): https://github.com/huggingface/diffusers
