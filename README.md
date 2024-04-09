# ComfyUI_ELLA
ComfyUI Implementaion of ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment

![image](https://github.com/ExponentialML/ComfyUI_ELLA/assets/59846140/dd7ce18d-5cb3-458e-b32f-18d39baf4629)


> [!NOTE]  
> As per the ELLA developers, only the SD 1.5 checkpoint is released.

# Quick Start Guide

## Models

These models must be placed in the corresponding directories under `models`. For example: `ComfUI/models/ella`

1. Place the ELLA Model under a new folder `ella`: https://huggingface.co/QQGYLab/ELLA/blob/main/ella-sd1.5-tsc-t5xl.safetensors

2. Create a folder called `t5_model`. Navigate to that folder (you must be in that directory), and `git clone https://huggingface.co/google/flan-t5-xl` to download the t5 model. If you don't wish to use git, you can simply navigate to that folder, and download the corresponding model directory.

## Custom Node

To install, simply navigate to `custom_nodes` and `git clone https://github.com/ExponentialML/ComfyUI_ELLA.git`

## Extra Notes

All conditioning nodes and prompt weighting should work as intended.
