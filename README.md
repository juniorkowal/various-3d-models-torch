# Various 3D Models

This repository contains a collection of experimental 3D models, focused on adapting existing architectures to handle 3D data, e.g. medical data. Many successful architectures in other domains have not been extensively applied to this field, so I am experimenting with converting them to 3D and analyzing their performance.

## Overview

- **UNet3+**: The primary modification here is straightforward—replacing `Conv2D` with `Conv3D`.
  
- **MUNet**: Required the most significant changes, including shape modifications. I used `einops` to manage and transform the tensor shapes correctly for 3D data. This model uses a **Mixing Block/Attention** mechanism derived from the **MixFormer** architecture, which is originally written in PaddlePaddle. **MixFormer** itself is partially based on **SwinUNETR**, which is written in PyTorch. My modifications convert the 2D MUNet to a 3D version by incorporating some code from **SwinUNETR**.

- **MSwinUNETR**: A custom model that combines **SwinUNETR** with the **Mixing Block/Attention** from MUNet. Unfortunately, I haven’t achieved good results with this model yet.

- **LambdaUNet**: A lightweight version of UNet using the **Lambda Layer** from the LambdaNetworks paper. I integrated the 'lightweight' Lambda Layer in the bottleneck of the UNet, which significantly reduces the parameter count. From my testing, the performance drops only by 1-2% with this optimization.

Feel free to explore and experiment with the models. Note that this is highly experimental and still a work in progress.
