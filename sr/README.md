# Instructions for 360 Video Super Resolution
## VEnhancer
### Installation
Please follow the instructions in the original repo: [VEnhancer](https://github.com/Vchitect/VEnhancer) for installation.  We recommend using its version 2 checkpoint: [venhancer_v2.pt](https://huggingface.co/jwhejwhe/VEnhancer/resolve/main/venhancer_v2.pt).

### Modifications for 360 close-loop
Put the panorama-related utils under VEnhancer folder and replace the `VEnhancer/video_to_video/video_to_video_model.py` file with our modified version.

```
cp -r Imagine360/src/utils VEnhancer/
cp Imagine360/sr/video_to_video_model.py VEnhancer/video_to_video/video_to_video_model.py
cp Imagine360/sr/inference_utils.py VEnhancer/inference_utils.py
cp Imagine360/sr/enhance_a_video.py VEnhancer/enhance_a_video.py
```
### Inference
```
python enhance_a_video.py \
--version v2 \
--up_scale 2 --target_fps 8 --noise_aug 200 --s_cond 8 \
--solver_mode 'fast' --steps 15 \
--filename_as_prompt \
--input_path path_to_input_video \
--save_dir 'sr_outputs' 
```

