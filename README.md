# Long-term video motion temporal super resolution
## 1. Introduction:
This research considered two tasks:  
### a.Long-term video temporal super resolution</br>
>![Image of TSR](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/Temporal_super_resolution.png)</br>

| Real World Ground Truth |  | Video with low temporal resolution |  | Recovered Video |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| ![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/super_resolution_gt.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/tsr_capture.png) | ![blk_gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/super_resolution_blk_gt.gif) | ![tsr_dnn](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/tsr_dnn.png) | ![pred](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/super_resolution_pred.gif) |

### b.Long-term video interpolation</br>
>![Image of VI](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/Video_interpolation.png) </br>

| Original Video |  | Input Video with missing frames |  | Recovered Video |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| ![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/video_interpolation_gt.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/vi_damaged.png) | ![blk_gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/video_interpolation_blk_gt.gif) | ![tsr_dnn](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/vi_dnn.png) | ![pred](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Introduction/video_interpolation_pred.gif) |
</br>
</br>

## 2. Problem Formulation:
### a.Long-term video temporal super resolution</br>
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Formulation/Temporal_super_resolution_formulation.png)
### b.Long-term video interpolation</br>
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/Formulation/Video_interpolation_formulation.png)
</br>
</br>
## 3. Basic Model:
### a.Long-term video temporal super resolution</br>
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/model/temporal_super_resolution_model.png)
### b.Long-term video interpolation</br>
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/model/video_interpolation.png)
</br>
</br>
## 3. Basic Model Result:
### a.Long-term video temporal super resolution</br>
temporal super resolution rate as 4(green frames is given, red frames are missing)</br>

| Methods | Running | boxing | clapping | waving |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Ground Truth | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person21_running_d2_118_gt.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_boxing_d4_104_gt.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handclapping_d2_116_gt.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handwaving_d3_224_gt.gif) |
| Ours | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person21_running_d2_118_pred.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_boxing_d4_104_pred.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handclapping_d2_116_pred.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handwaving_d3_224_pred.gif) |
| Niklaus et al. | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person21_running_d2_118_soa.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_boxing_d4_104_soa.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handclapping_d2_116_soa.gif) | ![captured](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/super_resolution_result/person22_handwaving_d3_224_soa.gif) |

### b.Long-term video interpolation</br>
Given first 5 frames and last 5 frames, we can interpolate 10 frames in the middle</br>
(green frames is given, red frames are missing)</br></br>
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/video_interpolation/person17_walking_d1_uncomp.365-375_pred.gif)
![gt](https://github.com/Xharlie/motion_temporal_super_resolution/blob/master/site-content/video_interpolation/person19_boxing_d2_uncomp.1-11_pred.gif)
