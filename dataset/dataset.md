# YouTube360

## Data Availability Statement
We are committed to maintaining transparency and compliance in our data collection and sharing methods. Please note the following:

- **Publicly Available Data**: The data utilized in our studies is publicly available. We do not use any exclusive or private data sources.

- **Data Sharing Policy**: Our data sharing policy aligns with precedents set by prior works, such as [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), [Panda-70M](https://snap-research.github.io/Panda-70M/) 
, and [Miradata](https://github.com/mira-space/MiraData). Rather than providing the original raw data, we only supply the YouTube video IDs necessary for downloading the respective content.

- **Usage Rights**: The data released is intended exclusively for research purposes. Any potential commercial usage is not sanctioned under this agreement.

- **Compliance with YouTube Policies**: Our data collection and release practices strictly adhere to YouTube’s data privacy policies and fair of use policies. We ensure that no user data or privacy rights are violated during the process.

- **Data License**: The dataset is made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Clarifications

- The YouTube360 dataset is only available for informational purposes only. The copyright remains with the original owners of the video.
- All videos of the YouTube360 dataset are obtained from the Internet which is not the property of our institutions. Our institution is not responsible for the content or the meaning of these videos.
- You agree not to reproduce, duplicate, copy, sell, trade, resell, or exploit for any commercial purposes, any portion of the videos, and any portion of derived data. You agree not to further copy, publish, or distribute any portion of the YouTube360 dataset.

## Datadaset Construction Pipeline
### Data Collection and Filtering
- **Sources**: Content was gathered from YouTube 360 videos, focusing on panoramic videos in virtual city tours, wildlife documentaries, and VR game captures. We manually filter out low-quality 360 videos with bad polar patterns and frequent human appearances.

### Data Processing Pipeline
- **Shot Segmentation**: Videos were *downsampled by x2*, segmented into 100-frame, fps20 clips with TransNetV2 and FFmpeg. If a video exceeds 40000 frames, we further downsample x3 and keep the first 20000 frames.
- **Motion Filtering**: Panoflow was used to filter clips based on motion.
- **Text Annotation**: VideoLLaMa2 annotated clips with content information.
- **Content Filtering**: Remove semantic duplicates based on text captions.

## Dataset Overview
### YouTube360
The `YouTube360` dataset consists of 9557 video clips curated from online 360 videos. Each data sample includes metadata on YouTubeID, videoID, captions, timestamps, originalFPS and TotalFrames. Please note the time duration of each clip is x2 times longer as we downsample each frame by x2. 

### Data Fields
The [`dataset/youtube360.csv`](youtube360.csv) file contains the following columns:
- **youtubeid**: The YouTube ID of the original video, e.g., 2Lq86MKesG4. The source video url can found at  `https://www.youtube.com/watch?v={youtubeid}`.
- **videoid**: The original video ID and corresponding splitting ID, with a format such as `2Lq86MKesG4_clip_0_019`.
- **caption**: Text caption summarizing each video clip.
- **tstart**: The start time of the video segment in seconds in original fps.
- **tend**: The end time of the video segment in original fps.
- **fps**: The fps of the original video.
- **totalframes**: The total number of frames in original video.

### Example Data Entry
| YouTubeID | VideoID | Caption | tStart | tEnd | FPS | TotalFrames |
|------|---------------| -----------| ------------|----------|-----|----------|
| 2Lq86MKesG4 | 2Lq86MKesG4_clip_0_019 | "a busy city street with cars, buses, and tall buildings. The street is wet, and the video is taken from a car's point of view." | 690.9 | 700.9 | 20.0| 26450.0 |

## License
The `YouTube360` dataset is available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please ensure proper attribution when using the dataset in research or other projects.


## Citation
If you use `YouTube360` in your research, please cite it as follows:

```markdown
@article{tan2024imagine360,
  title={Imagine360: Immersive 360 Video Generation from Perspective Anchor},
  author={Tan, Jing and Yang, Shuai and Wu, Tong and He, Jingwen and Guo, Yuwei and Liu, Ziwei and Lin, Dahua},
  journal={arXiv preprint arXiv:2412.03552},
  year={2024}
}
```
