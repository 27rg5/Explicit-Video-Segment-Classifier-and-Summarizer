import av
import os
import pdb
import moviepy
import logging
import sys
import numpy as np
from PIL import Image
# from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForCausalLM
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import warnings

warnings.filterwarnings("ignore")

processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")

logging.getLogger('moviepy').setLevel(logging.ERROR)

# set seed for reproducability
np.random.seed(45)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    #6 sec video: 180 frames seg_len=180, converted_len=24
    #2 sec video: 60 frames seg_len=60, converted_len=24
    converted_len = int(clip_len * frame_sample_rate)
    upper_bound = seg_len - converted_len - 1 if seg_len > converted_len else seg_len - 1
    start_idx = np.random.randint(0, upper_bound)
    end_idx = start_idx + converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def summarize(base_path, model, processor, device):
    concat = ""
    model = model.to(device)
    file_ = base_path.split('/')[-1]
    path = '/'.join(elem for elem in base_path.split('/')[:-1])
    iterations = 10
    for i in range(iterations):

        try:
            container = av.open(base_path)
            
            num_frames = model.config.num_image_with_embedding
            indices = sample_frame_indices(
                clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
            )
            frames = read_video_pyav(container, indices)

            pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values.to(device)
            #import pdb;pdb.set_trace()
            generated_ids = model.generate(pixel_values=pixel_values, max_length=80)
    #         print(i, processor.batch_decode(generated_ids, skip_special_tokens=True))
            concat = concat + " " + processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
            container.close()
        except Exception as e:
            print(f"Error {e} for {base_path}")
            continue
            

    return concat

#summarize(r"C:\Users\ragha\Downloads\stitched_videos\Lord.of.War\war1")