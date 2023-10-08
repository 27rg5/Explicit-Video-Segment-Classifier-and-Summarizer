"""
Note: raw videos should be like this 
Work Folder
    |
    └vid_samples
        ├explicit
        └non-explicit

This script will generate files in the following 
Work Folder
    |
    └audio
        ├explicit
        └non-explicit
"""

from pathlib import Path
import moviepy.editor as mp 

#Modify This Variable if the number of samples are too small
THRESHOLD = 150

ROOT = Path(__file__).parent 

VID_EX = Path(ROOT, "vid_samples", "explicit")
VID_NEX = Path(ROOT, "vid_samples", "non_explicit")
AUD = Path(ROOT, "audio")
AUD_EX = Path(AUD, "explicit")
AUD_NEX = Path(AUD, "non_explicit")

AUD_EX.mkdir(parents=True, exist_ok=True)
AUD_NEX.mkdir(parents=True, exist_ok=True)

def extract_audio() -> None:
    #Audio Extraction, explicit videos
    for vid in VID_EX.iterdir():        
        fn = chr(95).join(vid.name.split(chr(46))[:-1])
        clip = mp.VideoFileClip(str(vid))
        
        #If duration is over threshold this sample has incorrect timestamp
        if clip.duration > THRESHOLD:
            continue
        else:
            clip.audio.write_audiofile(Path(AUD_EX, f"{fn}.wav"))

    #Movie to Audio Generation of Non-explicit videos
    for vid in VID_NEX.iterdir():
        fn = chr(95).join(vid.name.split(chr(46))[:-1])
        clip = mp.VideoFileClip(str(vid))
        #If duration is over 5 min this sample has incorrect timestamp
        if clip.duration > THRESHOLD:
            continue
        else:
            clip.audio.write_audiofile(Path(AUD_NEX, f"{fn}.wav"))        

if __name__ == "__main__":
    extract_audio()