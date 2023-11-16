# Explicit Video Segment Classifier and Summarizer
To install whisper strictly use 'pip install git+https://github.com/openai/whisper.git'

python version = 3.10.0

In this project our objective is to take a video and extract the smaller segments of this video that contain explicit content and give a natural language summary of those video segments. We took a subset of two public datasets. For faster training we encoded the audio and language data in the form of audio and spectrogram encodings. The hierarchy of our data is

    - cls_data
      - processed_data
        - encoded_videos
          - explicit
            - Video directory
              - audio_encs (Has a file containing audio modality input)
              - spectro_encs (Has a file containing video modality input)

          - non_explicit
            (Same structure as above)

        - non_encoded_videos
          (mp4 files)

      - stitched_videos
        (mp4 files obtained from concatenating smaller video clips. Used during demo)

To install create a conda environment of python version 3.10.0 strictly
Install whisper strictly use `pip install git+https://github.com/openai/whisper.git`
Then run `pip install -r requirements.txt`

Below are the diagrams of our pipeline

<p align="center">
     <img src="https://github.com/27rg5/Explicit-Video-Segment-Classifier-and-Summarizer/blob/master/pipeline1.jpg" alt="Pipeline">
     <img src="https://github.com/27rg5/Explicit-Video-Segment-Classifier-and-Summarizer/blob/master/pipeline2.jpg" alt="Attention Mechanism">
</p>

To train the model run
```
python -W ignore main.py --n_epochs 20 --learning_rate 1e-3 --optimizer_name SGD  --root_dir ~/cls_data --language_model_name distilbert-base-uncased --video_model_name slowfast_r50 --experiment_name sample_trimodal_test_run --batch_size 1 --print_every 10 --spectrogram_model_name resnet18
```

`root_dir` - The root directory of all data

`experiment_name` - The desired name where the model checkpoint, tensorboard logs and file containing validation video names will be saved (from this file the names will be loaded and the corresponding video shall be loaded from the encoded data)

`optimizer_name` - The desired optimizer of choice eg. SGD, Adam

`language_model_name` - The pretrained language model of choice which will be trained for language modality

`video_model_name` - The pretrained video classifier of choice which will be trained for video modality

`spectrogram_model_name` - The pretrained audio classifier of choice (CNN) which will be trained for audio modality

`print_every` - Number of iterations or batches after which the running loss will be printed

To run the evaluation for explicit vs non_explicit classifier
```
python eval.py --root_dir_path ~/cls_data --experiment_name sgd_lr_1e-3_macro_f1_with_seed_42_feats_200_200 --get_classified_list 
```

`root_dir_path` - The root directory of all data

`get_classified_list` - Optional parameter to get a csv file having video name, predicted label and actual ground truth label

To run the demo for the entire pipeline
```
 python demo.py --stitched_videos_path ~/cls_data/stitched_videos/ --experiment_name sgd_lr_1e-3_macro_f1_with_seed_42_feats_200_200 
```

`stitched_videos_path` - Directory where stitched videos are saved

Below is an example output from our pipeline

<p align="center">
    <img src="https://github.com/27rg5/Explicit-Video-Segment-Classifier-and-Summarizer/blob/master/results.jpeg" alt="Results">
</p>
