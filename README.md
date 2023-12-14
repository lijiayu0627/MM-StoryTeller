### MM Storyteller -  A  multi-modal video captioner based on CLIP and GPT-2

This project builds a multi-modal model to narrate the story in the videos with natural language. It is based on [CLIP Captioning](https://arxiv.org/pdf/2111.09734.pdf), which bridges the embedding space of CLIP and generative language model to generate captions for images. In this project, I develop novel methods to generate captions for videos through modeling the image sequences with Transformer.  I evaluate the methods on the [YouTubeClips dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).


#### Usage 
1. `mkdir data` and put the downloaded dataset into the `data` folder

2. Convert videos into image frames

   `python preprocess/extract_frames.py` 

3. Set parameters and train models
    `python train_mean.py --only_prefix --out_dir ./msv_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40`

  `python train_seq.py --only_prefix --out_dir ./msv_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40`

### Methodology

#### Overview

The model first convert the input video into a sequence of images. Then it utilize CLIP vision encoder to represent images. It trains a projection module to align image embeddings to the embedding space of GPT-2 to serve as the prefix for caption generation. The framework is shown as the following image.

#### Video Preprocessing

Given an input video, the model converts it into image frames with a fixed sampling frequency. Then it encodes the images with the CLIP vision encoder. I propose two methods to generate the video embedding from image embeddings. The first way is to utilize the mean pooling to get an overall embedding of the video. The second way is to utilize a Transformer model to project the  $n\times m$embeedings of the image sequence into a $1\times m$ vector.

#### Caption generation

Given the video representation, I train a Transformer-based projector to project the video embedding vector into the representation space of GPT-2. Then I can use the frozen GPT-2 to generate the narration of the video.

### Evaluation Metrics
I evaluate the proposed video caption model with the following metrics: BLEU-4, CIDEr, Rouge-L and METEOR. The results are shown in the following table.

| Pre-processing | BLEU      | CIDEr     | METEOR    | ROUGE-L   |
| -------------- | --------- | --------- | --------- | --------- |
| Mean           | 0.523     | 0.544     | 0.302     | 0.640     |
| Sequence       | **0.537** | **0.596** | **0.327** | **0.658** |

### Examples

| Video                | 1    | 2    | 3    |
| ----------------------   | ----------- |----------- |----------- |
| Sample frame         | ![image](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e1.png) | ![image](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e2.png) | ![image](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e3.png) |
| Original Video       | [video 1](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e1.avi) | [video 2](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e2.avi) | [video 3](https://github.com/lijiayu0627/MM-StoryTeller/blob/main/images/e3.avi) |
| Generated Caption | a small cat jumps into a bag | people are traveling on a bus | a person is cutting a carrot |





