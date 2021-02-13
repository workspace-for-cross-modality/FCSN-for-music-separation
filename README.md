# Cycle-Refining Network (CRN) for Visually-guided Music Source Separation

Music source separation from a sound mixture remains a big challenge because there often exist heavy overlaps and interactions among similar music signals. In order to correctly separate mixed sources, we propose a novel Cycle-Refining Network (CRN) for visually-guided music source separation. With the guidance of visual features, the proposed CRN approach refines preliminarily separated music sources by minimizing the residual spectrogram which is calculated by removing separated music spectrograms from the original music mixture. The refining separation is repeated several times until the residual spectrogram becomes empty or leaves only noise. Extensive experiments are performed on three large-scale datasets, the MUSIC (MUSIC-21), the AudioSet, and the VGGSound. Our approach outperforms the state-of-the-art in all datasets, and both separation accuracies and visualization results demonstrate its effectiveness for solving the problem of overlap and interaction in music source separation.


## Audio-Vision Separation Results
The audio-visual results include three sub-parts: 2-mix samples, 3-mix samples, and real-mix duet samples. 
For each sample, we supply the following files: 
* music audio files
* object image files that corresponding to sound sources
* the spectrograms for every music audio files



### Visualization

Here, we visualize the experiment results as spectrograms of corresponding music audio files. 

And all the music audio source files can be found at the correspongding folders in /music files/--.

#### 2-mix samples
Sample 1：

|Detected Objects|<img src="https://github.com/avis-ma/music/blob/main/music_files/2-mix-separation/sample-1/Object_detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/Object_detection_2.png" width="150">
|---|---|---|
|Ground Truth Spectrogram|[<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/audio1_groundtruth_spectrogram.png" width="200">](https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/audio1_groundtruth.wav)|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/audio2_groundtruth_spectrogram.png" width="200">|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/spectrogram_mixture.png" width="200">||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/separated_2_spectrogram.png" width="200">|

Sample 2：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/Object_detection_2.png" width="150">
|---|---|---|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/audio2_groundtruth_spectrogram.png" width="200">|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/spectrogram_mixture.png" width="200">||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/separated_2_spectrogram.png" width="200">|


#### 3-mix samples

Sample 1：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_2.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_3.png" width="150" height="150">|
|---|---|---|---|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio2_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio3_groundtruth_spectrogram.png" width="200">|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/spectrogram_mixture.png" width="200">|||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_2_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_3_spectrogram.png" width="200">|

Sample 2：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_2.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_3.png" width="150" height="150">|
|---|---|---|---|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio2_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio3_groundtruth_spectrogram.png" width="200">|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/spectrogram_mixture.png" width="200">|||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_2_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_3_spectrogram.png" width="200">|

#### Real-mix duet samples

Sample 1:
|Frame|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/frame.png" width="150">||
|---|:-:|:-:|
|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/Object_detection_2.png" width="150" height="150">|
|Duet Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/duet_spectrogram_groundtruth.png" width="200">||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/separated_2_spectrogram.png" width="200">|

Sample 2:
|Frame|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/frame.png" width="150">||
|---|:-:|:-:|
|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/Object_detection_2.png" width="150" height="150">|
|Duet Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/duet_spectrogram_groundtruth.png" width="200">||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/separated_2_spectrogram.png" width="200">|

## Code Part
The README of our code can be seen at /code/CRN/README.md.
