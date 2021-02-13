# Cycle-Refining Network (CRN) for Visually-guided Music Source Separation

Music source separation from a sound mixture remains a big challenge because there often exist heavy overlaps and interactions among similar music signals. In order to correctly separate mixed sources, we propose a novel Cycle-Refining Network (CRN) for visually-guided music source separation. With the guidance of visual features, the proposed CRN approach refines preliminarily separated music sources by minimizing the residual spectrogram which is calculated by removing separated music spectrograms from the original music mixture. The refining separation is repeated several times until the residual spectrogram becomes empty or leaves only noise. Extensive experiments are performed on three large-scale datasets, the MUSIC (MUSIC-21), the AudioSet, and the VGGSound. Our approach outperforms the state-of-the-art in all datasets, and both separation accuracies and visualization results demonstrate its effectiveness for solving the problem of overlap and interaction in music source separation.


## Audio-Vision Separation Results
The audio-visual results include three sub-parts: 2-mix samples, 3-mix samples, and real-mix duet samples. 

For each sample, we supply the following files: 
* Music audio files;
* Object detection images corresponding to the sound sources;
* Spectrograms corresponding to each audio file.

For 2-mix samples, there are two sound sources in the mixture to be separated. And the same setting is applied for 3-mix samples.
For real-mix duet samples, we randomly select duet samples in the MUSIC dataset to simulate the separation results in real scenes. 



### Visualization

Here, we visualize the experiment results as spectrograms and music audios. 

You can directly listen to the music audios by clicking the links below the spectrograms.

#### → *2-mix samples*

 - Sample 1：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/Object_detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/Object_detection_2.png" width="150">
|---|:-:|:-:|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/audio2_groundtruth_spectrogram.png" width="200">|
|Ground Truth Audio|[Audio_GT_1](https://drive.google.com/file/d/1Cgdnu8fjDP4TK-pQrxTiwNlxsXHLyaLK/view?usp=sharing)|[Audio_GT_2](https://drive.google.com/file/d/1FKOezbpS1NwIQDtvBusKWz8TqfXWtnh0/view?usp=sharing)|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/spectrogram_mixture.png" width="200">||
|Mixture Audio|[Audio_Mixture](https://drive.google.com/file/d/10trqGrPmsEev6fLVj4nxHpIixKD_hZ-q/view?usp=sharing)||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-1/separated_2_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/1rYijTDySHUbKV5ybErnXwur5qKqOwsMd/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/1nA0IFwXHhiNn662huCGkeCiMVnFOf4N7/view?usp=sharing)|

 - Sample 2：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/Object_detection_2.png" width="150">
|---|:-:|:-:|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/audio2_groundtruth_spectrogram.png" width="200">|
|Ground Truth Audio|[Audio_GT_1](https://drive.google.com/file/d/1F1ymD9t2-RdQuLJY9r-RAlDfyB1N9tc5/view?usp=sharing)|[Audio_GT_2](https://drive.google.com/file/d/1ErmxMR-pUa1fqm8-jYbj8k6sEB65_nFG/view?usp=sharing)|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/spectrogram_mixture.png" width="200">||
|Mixture Audio|[Audio_Mixture](https://drive.google.com/file/d/1b_itzUbLRxcqxwV8d4f_ufotvCzHSjHm/view?usp=sharing)||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/2-mix-separation/sample-2/separated_2_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/1ID6xYdaHSQtLTJXGMjyBpxnoW5bWqf1m/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/1aqPHt8MeI3fv7niG3WF7cu2zrMKQ8p7Y/view?usp=sharing)|


#### → 3-mix samples

 - Sample 1：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_2.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/Object_detection_3.png" width="150" height="150">|
|---|:-:|:-:|:-:|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio2_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/audio3_groundtruth_spectrogram.png" width="200">|
|Ground Truth Audio|[Audio_GT_1](https://drive.google.com/file/d/1LYFSjyDkwwhhVwnTaQNxowRvdD11E3IF/view?usp=sharing)|[Audio_GT_2](https://drive.google.com/file/d/1B0zTVY7MeBtLRq6Uets6GE3O-M2iWg5N/view?usp=sharing)|[Audio_GT_3](https://drive.google.com/file/d/15St14F-84NrfhCpykQza5Q68UI9U4u1-/view?usp=sharing)|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/spectrogram_mixture.png" width="200">|||
|Mixture Audio|[Audio_Mixture](https://drive.google.com/file/d/137nhYJ_ZPP3k93qF3C0aCvL34Ju-Iuj8/view?usp=sharing)|||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_2_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-1/separated_3_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/1BUYLqxPxPFdtqR4nTyuW25bc0Y5hSuFK/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/11mmfcS9vqCUpoqGm6So3FV2Z_4qTL32d/view?usp=sharing)|[Audio_Separation_3](https://drive.google.com/file/d/1Ad_qvnKwaX1zOe9eATqy8bAV3IKszeod/view?usp=sharing)|

 - Sample 2：

|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_1.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_2.png" width="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/detection_3.png" width="150" height="150">|
|---|:-:|:-:|:-:|
|Ground Truth Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio1_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio2_groundtruth_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/audio3_groundtruth_spectrogram.png" width="200">|
|Ground Truth Audio|[Audio_GT_1](https://drive.google.com/file/d/1pwy_aW0yYbzy_QNTT2yL6DgpIuvdvCzn/view?usp=sharing)|[Audio_GT_2](https://drive.google.com/file/d/1ma_H7nq1pNhDJYNifbN34X6SVwcYDUwL/view?usp=sharing)|[Audio_GT_3](https://drive.google.com/file/d/119c6najFMPSvhaRojHQTRv8J3VIJ8EcQ/view?usp=sharing)|
|Mixture Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/spectrogram_mixture.png" width="200">|||
|Mixture Audio|[Audio_Mixture](https://drive.google.com/file/d/1tJfKIixsyvZFx27W9tENodprAYJyWs2Y/view?usp=sharing)|||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_2_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/3-mix-separation/sample-2/separated_3_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/1XexqJnTmRnGDQ7tzmP2Mp4lqIgPTt4mH/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/1KRLJlycYZvwrOx34wfV933nQ3aEkB2uR/view?usp=sharing)|[Audio_Separation_3](https://drive.google.com/file/d/1AmfnZFa4XYvP5EIP2sUvxctCwjO_75wh/view?usp=sharing)|

#### → Real-mix duet samples

 - Sample 1:
 
|Frame|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/frame.png" width="150">||
|---|:-:|:-:|
|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/Object_detection_2.png" width="150" height="150">|
|Duet Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/duet_spectrogram_groundtruth.png" width="200">||
|Duet Audio|[Audio_Duet](https://drive.google.com/file/d/1VC4gHCAKZoQPem1v-4RgoC_LJ3kmD6p2/view?usp=sharing)||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-1/separated_2_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/1Lc6MvTNnjE_FlUyhIpNCIVmvL1WGccwu/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/1jZ5CtP_6zwjlQC9I4f1Qp5BfhYhkuK_a/view?usp=sharing)|

 - Sample 2:
 
|Frame|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/frame.png" width="150">||
|---|:-:|:-:|
|Detected Objects|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/Object_detection_1.png" width="150" height="150">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/Object_detection_2.png" width="150" height="150">|
|Duet Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/duet_spectrogram_groundtruth.png" width="200">||
|Duet Audio|[Audio_Duet](https://drive.google.com/file/d/1GBDdO0GUiDBPqvl0pNBsc8mmb8I70OX7/view?usp=sharing)||
|Separated Spectrogram|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/separated_1_spectrogram.png" width="200">|<img src="https://github.com/workspace-for-cross-modality/CRN-for-music-separation/blob/main/music_files/Real-mix duet-separation/sample-2/separated_2_spectrogram.png" width="200">|
|Separated Audio|[Audio_Separation_1](https://drive.google.com/file/d/12VgUmI4AoUT1Sw4LK0bTRgkBSVWTOa91/view?usp=sharing)|[Audio_Separation_2](https://drive.google.com/file/d/15ToiHPZXiS-jnBOEuECHCDmTnHyFRhTv/view?usp=sharing)|

## Code Part
The README of our code can be seen at /code/CRN/README.md.
