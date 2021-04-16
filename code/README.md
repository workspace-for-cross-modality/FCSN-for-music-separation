# Vision-guided Music Source Separation via a Fine-grained Cycle-Separation Network (FCSN)
This text is used to explain the code involved in our work in this paper.

## Implementations
The implementation is based on Python 3.6 and Pytorch 1.1.
We recommend you use conda to install the dependencies.
All the requirements are listed bellow:
* numpy
* scipy
* opencv-python
* torchaudio
* matplotlib
* pyyaml
* pytorch
* torchvision
* h5py
* librosa
* mir_eval
* csv

## Training
You can train your own model by running the training file:
```
python separation_train.py
```

And the training parameters can be changed in options/train_options.py and base_options.py.

## Test
You can do three types of test options such as 2-mix, 3-mix, and duet, by running:
```
python test_2mix.py
python test_3mix.py
python test_duet.py 
```
respectively.

Also, the testing parameters can be changed in options/test_options.py and base_options.py.

The calculating process of the whole model can be seen in models/audioVisual_model.py, and the definition of each part of the model can be seen in models/networks.py.

In addition, we also provide a script to quickly detect objects in video frames, getDetectionResults.py.
