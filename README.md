# music_autotagging_ResCNN
a ResCNN model for music autotagging

### Introduction
This model used a simple ResCNN structure to implement music autotagging, which could be feed with two different inputs which consists of spectrogram input and waveform input.
- spectrogram as input

  reference paper: https://ieeexplore.ieee.org/abstract/document/7952585/
  
  related github projects : https://github.com/DENGQUANXIN/musicTagging-rcnn
  > you only need to use all files under the directory *"music_autotagging_ResCNN/spectrogram_model/"* to replace relevant files of above project, and run that project.

- waveform as input
  
  reference paper: https://ieeexplore.ieee.org/abstract/document/8681654
  
  related github projects: https://github.com/tae-jun/sampleaudio
  
  > you only need to use all files under the directory *"music_autotagging_ResCNN/waveform_model/"* to replace relevant files of above project, and run that project.

### model structure

- an example

![1](https://github.com/qmh1234567/music_autotagging_ResCNN/blob/master/imgs/%E5%9B%BE%E7%89%871.png)

- Resblock structure

![2](https://github.com/qmh1234567/music_autotagging_ResCNN/blob/master/imgs/%E5%9B%BE%E7%89%872.png)

- model structure
![3](https://github.com/qmh1234567/music_autotagging_ResCNN/blob/master/imgs/%E5%9B%BE%E7%89%873.png)

### Result

![4](https://github.com/qmh1234567/music_autotagging_ResCNN/blob/master/imgs/%E5%9B%BE%E7%89%874.png)
