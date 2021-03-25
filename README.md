# Monodepth2-TF2
It's a monodepth2 model implemented in TF2.x, original paper《Digging into Self-Supervised Monocular Depth Prediction》.

### Dependencies
```
tensorflow==2.3.1
(for gpu) cudatoolkit=10.1,  cudnn=7.6.5
```

### Note
It's currently just for personal use. So forgive me that I haven't used argument-parsing, you need to change some path settings when you run the demo.

However, no worries, it's just a simple code **merely for singlet depth estimation** (for now). That is, **no** PoseNet, training and evaluation yet. 
Anyways, Take one minute you will know what's going on in there.

`depth_esitmator_demo.py`: a short demo, where the model is encapsuled in Classes. Just change the vidoe & weights path, and of course put the weight (linke below) in the corresponding order..

`simple_run.py`: as the name suggests, it's a simple run, important functions are all there, no encapsulement.

The `depth_decoder_creater.py` and `encoder_creator.py` is used to 
- Useful part: build the Model in TF2.x the same way as the official *monodepth2* implemented in Pytroch.
- Neglectable part: weights loading. Weights were extracted from the official torch model in `numpy.ndarray` form, then directly loaded to the TF model layer-/parameter-wise. It's trivial. But I will upload the converted `SavedModel` directly, so no worries.

The [resnet18-encoder](https://drive.google.com/drive/folders/1yBIYsphJInPIjGtL3NjMzHhjVk6ExoRC?usp=sharing) and [depth-decoder](https://drive.google.com/drive/folders/19LdqNfcLJDneNu79TtUupDPael3vo0VM?usp=sharing) is trained on KITTI-Odometry dataset, size (640x192).

### TODO
- [X] implement the encoder-decoder model in TF2.x, just for singlet depth estimation. 
- [ ] multi-image-input version of the encoder
- [ ] PoseNet decoder
- [ ] training code
- [ ] evluation code

### Credits
- the Official repo: https://github.com/nianticlabs/monodepth2
- a TF1.x version implementation (complete with training and evaluation): https://github.com/FangGet/tf-monodepth2
