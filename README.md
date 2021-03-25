# Monodepth2-TF2
It's a monodepth2 model implemented in TF2.x, original paper《Digging into Self-Supervised Monocular Depth Prediction》.

### Dependencies
```
tensorflow==2.3.1
(for gpu) cudatoolkit=10.1,  cudnn=7.6.5
```

#### Note
Forgive me that I haven't used argument-parsing, so you need to take one minute to change some path. No worries, it's just a simple code merely for singlet depth estimation (for now). That is, **no** PoseNet, training and evaluation yet. Take one minute you will know what's going on in there.

`simple_run`: as the name suggests, it's a simple run. Just change the vidoe path.

`depth_esitmator_demo`: complete demo, where the model is encapsuled in Classes.

The `depth_decoder_creater.py` and `encoder_creator.py` is used to 
- Useful part: build the Model in TF2.x the same way as the official *monodepth2* implemented in Pytroch.
- Neglectable part: weights loading. Weights were extracted from the official torch model in `numpy.ndarray` form, then directly loaded to the TF model layer-/parameter-wise. It's trivial. But I will upload the converted `SavedModel` directly, so no worries.


### TODO
- [X] implement the encoder-decoder model in TF2.x, just for singlet depth estimation. 
- [ ] multi-image-input version of the encoder
- [ ] PoseNet decoder
- [ ] training code
- [ ] evluation code
