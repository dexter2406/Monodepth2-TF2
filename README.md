# Monodepth2-TF2
It's a monodepth2 model implemented in TF2.x, original paper《Digging into Self-Supervised Monocular Depth Prediction》.

### Dependencies
```
tensorflow==2.3.1
(for gpu) cudatoolkit=10.1,  cudnn=7.6.5
```

### Performance
I'm using a normal GTX1060-MaxQ GPU. The FPS for single-image depth estimation:
- using `tf.saved_model.load` (I think it's serving mode)
  - encoder: ~2ms (500 FPS)
  - decoder: ~2ms
  - overall: >200 FPS (but when I use with YOLOv4, it drops to ~120 FPS)
- using `tf.keras.models.load_model` with `model.predict()`:
  - overall : ~100 FPS (details forgot...)


### Note

It's currently just for personal use, but please feel free to contact me if you need anything. 

Forgive me that I haven't used argument-parsing, so you can't run with one command. You need to change some path settings when you run the demo. However, no worries, it's just a simple code **merely for singlet depth estimation** (for now). That is, **no** PoseNet, training and evaluation yet. 
Anyways, Take one minute you will know what's going on in there.

`simple_run.py`: as the name suggests, it's a simple run, important functions are all there, no encapsulement.

`depth_esitmator_demo.py`: a short demo, where the model is encapsuled in Classes. But I recommand to see `simple_run.py` first.

The `depth_decoder_creater.py` and `encoder_creator.py` is used to 
- Useful part: build the Model in TF2.x the same way as the official *monodepth2* implemented in Pytroch.
- Neglectable part: weights loading. Weights were extracted from the official torch model in `numpy.ndarray` form, then directly loaded to the TF model layer-/parameter-wise. It's trivial. But I will upload the converted `SavedModel` directly, so no worries.

The models (official weights, trained on KITTI-Odometry dataset, size (640x192).):
- [resnet18-encoder](https://drive.google.com/drive/folders/1yBIYsphJInPIjGtL3NjMzHhjVk6ExoRC?usp=sharing) 
- [depth-decoder_one_output](https://drive.google.com/drive/folders/19LdqNfcLJDneNu79TtUupDPael3vo0VM?usp=sharing) 
- [pose_encoder (same resnet18)](https://drive.google.com/drive/folders/1FW_Biq18WUNDV34sDZUt7Ztd5_1_U6Zo?usp=sharing)
- [pose_decoder](https://drive.google.com/drive/folders/1_H1HZNXFUAZgnBWNLcbeuHDM9eoNP-5k?usp=sharing)

### TODO
- [X] implement the encoder-decoder model in TF2.x, just for singlet depth estimation. 
- [X] Pose encoder: multi-image-input version of the encoder
- [X] Pose decoder
- [ ] training code
- [ ] evluation code

### Credits
- the Official repo: https://github.com/nianticlabs/monodepth2
- a TF1.x version implementation (complete with training and evaluation): https://github.com/FangGet/tf-monodepth2
