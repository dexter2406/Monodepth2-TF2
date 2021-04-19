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

### example usage:
Use `@tf.function` decorator for `Train.grad()` and `DataPprocessor.prepare_batch()` will allow much larger `batch_size`
```
# Start training (from scratch):
python train.py --from_scratch

# Continue training:
python train.py --weights_dir <weights_folder_path>

# To see intermediate result:
# remember to comment "tf.function" decorator for Train.grad() and DataPprocessor.prepare_batch()
python train.py --weights_dir <folder_path> --debug_mode True
```
Check `train.py` file for more details.


The models (transferred from official Pytorch model, trained on KITTI-Odometry dataset, size (640x192).):
- [weights_all_4_models](https://drive.google.com/drive/folders/1hPLVCowqvypekJy4UAB_HHAt1xtqR-H_?usp=sharing) 

Note：It's trained on *Odometry* split, so if applied on *Raw* data, the results won't be perfect.

### TODO
- [X] implement the encoder-decoder model in TF2.x, just for singlet depth estimation. 
- [X] Pose encoder: multi-image-input version of the encoder
- [X] Pose decoder
- [X] training code
- [ ] evluation code
- [ ] simple test code
- [ ] modify data loader to accept more dataset
- [ ] try new stuff in similar papers, e.g. *struct2dpeth*

### Note up-to-date
Now you can train your own model using the `train.py` and `new_trainer.py`. For now I just trained for 1 epoch, and the results (the reconstructed image and the disp image) seem to head to the correct way. Next step will be:
- completing the *evaluate* code.
- modify the code to take more dataset (for now only *KITTI_Raw* is supported), e.g. *KITTI_Odometry* and *Cityscapes*

---

### History note

#### March
Just for personal use, but please feel free to contact me if you need anything. 

I haven't used argument-parsing, so you can't run with one command. You need to change some path settings when you run the demo. However, no worries, it's just a simple code **merely for singlet depth estimation** (for now). The Pose networks and training pipeline is still in progress, needs double check. Though bug-free, I can't give any guarantees for total correctness.
Anyways, Take one minute you will know what's going on in there.

`simple_run.py`: as the name suggests, it's a simple run, important functions are all there, no encapsulement.

`depth_esitmator_demo.py`: a short demo, where the model is encapsuled in Classes. But I recommand to see `simple_run.py` first.

The `depth_decoder_creater.py` and `encoder_creator.py` is used to 
- Useful part: build the Model in TF2.x the same way as the official *monodepth2* implemented in Pytroch.
- Neglectable part: weights loading. Weights were extracted from the official torch model in `numpy.ndarray` form, then directly loaded to the TF model layer-/parameter-wise. It's trivial. But I will upload the converted `SavedModel` directly, so no worries.

### Credits
- the Official repo: https://github.com/nianticlabs/monodepth2
- a TF1.x version implementation (complete with training and evaluation): https://github.com/FangGet/tf-monodepth2
