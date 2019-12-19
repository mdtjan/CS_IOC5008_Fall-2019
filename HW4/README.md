### HW4
Instance segmentation based on Yolact (You Only Look At CoefficienTs).

Please refer to [original repo][web1] for more information. All credits goes to Bolya et al.
The paper is available at [arXiv][web2].

#### Requirements
Python 3, Torch (>=1.0.1), Cython, OpenCV, Pillow, pycocotools, matplotlib, etc

#### Training
Download the pre-train Resnet-101 on ImageNet dataset at [resnet101_reducedfc.pt][web3]
```python
python train.py --config=yolact_base_config --resume=weights/resnet101_reduced.pt --batch_size=16 --lr 0.005 --save_interval 900 --validation_epoch 10
```
The code above will save the weight for every 900 iterations and validate the result (using validation dataset) for every 10 epoch.

#### Evaluation
# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder

python eval.py --trained_model=weights/yolact_base_1735_145800.pth  --image=data/dataset/test_images/2007_000629.jpg  --output_coco_json

python eval.py --trained_model=weights/yolact_base_1735_145800.pth --output_coco_json --dataset=tiny_voc_dataset
annotations can't be simply empty array

#### Multi-GPU Support
Naturally PyTorch would recognize how many visible CUDA devices (the GPU) in the machine, but it is always a good practice, before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`. We can set `[gpus]` into `[0,1,2,3]` if we have 4 GPUs. It is very recommended to do this although we only have 1 GPU. Check the indices of the GPus with nvidia-smi. Then, simply set the batch size to 8*num_gpus with the training commands above. The training script will automatically scale the hyperparameters to the right values. If we have memory to spare we can increase the batch size further, but keep it a multiple of the number of GPUs were're using.

#### Logging
Please refer to the original repo above, but naturally it saves the log to `logs/yolact_base.log` file 

#### Custom Datasets
Please refer to the original repo. Note that for Tiny VOC Dataset, we use set `class_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]` (21 class, including 0/background).
