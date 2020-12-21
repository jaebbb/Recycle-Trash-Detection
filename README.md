# Recycle Trash Detection
**AI Grand Challenge (Nov 16, 2020  ~ Nov 20,2020)**  

## Environments
```
Ubuntu 18.04.5 LTS   
Python 3.7  
CUDA 10.2  
```
`mmdet` is forked from [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection). Difference in `mmdet/datasets/coco.py`, variable `CLASSES`


## Install  
### Requirements  
```bash
$ git clone https://github.com/jaebbb/recycle-trash-detection.git
$ cd recycle-trash-detection
$ pip install -r requirements.txt
$ pip install mmpycocotools
$ pip install mmcv-full==1.1.6+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html --use-deprecated=legacy-resolver
$ pip install -v -e .
```
### Change json format to coco style
You can refer to the sample images and jsons' format in the dataset folder.  
```bash
$ python3 change_jsonstyle/trash2coco.py
```  

## Training    
```bash
# single-gpu training
$ python3 tools/train.py model/cascade.py   

# multi-gpu testing  
$ ./tools/dist_train.sh model/cascade.py  ${GPU_NUM}  
```    

## Inference  
You will get json files `xxxx.json` to be submit to the official evaluation server.
```bash
$ python3 predict.py {CHECKPOINT} {IMAGES_PATH}
```  
You can use the following commands to get result images.  
Detection results will be plotted on the images and saved to the specified directory.  
```bash
$ python3 ....
```

## Another option : Docker  
We provide a Dockerfile to build an image.  
```bash
now updating...
```  
Run it with  
```
$ docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/Recycle-Trash-Detection/dataset RTD
```

