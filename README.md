# Recycle Trash Detection
**[AI Grand Challenge](http://www.ai-challenge.kr) (Nov 16, 2020  ~ Nov 20,2020)**  
**🥈 2nd Place Winner** of Object Classification Track  
<img src =https://user-images.githubusercontent.com/52495256/102849932-8e987180-445b-11eb-8728-96b52d696c5f.png width="700" height="300" />  

---
## Environments
```
Ubuntu 18.04.5 LTS   
Python 3.7  
CUDA 10.2  
```
`mmdet` is forked from [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection). Difference in `mmdet/datasets/coco.py`, variable `CLASSES`

---
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
### Preparing dataset  
- You can prepare datasets in [here](dataset/README.md)  

### Change json format to coco style
- You can refer to the sample images and jsons' format in the dataset folder.  
```bash
$ python3 change_jsonstyle/trash2coco.py
```  
---
## Training    
**\*Important\***: According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size  
if you use different GPUs or images per GPU,  
e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.  

```bash
# single-gpu training
$ python3 tools/train.py ${MODEL_FILE}   

# multi-gpu training  
$ ./tools/dist_train.sh ${MODEL_FILE}  ${GPU_NUM}  
```    
e.g. `python3 tools/train.py model/cascade.py`

## Inference  
- You will get json files `result.json` to be submit to the official evaluation server.  
- You can use the following commands to get result images.  
- Detection results will be plotted on the images and saved to the specified directory.  
```bash
# single-gpu testing
$ python3 predict.py ${MODEL_FILE} ${WEIGHT_FILE} ${RESULT_FOLDER}
# multi-gpu testing
$ ./tools/dist_test.sh ${MODEL_FILE} ${WEIGHT_FILE} ${RESULT_FOLDER} ${GPU_NUM}

```
e.g. `python3 tools/test.py model/cascade.py work_dirs/cascade/latest.pth ./result`

---  
## Another option : Docker  
- We provide a Dockerfile to build an image.  
```bash
now updating...
```  
- Run it with  
```
$ docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/Recycle-Trash-Detection/dataset RTD
```

