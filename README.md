# Recycle Trash Detection
**AI Grand Challenge (Nov 16, 2020  ~ Nov 20,2020)**  

# Install  
### Requirements  
```bash
git clone https://github.com/jaebbb/recycle-trash-detection.git
cd recycle-trash-detection
pip install -r requirements.txt
pip install mmpycocotools
pip install mmcv-full==1.1.6+torch1.5.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html --use-deprecated=legacy-resolver
pip install -v -e .
```
### Change json format to coco style
You can refer to the sample images and jsons' format in the dataset folder.  
```bash
python3 change_jsonstyle/trash2coco.py
```  

# Run  
```bash
# single-gpu training
python3 tools/train.py model/cascade.py   

# multi-gpu testing  
./tools/dist_train.sh model/cascade.py  ${GPU_NUM}  
```  