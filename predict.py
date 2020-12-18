
import argparse
import os

import json
# import natsort
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
from collections import OrderedDict

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('--filepath', default='./dataset/trash/test/imgs', help='test file path')
    parser.add_argument('--model', default='./work_dirs/cascade/cascade.py')
    parser.add_argument('--weight', default='work_dirs/cascade/epoch_4.pth')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
        
    return args


def geojson2coco(imageroot: str, geojsonpath: str, destfile, difficult='-1'):
    CLASS_NAMES_EN = ('background', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7')
    # set difficult to filter '2', '1', or do not filter, set '-1'
    if not geojsonpath:
        # images_list = natsort.natsorted(glob(imageroot+'/*.jpg'))        
        images_list = glob(imageroot+'/*.jpg')
        img_id_map = {images_list[i].split('/')[-1]:i+1 for i in range(len(images_list))}
        data_dict = {}
        data_dict['images']=[]
        data_dict['categories'] = []
        
        for idex, name in enumerate(CLASS_NAMES_EN[1:]):
            single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
            data_dict['categories'].append(single_cat)
            
        for imgfile in tqdm(img_id_map, desc='saving img info'):
            imagepath = os.path.join(imageroot, imgfile)
            img_id = img_id_map[imgfile]
            img = cv2.imread(imagepath)
            height, width, c = img.shape
            single_image = {}
            single_image['file_name'] = imgfile
            single_image['id'] = img_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

        with open(destfile, 'w') as f_out:
            json.dump(data_dict, f_out)
    
        return images_list




def main():
    args = parse_args()  
    # load model.py
    cfg = Config.fromfile(args.model)
    images_list = geojson2coco(imageroot= args.filepath,
                    geojsonpath = None,
                    destfile='./testcoco.json')
    # change the test filepath
    cfg.data_root = args.filepath
    cfg.data.test['img_prefix'] = args.filepath
    cfg.data.test['ann_file'] = './testcoco.json'
      
      
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False) 

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.weight , map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    show_score_thr = 0.5
    outputs = single_gpu_test(model, data_loader, False, None, show_score_thr)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {'jsonfile_prefix': os.path.dirname(os.path.realpath(__file__))+'/result'}
        dataset.format_results(outputs, **kwargs)

    # json file    
    # apply score_thr 
    with open(os.path.dirname(os.path.realpath(__file__))+'/result.bbox.json') as f:
        pred = json.load(f)

    while True:
        count = 0
        for data in pred:
            if data['score'] < show_score_thr:
                pred.remove(data)
                count +=1
        if count ==0:
            break

    with open(os.path.dirname(os.path.realpath(__file__))+'/result.bbox.json', 'w') as f:
        json.dump(pred,f,indent='\t')

    # convert to submission style
    with open(os.path.dirname(os.path.realpath(__file__))+'/result.bbox.json') as json_file:
        json_data = json.load(json_file)
    
    size = len(images_list)
    check = [False for i in range(size)]
    dic = OrderedDict({key:{'id': key+1, 'file_name': images_list[key].split('/')[-1], 'object':[{'box':[], 'label': ""}]} for key in range(size)})

    f = open('./t3_res_0030.json', 'w')
    for item in json_data:
        cur_id = item["image_id"] - 1

        if check[cur_id] == False: # nothing 
            x, y, w, h = [int(i) for i in item["bbox"]]
            dic[cur_id]['object'] = [{"box":[x, y, x+w, y+h], "label": "c" + str(item["category_id"])}]
            check[cur_id] = True
        else:
            x, y, w, h = [int(i) for i in item["bbox"]]
            dic[cur_id]["object"].append({"box":[x, y, x+w, y+h], "label": "c" + str(item["category_id"])})

    dic = list(dic.values())
    last = {"annotations":dic}
    last = str(last).replace("'", '"')

    f.write(last)
    f.close()
    
    os.remove(os.path.dirname(os.path.realpath(__file__))+'/result.bbox.json')
    os.remove(os.path.dirname(os.path.realpath(__file__))+'/testcoco.json')
    
    

if __name__ == '__main__':
    main()
