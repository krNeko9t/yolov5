import argparse
from enum import Flag
import json
import yaml
from msilib.schema import Class
import os
import platform
import sys
from pathlib import Path
import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def getColor(i):
    cs = [
    (166,206,227),
    (31,120,180),
    (178,223,138),
    (255,255,255),
    (51,160,44),
    (251,154,153),
    (10,10,10),
    (227,26,28),
    (253,191,111),
    (255,127,0),
    (202,178,214),
    (106,61,154),
    (255,255,153),
    (177,89,40)]
    return cs[int(i)%len(cs)]

class Detect:
    # RELATE/ABSOLUTE
    # sourcePath = Path('E:/datasets/test/google/1') # Pic to detect
    # sourcePath = ROOT / 'data' / 'facade' / '8301703' # Pic to detect
    sourcePath = 'E:/yolov5/data/facade/mesh_soup' # Pic to detect
    # sourcePath = 'E:/datasets/test/facade' # Pic to detect
    # source = 'e:/datasets/test/ppt-1'

    weightsPath= ROOT / 'building_det.pt' # trained model name

    # 'DetectionResults' / ['Window','Building']
    targetPath = ROOT / 'DetectionResults' / 'Window' # Save Main Dir

    # USE_TIME
    subTarget = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))  # Save Sub Dir
    # subName = 'PPT'  # Save Sub Dir

    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False
    max_det=1000  # max object

    save_crop = True
    save_conf = True
    hide_labels = False
    hide_conf = False
    imgsz=(640, 640)
    line_width = 3
    device = ''

    def parseConfig(self):
        config = yaml.load(open("facade.yml"),yaml.FullLoader)
        for k in config["parameters"]:
            if getattr(self,k,None):
                setattr(self,k,config["parameters"][k])

        for k in config["path-config"]:
            if getattr(self,k,None):
                if config["path-config"][k] == "USE_TIME":
                    FullPath = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                else:
                    parts = str.split(config["path-config"][k],'/')
                    FullPath = Path()
                    for part in parts:
                        if part.strip() == "ROOT":
                            FullPath = ROOT
                        else:
                            FullPath = FullPath / part.strip()   
                    a = FullPath.resolve()                 
                setattr(self,k,FullPath)
        return 

    def __init__(self) -> None:
        self.parseConfig()
        # pass

    @torch.no_grad()
    def detectWindow(self):
        t0 = time_sync()

        # make dirs
        save_dir = increment_path(Path(self.targetPath) / self.subTarget, exist_ok=False)
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
        (save_dir / 'origin').mkdir(parents=True, exist_ok=True)
        (save_dir / 'mask').mkdir(parents=True, exist_ok=True)
        (save_dir / 'annotate').mkdir(parents=True, exist_ok=True)
        (save_dir / 'crops').mkdir(parents=True, exist_ok=True)

        # custom parameter
            # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weightsPath, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # prepare data
        dataset = LoadImages(self.sourcePath, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

        # run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        for path, im, im0s, vid_cap, s in dataset:
            # prepare image
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            # get prediction
            pred = model(im, augment=False, visualize=False) #?
            
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                # im0 原图
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_width, example=str(names))

                imc = im0.copy() if self.save_crop else im0  # for save_crop
                imOrigin = im0.copy()
                imAllMask = im0.copy()
                imMaskImgs = {}
                for c in names:
                    imMaskImgs[c] = im0.copy()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # process results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format

                        # Write label to .txt
                        with open(str(save_dir/'labels'/f'{p.stem}.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        c = int(cls)  # integer class

                        # label object with box
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True)) 

                        # save cropped object
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # mask image
                        cv2.rectangle(imMaskImgs[names[c]],(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),getColor(t0),-1)
                        cv2.rectangle(imAllMask,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),getColor(t0),-1)

                # save result
                imDetected = annotator.result()
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir/'origin'/f'{p.stem}.jpg'), imOrigin)
                    cv2.imwrite(str(save_dir/'mask'/f'{p.stem}.jpg'), imAllMask)
                    cv2.imwrite(str(save_dir/'annotate'/f'{p.stem}.jpg'), imDetected)

if __name__== "__main__":
    a = Detect()
    a.detectWindow()