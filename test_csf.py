import os
import cv2
import pdb
import time
import pickle
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F

from detectors import MTCNN
from detectors import TinyFace
from detectors import S3FD
from detectors import DSFD
from detectors import FaceBoxes
from detectors import PyramidBox

import warnings
warnings.filterwarnings("ignore")

# class module2(nn.Module):
#     def __init__(self):
#         super(module2, self).__init__()
#         self.features = torchvision.models.vgg16(pretrained=True).features
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 8),
#         )
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x.view(x.size(0), -1))
#         return x

class module2(nn.Module):
    def __init__(self):
        super(module2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 8),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

def map_without_depth(bbox_vis):
    img_w = 640
    img_h = 480
    scale_x = 0.95
    scale_y = 0.79
    x_pan=-23
    y_pan=-5

    x_min = bbox_vis[0]
    y_min = bbox_vis[1]
    x_max = bbox_vis[2]
    y_max = bbox_vis[3]
    bbox_w = x_max-x_min
    bbox_h = y_max-y_min

    # ===============对bbox进行缩放=====================
    bbox_w = bbox_w*1.
    bbox_h = bbox_h*1.
    x_center = (x_min+x_max)/2.
    y_center = (y_min+y_max)/2.
    x_min = x_center-bbox_w/2
    y_min = y_center-bbox_h/2
    x_max = x_center+bbox_w/2
    y_max = y_center+bbox_h/2
    # ===============对bbox进行缩放=====================

    # =============坐标原点移动到图像中心================
    x_min = x_min-img_w/2
    x_max = x_max-img_w/2
    y_min = y_min-img_h/2
    y_max = y_max-img_h/2
    # =============坐标原点移动到图像中心================

    # =============缩放变换================
    x_min = x_min*scale_x
    x_max = x_max*scale_x
    y_min = y_min*scale_y
    y_max = y_max*scale_y
    # =============缩放变换================

    # =============平移变换================
    x_min = x_min+x_pan
    x_max = x_max+x_pan
    y_min = y_min+y_pan
    y_max = y_max+y_pan
    # =============平移变换================
    
    # =============坐标原点移动到图像左上角================
    x_min = x_min+img_w/2
    x_max = x_max+img_w/2
    y_min = y_min+img_h/2
    y_max = y_max+img_h/2
    # =============坐标原点移动到图像左上角================

    x_min = max(x_min,0)
    y_min = max(y_min,0)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)
    bbox_ir = [x_min, y_min, x_max, y_max]

    return bbox_ir

def map_with_depth(bbox_vis):
    img_w = 640
    img_h = 480
    scale_x = 0.95
    scale_y = 0.79
    x_pan=-23
    y_pan=-5

    x_min = bbox_vis[0]
    y_min = bbox_vis[1]
    x_max = bbox_vis[2]
    y_max = bbox_vis[3]
    bbox_w = x_max-x_min
    bbox_h = y_max-y_min

    # ===============对bbox进行缩放=====================
    bbox_w = bbox_w*1.
    bbox_h = bbox_h*1.
    x_center = (x_min+x_max)/2.
    y_center = (y_min+y_max)/2.
    x_min = x_center-bbox_w/2
    y_min = y_center-bbox_h/2
    x_max = x_center+bbox_w/2
    y_max = y_center+bbox_h/2
    # ===============对bbox进行缩放=====================

    # =============坐标原点移动到图像中心================
    x_min = x_min-img_w/2
    x_max = x_max-img_w/2
    y_min = y_min-img_h/2
    y_max = y_max-img_h/2
    # =============坐标原点移动到图像中心================

    # =============缩放变换================
    x_min = x_min*scale_x
    x_max = x_max*scale_x
    y_min = y_min*scale_y
    y_max = y_max*scale_y
    # =============缩放变换================

    # =============平移变换================
    x_min = x_min+x_pan
    x_max = x_max+x_pan
    y_min = y_min+y_pan
    y_max = y_max+y_pan
    # =============平移变换================

    # =============距离补偿================
    x_min = x_min-0.3*bbox_h+5
    x_max = x_max-0.3*bbox_h+5
    # y_min = y_min-0.1*bbox_h
    # y_max = y_max-0.2*bbox_h
    # =============距离补偿================
    
    # =============坐标原点移动到图像左上角================
    x_min = x_min+img_w/2
    x_max = x_max+img_w/2
    y_min = y_min+img_h/2
    y_max = y_max+img_h/2
    # =============坐标原点移动到图像左上角================

    x_min = max(x_min,0)
    y_min = max(y_min,0)
    x_max = min(x_max, img_w)
    y_max = min(y_max, img_h)
    bbox_ir = [x_min, y_min, x_max, y_max]

    return bbox_ir

def infer(face_vis, img_ir, input_size, model, x_min, y_min, x_max, y_max, device):
    if (x_max-x_min>1) and (y_max-y_min>1):
        face_ir = img_ir[int(y_min):int(y_max), int(x_min):int(x_max)]
        face_ir = Image.fromarray(cv2.cvtColor(face_ir,cv2.COLOR_BGR2RGB))
        face_ir =  F.normalize(F.to_tensor(face_ir.resize(input_size)), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).to(device)
        output1 = model(torch.stack([face_vis]))
        output2 = model(torch.stack([face_ir]))
        ap_dist = torch.norm(output1-output2, 2, dim=1).view(-1)
        return ap_dist.item()
    else:
        return 100

def correct(bbox_vis, bbox_map, model, img_ir, img_vis, device):
    img_w = 640
    img_h = 480
    input_size = [32, 32]

    x_min = max(bbox_vis[0], 0)
    y_min = max(bbox_vis[1], 0)
    x_max = min(bbox_vis[2], img_w)
    y_max = min(bbox_vis[3], img_h)
    face_vis = img_vis[int(y_min):int(y_max), int(x_min):int(x_max)]
    face_vis = Image.fromarray(cv2.cvtColor(face_vis,cv2.COLOR_BGR2RGB))
    face_vis =  F.normalize(F.to_tensor(face_vis.resize(input_size)), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).to(device)

    [x_min, y_min, x_max, y_max] = bbox_map
    bbox_w = x_max-x_min
    bbox_h = y_max-y_min

    outputs = []
    proposals = []

    outputs.append(infer(face_vis, img_ir, input_size, model, x_min, y_min, x_max, y_max, device))
    proposals.append([x_min, y_min, x_max, y_max])

    # steps = [bbox_w/8., bbox_w*2/8., bbox_w*3/8., bbox_w*4/8.]
    # steps = [bbox_w/8., bbox_w*2/8., bbox_w*3/8., bbox_w*4/8., bbox_w*5/8., bbox_w*6/8.]
    steps = [bbox_w/16., bbox_w*2/16., bbox_w*3/16., bbox_w*4/16., bbox_w*5/16., bbox_w*6/16.]
    # steps = [bbox_w/16., bbox_w*2/16., bbox_w*3/16., bbox_w*4/16., bbox_w*5/16., bbox_w*6/16., bbox_w*7/16., bbox_w*8/16.]
    
    for step in steps:
        x_min_face = max(x_min-step,0)
        x_max_face = max(x_max-step,0)
        outputs.append(infer(face_vis, img_ir, input_size, model, x_min_face, y_min, x_max_face, y_max, device))
        proposals.append([x_min_face, y_min, x_max_face, y_max])

        x_min_face = min(x_min+step,img_w)
        x_max_face = min(x_max+step,img_w)
        outputs.append(infer(face_vis, img_ir, input_size, model, x_min_face, y_min, x_max_face, y_max, device))
        proposals.append([x_min_face, y_min, x_max_face, y_max])

    proposals = np.array(proposals)
    outputs = np.array(outputs)

    bbox_correct = proposals[outputs==outputs.min()]

    return bbox_correct[0]

def read_bbox(bbox_file):
    bboxes = []

    f = open(bbox_file, 'r')
    _ = f.readline().strip()
    n = int(f.readline().strip())
    for i in range(n):
        parts = f.readline().strip().split(' ')
        x_min = float(parts[0])
        y_min = float(parts[1])
        x_max = float(parts[2])
        y_max = float(parts[3])
        score = float(parts[4])
        bboxes.append([x_min, y_min, x_max, y_max, score])
    f.close()
    return bboxes

def write_to_txt(bbox_file, det, sample_id):
    det = np.array(det)
    f = open(bbox_file, 'w')
    f.write('{:s}\n'.format(sample_id))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        # score = det[i][4] 
        score = 1.0
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.2f}\n'.format(xmin, ymin, xmax, ymax, score))
    f.close()

def test(data_root, sample_ids, img_irs, img_vises, branch2_weight_file, results_dir):
    img_ir_dir = os.path.join(data_root, 'ir')
    img_vis_dir = os.path.join(data_root, 'vis')
    bbox_vis_dir = os.path.join(data_root, 'bbox_vis_dsfd')

    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print('device:', device)

    model_dsfd = DSFD(device=device)
    model_correct = torch.load(branch2_weight_file).to(device)

    # sample_ids = [x[:18] for x in os.listdir(img_vis_dir) if x[-7:-4]=='vis']
    # sample_ids = np.sort(np.array(sample_ids))

    with torch.no_grad():
        # for sample_id in tqdm(sample_ids):
        #     img_ir = cv2.imread(os.path.join(img_ir_dir, sample_id+'_ir.jpg'))
        #     img_vis = cv2.imread(os.path.join(img_vis_dir, sample_id+'_vis.jpg'))
        for sample_id, img_ir, img_vis in tqdm(zip(sample_ids, img_irs, img_vises)):

            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            bboxes_vis = model_dsfd.detect_faces(img_vis_rgb, conf_th=0.5, scales=[1])

            # bbox_vis_file = os.path.join(bbox_vis_dir, sample_id+'.txt')
            # if not os.path.isfile(bbox_vis_file): continue
            # bboxes_vis = read_bbox(bbox_vis_file)

            bbox_map_out_wod = []
            bbox_map_out_wd = []
            bbox_res_out = []
            for bbox_vis in bboxes_vis:
                bbox_map_wod = map_without_depth(bbox_vis)
                bbox_map_out_wod.append(bbox_map_wod)

                bbox_map_wd = map_with_depth(bbox_vis)
                bbox_map_out_wd.append(bbox_map_wd)

                bbox_correct = correct(bbox_vis, bbox_map_wd, model_correct, img_ir, img_vis, device)
                bbox_res_out.append(bbox_correct)

            bbox_vis_file = os.path.join(results_dir, sample_id+'_vis.txt')
            write_to_txt(bbox_vis_file, bboxes_vis, sample_id)

            bbox_map_file = os.path.join(results_dir, sample_id+'_map_wod.txt')
            write_to_txt(bbox_map_file, bbox_map_out_wod, sample_id)

            bbox_map_file = os.path.join(results_dir, sample_id+'_map_wd.txt')
            write_to_txt(bbox_map_file, bbox_map_out_wd, sample_id)

            bbox_res_file = os.path.join(results_dir, sample_id+'_res.txt')
            write_to_txt(bbox_res_file, bbox_res_out, sample_id)
            # pdb.set_trace()

def prepare_test_data(data_root):
    cachefile = os.path.join(data_root, 'cache.pkl')
    if os.path.isfile(cachefile):
        print('loading cache file......')
        with open(cachefile, "rb") as f:
            sample_ids, img_irs, img_vises = pickle.load(f)
    else:
        img_ir_dir = os.path.join(data_root, 'ir')
        img_vis_dir = os.path.join(data_root, 'vis')
        sample_ids = [x[:18] for x in os.listdir(img_vis_dir) if x[-7:-4]=='vis']
        sample_ids = np.sort(np.array(sample_ids))
        img_irs = []
        img_vises = []

        for sample_id in tqdm(sample_ids):
            img_ir = cv2.imread(os.path.join(img_ir_dir, sample_id+'_ir.jpg'))
            img_vis = cv2.imread(os.path.join(img_vis_dir, sample_id+'_vis.jpg'))
            img_irs.append(img_ir)
            img_vises.append(img_vis)

        with open(cachefile, "wb") as f:
            pickle.dump([sample_ids, img_irs, img_vises], f)

    return sample_ids, img_irs, img_vises

if __name__ == '__main__':

    split = 'csf_all' #csf_all/csf_daylight/csf_night
    
    data_root = 'data/test/{}'.format(split)
    print('data_root', data_root)

    branch2_weight_file = 'model/weights_1neg/vgg11_data_epoch50.pt'
    print('branch2_weight_file', branch2_weight_file)

    results_dir = 'results/{}'.format(split)
    print('results_dir', results_dir)
    if not os.path.exists(results_dir): os.mkdir(results_dir)

    sample_ids, img_irs, img_vises = prepare_test_data(data_root)
    test(data_root, sample_ids, img_irs, img_vises, branch2_weight_file, results_dir)