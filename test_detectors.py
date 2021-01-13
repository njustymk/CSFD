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
        score = det[i][4] 
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.2f}\n'.format(xmin, ymin, xmax, ymax, score))
    f.close()

def test(data_root, model, results_dir):
    img_ir_dir = os.path.join(data_root, 'ir')

    sample_ids = [x[:18] for x in os.listdir(img_ir_dir)]
    sample_ids = np.sort(np.array(sample_ids))

    with torch.no_grad():
        for sample_id in tqdm(sample_ids):
            img_ir = cv2.imread(os.path.join(img_ir_dir, sample_id+'_ir.jpg'))
            img_ir_rgb = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
            bboxes_ir = model.detect_faces(img_ir_rgb, conf_th=0.9, scales=[1])
            # print(bboxes_ir)
            bbox_res_file = os.path.join(results_dir, sample_id+'_ir.txt')
            write_to_txt(bbox_res_file, bboxes_ir, sample_id)
            # pdb.set_trace()

if __name__ == '__main__':

    split = 'csf_all' #csf_all/csf_daylight/csf_night
    detector = 'pyramidbox' #mtcnn/tinyface/s3fd/dsfd/faceboxes/pyramidbox

    data_root = 'data/test/{}'.format(split)
    print('data_root', data_root)

    results_dir = 'results_detectors/{}_{}'.format(split, detector)
    print('results_dir', results_dir)
    if not os.path.exists(results_dir): os.mkdir(results_dir)

    if torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print('device:', device)
    
    if detector=='mtcnn':
        model = MTCNN(device=device)
    elif detector=='tinyface':
        model = TinyFace(device=device)
    elif detector=='s3fd':
        model = S3FD(device=device)
    elif detector=='dsfd':
        model = DSFD(device=device)
    elif detector=='faceboxes':
        model = FaceBoxes(device=device)
    elif detector=='pyramidbox':
        model = PyramidBox(device=device)

    test(data_root, model, results_dir)