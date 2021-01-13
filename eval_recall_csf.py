import os
import cv2
import pdb
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import ops
import xml.etree.ElementTree as ET

def calculate_recall(bbox_gt_dir, result_dir, ovthresh):
    img_w = 640
    img_h = 480

    sample_ids = [x[:18] for x in os.listdir(bbox_gt_dir) if x[-3:]=='txt']

    tp = 0
    num_face = 0
    for index, sample_id in enumerate(sample_ids):

        bbox_gt_file = os.path.join(bbox_gt_dir, sample_id+'_gt.txt')
        f = open(bbox_gt_file, 'r')
        bbox_gts = []
        for parts in f.readlines():
            parts = parts.strip().split(' ')
            center_x = float(parts[1])*img_w
            center_y = float(parts[2])*img_h
            bbox_w   = float(parts[3])*img_w
            bbox_h   = float(parts[4])*img_h
            x_min = center_x-bbox_w/2
            y_min = center_y-bbox_h/2
            x_max = center_x+bbox_w/2
            y_max = center_y+bbox_h/2
            bbox_gts.append([x_min, y_min, x_max, y_max])
            num_face+=1
        bbox_gts = np.array(bbox_gts)

        bbox_res = []
        result_file = os.path.join(result_dir, sample_id+'_map_wd.txt')
        if os.path.isfile(result_file):
            f = open(result_file, 'r')
            _ = f.readline().strip()
            num_bbox = int(f.readline().strip())
            for i in range(num_bbox):
                parts = f.readline().strip().split(' ')
                x_min = float(parts[0])
                y_min = float(parts[1])
                x_max = float(parts[2])
                y_max = float(parts[3])
                score = float(parts[4])
                # if score<conf_th:
                #     continue
                bbox_w = x_max-x_min
                bbox_h = y_max-y_min
                bbox_res.append([x_min, y_min, x_max, y_max])

                steps = [bbox_w/8., bbox_w*2/8., bbox_w*3/8., bbox_w*4/8.]
                for step in steps:
                    x_min_face = max(x_min-step,0)
                    x_max_face = max(x_max-step,0)
                    bbox_res.append([x_min_face, y_min, x_max_face, y_max])

                    x_min_face = min(x_min+step,img_w)
                    x_max_face = min(x_max+step,img_w)
                    bbox_res.append([x_min_face, y_min, x_max_face, y_max])

                # bbox_res.append([x_min, y_min-1/8*bbox_h, x_max, y_max-1/8*bbox_h])
                # bbox_res.append([x_min, y_min+1/8*bbox_h, x_max, y_max+1/8*bbox_h])

                # bbox_res.append([x_min-1/8*bbox_w, y_min, x_max-1/8*bbox_w, y_max])
                # bbox_res.append([x_min+1/8*bbox_w, y_min, x_max+1/8*bbox_w, y_max])

                # bbox_res.append([x_min-2/8*bbox_w, y_min, x_max-2/8*bbox_w, y_max])
                # bbox_res.append([x_min+2/8*bbox_w, y_min, x_max+2/8*bbox_w, y_max])


        if bbox_res==[]:
            continue
        bbox_gts = torch.tensor(bbox_gts, dtype=torch.float64)
        bbox_res = torch.tensor(bbox_res, dtype=torch.float64)

        overlaps = ops.box_iou(bbox_gts, bbox_res)
        overlap_masks = overlaps >= ovthresh

        for overlap_mask in overlap_masks:
            if overlap_mask.any():
                tp+=1

        print(f'\r***Recall: {tp/num_face:.04f} index:{index} ***', end='')
        # pdb.set_trace()

if __name__ == '__main__':
    
    split = 'csf_all' #csf_all/csf_daylight/csf_night

    data_root = 'data/test/{}'.format(split)

    bbox_ir_gt_dir =  osp.join(data_root, 'bbox_ir_gt')
    print('bbox_ir_gt_dir', bbox_ir_gt_dir)

    result_dir = 'results/{}'.format(split)
    print('result_dir', result_dir)

    ovthresh = 0.5
    print('ovthresh', ovthresh)
    calculate_recall(bbox_ir_gt_dir, result_dir, ovthresh)

    ovthresh = 0.3
    print('ovthresh', ovthresh)
    calculate_recall(bbox_ir_gt_dir, result_dir, ovthresh)