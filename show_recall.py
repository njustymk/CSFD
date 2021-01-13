import os
import cv2
import pdb
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import ops
import xml.etree.ElementTree as ET

def calculate_recall(bbox_gt_dir, result_dir, ovthresh, img_ir_dir, img_vis_dir):
    img_w = 640
    img_h = 480

    sample_ids = [x[:18] for x in os.listdir(bbox_gt_dir) if x[-3:]=='txt']
    sample_ids = np.sort(np.array(sample_ids))
    tp = 0
    num_face = 0
    for index, sample_id in enumerate(sample_ids):
        ir_file = os.path.join(img_ir_dir, sample_id+'_ir.jpg')
        img_ir = cv2.imread(ir_file)

        proposals = []
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

                bbox_w = x_max-x_min
                bbox_h = y_max-y_min
                bbox_map = [x_min, y_min, x_max, y_max]
                proposals.append([x_min, y_min, x_max, y_max])

                cv2.rectangle(img_ir, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

                steps = [bbox_w/8., bbox_w*2/8., bbox_w*3/8., bbox_w*4/8.]
                for step in steps:
                    x_min_face = max(x_min-step,0)
                    x_max_face = max(x_max-step,0)
                    proposals.append([x_min_face, y_min, x_max_face, y_max])

                    # img_ir_show = img_ir.copy()
                    # cv2.rectangle(img_ir_show, (int(x_min_face),int(y_min)), (int(x_max_face), int(y_max)), (0, 0, 255), 2)
                    # cv2.imshow('img_ir_show', img_ir_show)
                    # cv2.waitKey()

                    x_min_face = min(x_min+step,img_w)
                    x_max_face = min(x_max+step,img_w)
                    proposals.append([x_min_face, y_min, x_max_face, y_max])

                    # img_ir_show = img_ir.copy()
                    # cv2.rectangle(img_ir_show, (int(x_min_face),int(y_min)), (int(x_max_face), int(y_max)), (0, 0, 255), 2)
                    # cv2.imshow('img_ir_show', img_ir_show)
                    # cv2.waitKey()


        

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
            bbox_gt = [x_min, y_min, x_max, y_max]
            # bbox_gts.append(bbox_gt)
            num_face=+1

            if len(proposals)==0:
                img_ir = cv2.rectangle(img_ir, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                cv2.imshow('img_ir_show', img_ir)
                cv2.waitKey()
            else:
                bbox_gt = torch.tensor([bbox_gt], dtype=torch.float64)
                proposals = torch.tensor(proposals, dtype=torch.float64)
                overlaps = ops.box_iou(bbox_gt, proposals)
                overlap_masks = overlaps >= ovthresh

                if overlap_masks[0].any():
                    print('recall')
                else:
                    img_ir = cv2.rectangle(img_ir, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                    cv2.imshow('img_ir_show', img_ir)
                    cv2.waitKey()


        # bbox_gts = np.array(bbox_gts)

        # if bbox_res==[]:
        #     continue
        # bbox_gts = torch.tensor(bbox_gts, dtype=torch.float64)
        # bbox_res = torch.tensor(bbox_res, dtype=torch.float64)

        # overlaps = ops.box_iou(bbox_gts, bbox_res)
        # overlap_masks = overlaps >= ovthresh

        # for overlap_mask in overlap_masks:
        #     if overlap_mask.any():
        #         tp+=1
        #     else:
        #         cv2.imshow('img_ir', img_ir)
        #         cv2.waitKey()


        # print(f'\r***Recall: {tp/num_face:.04f} index:{index} ***', end='')
        # pdb.set_trace()

if __name__ == '__main__':
    
    split = 'daylight' #daylight/night/daylight_cold

    ovthresh = 0.5
    print('ovthresh', ovthresh)

    data_root = 'data/test/{}'.format(split)

    bbox_ir_gt_dir =  osp.join(data_root, 'bbox_ir_gt')
    print('bbox_ir_gt_dir', bbox_ir_gt_dir)

    result_dir = 'results/{}'.format(split)
    print('result_dir', result_dir)

    img_ir_dir = osp.join(data_root, 'ir')
    img_vis_dir = osp.join(data_root, 'vis')
    
    calculate_recall(bbox_ir_gt_dir, result_dir, ovthresh, img_ir_dir, img_vis_dir)