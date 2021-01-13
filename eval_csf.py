import os
import cv2
import pdb
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import ops
import xml.etree.ElementTree as ET

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_ap_map_without_depth(bbox_gt_dir, result_dir, ovthresh):
    img_w = 640
    img_h = 480
    sample_ids = [x[:18] for x in os.listdir(bbox_gt_dir) if x[-3:]=='txt']
    tp = []
    fp = []
    npos = 0
    sample_ids = np.array(sample_ids)
    sample_ids = np.sort(sample_ids)
    for index, sample_id in tqdm(enumerate(sample_ids)):

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
            bbox_gts.append([x_min, y_min, x_max, y_max, 1])
            npos = npos+1
        bbox_gts = np.array(bbox_gts)

        result_file = os.path.join(result_dir, sample_id+'_map_wod.txt')
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

                ixmin = np.maximum(bbox_gts[:, 0], x_min)
                iymin = np.maximum(bbox_gts[:, 1], y_min)
                ixmax = np.minimum(bbox_gts[:, 2], x_max)
                iymax = np.minimum(bbox_gts[:, 3], y_max)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((x_max - x_min + 1.) * (y_max - y_min + 1.) +
                       (bbox_gts[:, 2] - bbox_gts[:, 0] + 1.) *
                       (bbox_gts[:, 3] - bbox_gts[:, 1] + 1.) - inters)

                overlaps = inters / uni  #IOU
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if ovmax > ovthresh:
                    if bbox_gts[jmax][4]==1:
                        bbox_gts[jmax][4]=0
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) #召回率
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # 精准率,查准率
    ap = voc_ap(rec, prec, use_07_metric=True)
    print('ap_map_without_depth', ap)

def calculate_ap_map(bbox_gt_dir, result_dir, ovthresh):
    img_w = 640
    img_h = 480
    sample_ids = [x[:18] for x in os.listdir(bbox_gt_dir) if x[-3:]=='txt']
    tp = []
    fp = []
    npos = 0
    sample_ids = np.array(sample_ids)
    sample_ids = np.sort(sample_ids)
    for index, sample_id in tqdm(enumerate(sample_ids)):

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
            bbox_gts.append([x_min, y_min, x_max, y_max, 1])
            npos = npos+1
        bbox_gts = np.array(bbox_gts)

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

                ixmin = np.maximum(bbox_gts[:, 0], x_min)
                iymin = np.maximum(bbox_gts[:, 1], y_min)
                ixmax = np.minimum(bbox_gts[:, 2], x_max)
                iymax = np.minimum(bbox_gts[:, 3], y_max)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((x_max - x_min + 1.) * (y_max - y_min + 1.) +
                       (bbox_gts[:, 2] - bbox_gts[:, 0] + 1.) *
                       (bbox_gts[:, 3] - bbox_gts[:, 1] + 1.) - inters)

                overlaps = inters / uni  #IOU
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if ovmax > ovthresh:
                    if bbox_gts[jmax][4]==1:
                        bbox_gts[jmax][4]=0
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) #召回率
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # 精准率,查准率
    ap = voc_ap(rec, prec, use_07_metric=True)
    print('ap_map', ap)


def calculate_ap_res(bbox_gt_dir, result_dir, ovthresh):
    img_w = 640
    img_h = 480
    sample_ids = [x[:18] for x in os.listdir(bbox_gt_dir) if x[-3:]=='txt']
    tp = []
    fp = []
    npos = 0
    sample_ids = np.array(sample_ids)
    sample_ids = np.sort(sample_ids)
    for index, sample_id in tqdm(enumerate(sample_ids)):

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
            bbox_gts.append([x_min, y_min, x_max, y_max, 1])
            npos = npos+1
        bbox_gts = np.array(bbox_gts)

        result_file = os.path.join(result_dir, sample_id+'_res.txt')
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

                ixmin = np.maximum(bbox_gts[:, 0], x_min)
                iymin = np.maximum(bbox_gts[:, 1], y_min)
                ixmax = np.minimum(bbox_gts[:, 2], x_max)
                iymax = np.minimum(bbox_gts[:, 3], y_max)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((x_max - x_min + 1.) * (y_max - y_min + 1.) +
                       (bbox_gts[:, 2] - bbox_gts[:, 0] + 1.) *
                       (bbox_gts[:, 3] - bbox_gts[:, 1] + 1.) - inters)

                overlaps = inters / uni  #IOU
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if ovmax > ovthresh:
                    if bbox_gts[jmax][4]==1:
                        bbox_gts[jmax][4]=0
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) #召回率
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # 精准率,查准率
    ap = voc_ap(rec, prec, use_07_metric=True)
    print('ap_res', ap)

if __name__ == '__main__':
    split = 'csf_all' #csf_all/csf_daylight/csf_night

    data_root = 'data/test/{}'.format(split)
    print('data_root', data_root)
    bbox_ir_gt_dir =  osp.join(data_root, 'bbox_ir_gt')

    result_dir = 'results/{}'.format(split)
    print(result_dir)

    ovthresh = 0.7
    print('ovthresh', ovthresh)
    calculate_ap_map_without_depth(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_map(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_res(bbox_ir_gt_dir, result_dir, ovthresh)

    ovthresh = 0.5
    print('ovthresh', ovthresh)
    calculate_ap_map_without_depth(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_map(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_res(bbox_ir_gt_dir, result_dir, ovthresh)

    ovthresh = 0.3
    print('ovthresh', ovthresh)
    calculate_ap_map_without_depth(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_map(bbox_ir_gt_dir, result_dir, ovthresh)
    calculate_ap_res(bbox_ir_gt_dir, result_dir, ovthresh)
    