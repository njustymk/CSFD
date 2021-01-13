import os
import cv2
import pdb
import numpy as np
import os.path as osp
from tqdm import tqdm

def show_res(data_root, res_dir, res_dir_show):
    img_w = 640
    img_h = 480
    bbox_ir_gt_dir =  osp.join(data_root, 'bbox_ir_gt')
    img_vis_dir = osp.join(data_root, 'vis')
    img_ir_dir = osp.join(data_root, 'ir')

    sample_ids = [x[:18] for x in os.listdir(img_ir_dir) if x[-3:]=='jpg']
    sample_ids = np.array(sample_ids)
    sample_ids = np.sort(sample_ids)
    for index, sample_id in enumerate(sample_ids):
        print(sample_id)
        # if index<300:
        #     continue
        img_vis_file = os.path.join(img_vis_dir, sample_id+'_vis.jpg')
        img_vis = cv2.imread(img_vis_file)

        img_ir_file = os.path.join(img_ir_dir, sample_id+'_ir.jpg')
        img_ir = cv2.imread(img_ir_file)

        bbox_vis_file = os.path.join(res_dir, sample_id+'_vis.txt')
        if os.path.isfile(bbox_vis_file):
            f = open(bbox_vis_file, 'r')
            _ = f.readline().strip()
            num_bbox = int(f.readline().strip())
            for i in range(num_bbox):
                parts = f.readline().strip().split(' ')
                x_min = float(parts[0])
                y_min = float(parts[1])
                x_max = float(parts[2])
                y_max = float(parts[3])
                img_vis = cv2.rectangle(img_vis, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        img_ir1 = img_ir.copy()
        img_ir2 = img_ir.copy()
        img_ir3 = img_ir.copy()
        
        # bbox_gt_file = os.path.join(bbox_ir_gt_dir, sample_id+'_gt.txt')
        # f = open(bbox_gt_file, 'r')
        # for parts in f.readlines():
        #     parts = parts.strip().split(' ')
        #     center_x = float(parts[1])*img_w
        #     center_y = float(parts[2])*img_h
        #     bbox_w   = float(parts[3])*img_w
        #     bbox_h   = float(parts[4])*img_h
        #     x_min = center_x-bbox_w/2
        #     y_min = center_y-bbox_h/2
        #     x_max = center_x+bbox_w/2
        #     y_max = center_y+bbox_h/2
        #     img_ir1 = cv2.rectangle(img_ir1, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        #     img_ir2 = cv2.rectangle(img_ir2, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        #     img_ir3 = cv2.rectangle(img_ir3, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        bbox_map_file = os.path.join(res_dir, sample_id+'_map_wod.txt')
        if os.path.isfile(bbox_map_file):
            f = open(bbox_map_file, 'r')
            _ = f.readline().strip()
            num_bbox = int(f.readline().strip())
            for i in range(num_bbox):
                parts = f.readline().strip().split(' ')
                x_min = float(parts[0])
                y_min = float(parts[1])
                x_max = float(parts[2])
                y_max = float(parts[3])
                img_ir1 = cv2.rectangle(img_ir1, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        bbox_map_file = os.path.join(res_dir, sample_id+'_map_wd.txt')
        if os.path.isfile(bbox_map_file):
            f = open(bbox_map_file, 'r')
            _ = f.readline().strip()
            num_bbox = int(f.readline().strip())
            for i in range(num_bbox):
                parts = f.readline().strip().split(' ')
                x_min = float(parts[0])
                y_min = float(parts[1])
                x_max = float(parts[2])
                y_max = float(parts[3])
                img_ir2 = cv2.rectangle(img_ir2, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        
        bbox_res_file = os.path.join(res_dir, sample_id+'_res.txt')
        if os.path.isfile(bbox_res_file):
            f = open(bbox_res_file, 'r')
            _ = f.readline().strip()
            num_bbox = int(f.readline().strip())
            for i in range(num_bbox):
                parts = f.readline().strip().split(' ')
                x_min = float(parts[0])
                y_min = float(parts[1])
                x_max = float(parts[2])
                y_max = float(parts[3])
                img_ir3 = cv2.rectangle(img_ir3, (int(x_min),int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        # cv2.imwrite(os.path.join(res_dir_show, sample_id+'_ir1.jpg'), img_ir1)
        # cv2.imwrite(os.path.join(res_dir_show, sample_id+'_ir2.jpg'), img_ir2)
        # cv2.imwrite(os.path.join(res_dir_show, sample_id+'_ir3.jpg'), img_ir3)

        cv2.imshow('img_ir1', img_ir1)
        cv2.imshow('img_ir2', img_ir2)
        cv2.imshow('img_ir3', img_ir3)
        cv2.waitKey()

def make_video(res_dir_show, video_file):
    img_w = 640
    img_h = 480
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 7
    size = (img_w*2, img_h)
    videoWriter = cv2.VideoWriter(video_file, fourcc , fps, size)

    sample_ids = [x[:18] for x in os.listdir(res_dir_show) if x[-6:-4]=='ir']
    sample_ids = np.sort(np.array(sample_ids))

    for sample_id in tqdm(sample_ids):  
        img_vis = cv2.imread(os.path.join(res_dir_show, sample_id+'_vis.jpg'))
        img_ir = cv2.imread(os.path.join(res_dir_show, sample_id+'_ir.jpg'))

        image = np.concatenate((img_vis, img_ir), axis=1) 
        # cv2.imshow('image', image)
        # cv2.waitKey()
        videoWriter.write(image)
    videoWriter.release()

if __name__ == '__main__':
    split = 'csf_daylight' #csf_all/csf_daylight/csf_night
    
    data_root = 'data/test/{}'.format(split)
    res_dir =  'results/{}'.format(split)

    res_dir_show = 'results/{}'.format(split+'_show')
    if not os.path.exists(res_dir_show): os.mkdir(res_dir_show)

    show_res(data_root, res_dir, res_dir_show)

    # video_file = ''results'/{}'.format(split+'.mp4')
    # print('video_file', video_file)
    # make_video(res_dir_show, video_file)

