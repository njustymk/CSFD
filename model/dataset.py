import os
import pdb
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from torch.utils import data
import torchvision.transforms.functional as F

class irfaceDataset(data.Dataset):
    def __init__(self, root, datasplit, transform):
        super(irfaceDataset, self).__init__()

        cachefile = os.path.join(root, datasplit+'.pkl')
        if os.path.isfile(cachefile):
            with open(cachefile, "rb") as f:
                self.img_vises, \
                self.img_ir_poses, \
                self.img_ir_neg0s, \
                self.img_ir_neg1s, \
                self.img_ir_neg2s, \
                self.img_ir_neg3s, \
                self.img_ir_neg4s, \
                self.img_ir_neg5s, \
                self.img_ir_neg6s, \
                self.img_ir_neg7s, \
                self.img_ir_neg8s = pickle.load(f)
        else:
            self.img_vises, \
            self.img_ir_poses, \
            self.img_ir_neg0s, \
            self.img_ir_neg1s, \
            self.img_ir_neg2s, \
            self.img_ir_neg3s, \
            self.img_ir_neg4s, \
            self.img_ir_neg5s, \
            self.img_ir_neg6s, \
            self.img_ir_neg7s, \
            self.img_ir_neg8s = self.readdata(os.path.join(root, datasplit))
            with open(cachefile, "wb") as f:
                pickle.dump([
                    self.img_vises, \
                    self.img_ir_poses, \
                    self.img_ir_neg0s, \
                    self.img_ir_neg1s, \
                    self.img_ir_neg2s, \
                    self.img_ir_neg3s, \
                    self.img_ir_neg4s, \
                    self.img_ir_neg5s, \
                    self.img_ir_neg6s, \
                    self.img_ir_neg7s, \
                    self.img_ir_neg8s], f)
    
    def readdata(self, data_root):
        input_size = [32, 32]
        filenames = [x[:8] for x in os.listdir(data_root) if x[:1]=='0']
        filenames = list(set(filenames))
        img_vises = []
        img_ir_poses = []
        img_ir_neg0s = []
        img_ir_neg1s = []
        img_ir_neg2s = []
        img_ir_neg3s = []
        img_ir_neg4s = []
        img_ir_neg5s = []
        img_ir_neg6s = []
        img_ir_neg7s = []
        img_ir_neg8s = []
        for filename in tqdm(filenames):
            img_vis = Image.open(os.path.join(data_root, filename+'_vis.jpg')).convert('RGB')
            img_vis = img_vis.resize(input_size)
            img_vis = F.normalize(F.to_tensor(img_vis), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_pos = Image.open(os.path.join(data_root, filename+'_ir_pos.jpg')).convert('RGB')
            img_ir_pos = img_ir_pos.resize(input_size)
            img_ir_pos = F.normalize(F.to_tensor(img_ir_pos), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg0 = Image.open(os.path.join(data_root, filename+'_ir_neg0.jpg')).convert('RGB')
            img_ir_neg0 = img_ir_neg0.resize(input_size)
            img_ir_neg0 = F.normalize(F.to_tensor(img_ir_neg0), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg1 = Image.open(os.path.join(data_root, filename+'_ir_neg1.jpg')).convert('RGB')
            img_ir_neg1 = img_ir_neg1.resize(input_size)
            img_ir_neg1 = F.normalize(F.to_tensor(img_ir_neg1), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg2 = Image.open(os.path.join(data_root, filename+'_ir_neg2.jpg')).convert('RGB')
            img_ir_neg2 = img_ir_neg2.resize(input_size)
            img_ir_neg2 = F.normalize(F.to_tensor(img_ir_neg2), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg3 = Image.open(os.path.join(data_root, filename+'_ir_neg3.jpg')).convert('RGB')
            img_ir_neg3 = img_ir_neg3.resize(input_size)
            img_ir_neg3 = F.normalize(F.to_tensor(img_ir_neg3), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg4 = Image.open(os.path.join(data_root, filename+'_ir_neg4.jpg')).convert('RGB')
            img_ir_neg4 = img_ir_neg4.resize(input_size)
            img_ir_neg4 = F.normalize(F.to_tensor(img_ir_neg4), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            
            img_ir_neg5 = Image.open(os.path.join(data_root, filename+'_ir_neg5.jpg')).convert('RGB')
            img_ir_neg5 = img_ir_neg5.resize(input_size)
            img_ir_neg5 = F.normalize(F.to_tensor(img_ir_neg5), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg6 = Image.open(os.path.join(data_root, filename+'_ir_neg6.jpg')).convert('RGB')
            img_ir_neg6 = img_ir_neg6.resize(input_size)
            img_ir_neg6 = F.normalize(F.to_tensor(img_ir_neg6), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg7 = Image.open(os.path.join(data_root, filename+'_ir_neg7.jpg')).convert('RGB')
            img_ir_neg7 = img_ir_neg7.resize(input_size)
            img_ir_neg7 = F.normalize(F.to_tensor(img_ir_neg7), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_ir_neg8 = Image.open(os.path.join(data_root, filename+'_ir_neg8.jpg')).convert('RGB')
            img_ir_neg8 = img_ir_neg8.resize(input_size)
            img_ir_neg8 = F.normalize(F.to_tensor(img_ir_neg8), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_vises.append(img_vis)
            img_ir_poses.append(img_ir_pos)
            img_ir_neg0s.append(img_ir_neg0)
            img_ir_neg1s.append(img_ir_neg1)
            img_ir_neg2s.append(img_ir_neg2)
            img_ir_neg3s.append(img_ir_neg3)
            img_ir_neg4s.append(img_ir_neg4)
            img_ir_neg5s.append(img_ir_neg5)
            img_ir_neg6s.append(img_ir_neg6)
            img_ir_neg7s.append(img_ir_neg7)
            img_ir_neg8s.append(img_ir_neg8)

        return img_vises, img_ir_poses, img_ir_neg0s, img_ir_neg1s, img_ir_neg2s, img_ir_neg3s, \
                img_ir_neg4s, img_ir_neg5s, img_ir_neg6s, img_ir_neg7s, img_ir_neg8s


    def __getitem__(self, index):
        img_vis = self.img_vises[index]
        img_ir_pos = self.img_ir_poses[index]
        img_ir_neg0 = self.img_ir_neg0s[index]
        img_ir_neg1 = self.img_ir_neg1s[index]
        img_ir_neg2 = self.img_ir_neg2s[index]
        img_ir_neg3 = self.img_ir_neg3s[index]
        img_ir_neg4 = self.img_ir_neg4s[index]
        img_ir_neg5 = self.img_ir_neg5s[index]
        img_ir_neg6 = self.img_ir_neg6s[index]
        img_ir_neg7 = self.img_ir_neg7s[index]
        img_ir_neg8 = self.img_ir_neg8s[index]

        return img_vis, img_ir_pos, img_ir_neg0, img_ir_neg1, img_ir_neg2, img_ir_neg3, img_ir_neg4, \
                img_ir_neg5, img_ir_neg6, img_ir_neg7, img_ir_neg8
    
    def __len__(self):
        return len(self.img_vises)
