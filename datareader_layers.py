import sys
sys.path.append('/home/zss/zsl/caffe-ssd/python')
import caffe

import numpy as np
from PIL import Image
import scipy.io as sco
import scipy.misc as smc
import random

class OriDataReaderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.train_data_dir = params['train_data_dir']
        self.mean_file = params['mean_file']
        self.label_file = params['label_file']
        self.attr_pc_file = params['attr_pc_file']
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', True)
        self.batch_size = int(params.get('batch_size', 10))
        self.resize = np.array([int(params.get('resize_width', 224)), int(params.get('resize_height', 224))])
        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two top: one data and one label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.img_indices = open(self.train_data_dir+'/'+self.label_file, 'r').read().splitlines()
        mean_file = open(self.mean_file, 'r')
        mean_value = mean_file.readline().split()
        mean_file.close()
        self.mean_value = np.array([float(mean_value[0]), float(mean_value[1]), float(mean_value[2])])
        self.idx = 0
        self.attr_cls = dict()
        self.attr_size = 0
        with open(self.train_data_dir+'/'+self.attr_pc_file, 'r') as f:
            for line in f.readlines():
                ac = line.split(',')
                cls = ac[0]
                spos = ac[1].find('[')
                epos = ac[1].find(']')
                attrs = ac[1][spos+1:epos].split()
                self.attr_cls[cls] = np.zeros((len(attrs), 1, 1), dtype=np.float32)
                self.attr_size = len(attrs)
                for idx in range(0, len(attrs)):
                    self.attr_cls[cls][idx] = float(attrs[idx])

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.randint(0, len(self.img_indices)-1, self.batch_size)
            
    def reshape(self, bottom, top):
        # load image + label image pair
        self.img, self.attr = self.load_data(self.idx)
        top[0].reshape(*self.img.shape)
        top[1].reshape(*self.attr.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img
        top[1].data[...] = self.attr
        
        # pick next input
        if self.random:
            self.idx = np.random.randint(0, len(self.img_indices)-1, self.batch_size)
        else:
            self.idx += 1
            if self.idx == len(self.img_indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_data(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        n = 0
        ims = np.zeros((self.batch_size,3,self.resize[1],self.resize[0]), dtype=np.float32)
        attrs = np.zeros((self.batch_size,self.attr_size,1,1), dtype=np.float32)
        for index in self.idx:

            line = self.img_indices[index]
            img_info = line.replace(' ', '').split(',')
            cls = img_info[1]
            spos = line.find('[')
            epos = line.find(']')
            bbox = line[spos+1:epos].split(',')
	    xmin = int(bbox[1])
            ymin = int(bbox[0])
            xmax = int(bbox[3])
            ymax = int(bbox[2])

            im = Image.open('{}/{}'.format(self.train_data_dir, img_info[len(img_info)-1]))
            
            #crop
            region = im.crop((xmin, ymin, xmax, ymax))
            
            region = region.resize(self.resize, Image.BILINEAR)
            in_ = np.array(region, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean_value
            in_ = in_.transpose((2,0,1))
            ims[n,:,:,:] = in_.copy()
            attrs[n,:,:,:] = self.attr_cls[cls]
            n = n + 1
        return ims, attrs

class AnnoClassificationDataReaderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.train_data_dir = params['train_data_dir']
        self.mean_file = params['mean_file']
        self.label_file = params['label_file']
        self.attr_pc_file = params['attr_pc_file']
        self.attr_file = params['attr_file']
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', True)
        self.batch_size = int(params.get('batch_size', 10))
        self.resize = np.array([int(params.get('resize_width', 224)), int(params.get('resize_height', 224))])
        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two top: one data and one label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.img_indices = open(self.train_data_dir+'/'+self.label_file, 'r').read().splitlines()
        mean_file = open(self.mean_file, 'r')
        mean_value = mean_file.readline().split()
        self.mean_value = np.array([float(mean_value[0]), float(mean_value[1]), float(mean_value[2])])
        self.idx = 0
        #attr_cls = open(self.train_data_dir+'/'+self.attr_pc_file, 'r').read().splitlines()
        self.attr_cls = dict()
        self.attr_size = 0
        with open(self.train_data_dir+'/'+self.attr_pc_file, 'r') as f:
            for line in f.readlines():
                ac = line.split(',')
                cls = ac[0]
                spos = ac[1].find('[')
                epos = ac[1].find(']')
                attrs = ac[1][spos+1:epos].split()
                self.attr_cls[cls] = np.zeros((len(attrs), 1, 1), dtype=np.float32)
                self.attr_size = len(attrs)
                for idx in range(0, len(attrs)):
                    self.attr_cls[cls][idx] = float(attrs[idx])

        self.attr_img = dict()
        with open(self.train_data_dir+'/'+self.attr_file, 'r') as f:
            for line in f.readlines():
                line_ = line.replace(' ','').split(',')
                img = line_[1]
                spos = line.find('[')
                epos = line.find(']')
                attrs = line[spos+1:epos].split()
                self.attr_img[img] = np.zeros((self.attr_size, 1, 1), dtype=np.float32)
                for idx in range(0, self.attr_size):
                    self.attr_img[img][idx] = float(attrs[idx])

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.randint(0, len(self.img_indices)-1, self.batch_size)
            
    def reshape(self, bottom, top):
        # load image + label image pair
        self.img, self.attr = self.load_data(self.idx)
        top[0].reshape(*self.img.shape)
        top[1].reshape(*self.attr.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img
        top[1].data[...] = self.attr
        
        # pick next input
        if self.random:
            self.idx = np.random.randint(0, len(self.img_indices)-1, self.batch_size)
        else:
            self.idx += 1
            if self.idx == len(self.img_indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_data(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        n = 0
        ims = np.zeros((self.batch_size,3,self.resize[1],self.resize[0]), dtype=np.float32)
        attrs = np.zeros((self.batch_size,self.attr_size,1,1), dtype=np.float32)
        for index in self.idx:

            line = self.img_indices[index]
            img_info = line.replace(' ', '').split(',')
            cls = img_info[1]
            spos = line.find('[')
            epos = line.find(']')
            bbox = line[spos+1:epos].split(',')
            xmin = int(bbox[1])
            ymin = int(bbox[0])
            xmax = int(bbox[3])
            ymax = int(bbox[2])

            img_name = img_info[len(img_info)-1]
            im = Image.open('{}/{}'.format(self.train_data_dir, img_name))
            
            #crop
            region = im.crop((xmin, ymin, xmax, ymax))
            
            region = region.resize(self.resize, Image.BILINEAR)
            in_ = np.array(region, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean_value
            in_ = in_.transpose((2,0,1))
            ims[n,:,:,:] = in_.copy()
            if img_name in self.attr_img.keys():
                attrs[n,:,:,:] = self.attr_img[img_name]
            else:
                attrs[n,:,:,:] = self.attr_cls[cls]
            n = n + 1
        return ims, attrs

