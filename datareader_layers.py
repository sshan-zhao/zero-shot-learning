import sys
sys.path.append('/home/zss/zsl/caffe-ssd/python')
import caffe

import numpy as np
from PIL import Image
import scipy.io as sco
import scipy.misc as smc
import random
import os

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
	    im = im.convert('RGB')            
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
            n += 1
        return ims, attrs

class ClsDataReaderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.train_data_dir = params['train_data_dir']
        self.mean_file = params['mean_file']
        self.label_file = params['label_file']
        self.label_list_file = params['label_list_file']
        self.seed = params.get('seed', None)
        self.cls_num = params.get('cls_num', 1000)
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
        
        self.label_list = dict()
        with open(self.train_data_dir+'/'+self.label_list_file, 'r') as f:
            idx = 0
            for line in f.readlines():
                info = line.strip().split(',')
                lab = info[0]
                self.label_list[lab] = idx
                idx += 1

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.randint(0, len(self.img_indices)-1, self.batch_size)
            
    def reshape(self, bottom, top):
        # load image + label image pair
        self.img, self.lab = self.load_data(self.idx)
        top[0].reshape(*self.img.shape)
        top[1].reshape(*self.lab.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img
        top[1].data[...] = self.lab
        
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
        labs = np.zeros((self.batch_size,1,1,1), dtype=np.float32)
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
            im = im.convert('RGB')
            #crop
            region = im.crop((xmin, ymin, xmax, ymax))
            
            region = region.resize(self.resize, Image.BILINEAR)
            in_ = np.array(region, dtype=np.float32)
            in_ = in_[:,:,::-1]
            in_ -= self.mean_value
            in_ = in_.transpose((2,0,1))
            ims[n,:,:,:] = in_.copy()
            labs[n,:,:,:] = self.label_list[cls]
            n += 1
        return ims, labs

class QuadrupletDataReaderLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.train_data_dir = params['train_data_dir']
        self.feat_file = params['feat_file']
        self.attr_file = params['attr_file']
        self.quadruplet_list_dir = params['quadruplet_list_dir']
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', False)
        self.batch_size = int(params.get('batch_size', 10))
        
        # two tops: data and label
        if len(top) != 6:
            raise Exception("Need to define six top: y, xi, xj, xk, delta_j, delta_k.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        self.attr2label = dict()
        with open(self.train_data_dir+'/'+self.attr_file, 'r') as f:
            for line in f.read().splitlines()[2:]:
                lab = line.split(':')[0].replace(' ','')
                attr_str = line.split(':')[1].replace('[','').replace(']','').split()
                self.attr2label[lab] = np.zeros([len(attr_str),1,1],dtype=np.float32)
                self.attr_size = len(attr_str)
                for i in range(len(attr_str)):
                    self.attr2label[lab][i] = float(attr_str[i])
	self.img2feat = dict()
        with open(self.train_data_dir+'/'+self.feat_file, 'r') as f:
            for line in f.read().splitlines()[2:]:
                img = line.split(',')[0].replace(' ','')
                feat_str = line.split(',')[2].replace('[','').replace(']','').split()
                self.img2feat[img] = np.zeros([len(feat_str),1,1],dtype=np.float32)
                self.feat_size = len(feat_str)
                for i in range(len(feat_str)):
                    self.img2feat[img][i] = float(feat_str[i])

        # Find most recent quadruplet list.
        max_epoch = -1
        for file in os.listdir(self.quadruplet_list_dir):
            if file.endswith(".txt"):
                basename = os.path.splitext(file)[0]
                epoch = int(basename.split("quadruplet_list_")[1])
                if max_epoch < epoch:
                    max_epoch = epoch
	if max_epoch >= 0:
        	self.quadruplet_list = open( "{}/quadruplet_list_{}.txt".format(self.quadruplet_list_dir,max_epoch), 'r').read().splitlines()
	else:
		print lab, img
		self.quadruplet_list = []
		self.quadruplet_list.append("{}, {}, {}, {}, 0, 0".format(lab, img, img, img))
        print len(self.quadruplet_list)
	self.quadruplet_order = np.random.randint(0, len(self.quadruplet_list),len(self.quadruplet_list))
        self.idx = 0
        self.l_ord = 0
        self.h_ord = self.batch_size
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = np.random.randint(0, len(self.quadruplet_order)-1, self.batch_size)
        else:
            self.idx = self.quadruplet_order[0:min(self.batch_size, len(self.quadruplet_list))]
            
    def reshape(self, bottom, top):
        # load image + label image pair
        self.attr, self.xi, self.xj, self.xk, self.delta_j, self.delta_k = self.load_data(self.idx)
	#print self.attr.shape, self.xi.shape, self.xj.shape, self.xk.shape
        top[0].reshape(*self.attr.shape)
        top[1].reshape(*self.xi.shape)
        top[2].reshape(*self.xj.shape)
        top[3].reshape(*self.xk.shape)
        top[4].reshape(*self.delta_j.shape)
        top[5].reshape(*self.delta_k.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.attr
        top[1].data[...] = self.xi
        top[2].data[...] = self.xj
        top[3].data[...] = self.xk
        top[4].data[...] = self.delta_j
        top[5].data[...] = self.delta_k
        
        # pick next input
        if self.random:
            self.idx = np.random.randint(0, len(self.quadruplet_order)-1, self.batch_size)
        else:
            self.l_ord = self.h_ord
            self.h_ord = min(self.l_ord+self.batch_size, len(self.quadruplet_order))
            self.idx = self.quadruplet_order[self.l_ord:self.h_ord]

            if self.l_ord >= len(self.quadruplet_order):
                self.l_ord = 0
                self.h_ord = self.l_ord+self.batch_size
                self.quadruplet_order = np.random.randint(0, len(self.quadruplet_list),
                                     len(self.quadruplet_list))
                self.idx = self.quadruplet_order[self.l_ord:self.h_ord]

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
        attrs = np.zeros((self.idx.size,self.attr_size), dtype=np.float32)
        xis = np.zeros((self.idx.size,self.feat_size), dtype=np.float32)
        xjs = np.zeros((self.idx.size,self.feat_size), dtype=np.float32)
        xks = np.zeros((self.idx.size,self.feat_size), dtype=np.float32)
        delta_js = np.zeros((self.idx.size,1), dtype=np.float32)
        delta_ks = np.zeros((self.idx.size,1), dtype=np.float32)
        
	#print '---------------------', self.idx, '--------------------'
	for index in self.idx:
	  #  print n
            line = self.quadruplet_list[index].replace(' ','').split(',')
          #  print line
	    attr = self.attr2label[line[0]]
            xi = self.img2feat[line[1]]
	  #  print xi
            xj = self.img2feat[line[2]]
	  #  print xj
            xk = self.img2feat[line[3]]
            delta_j = float(line[4])
	 #   print delta_j 
            delta_k = float(line[5])
	#    print '-------------------------------------------'
            attrs[n,:] = attr.reshape(self.attr_size)
            xis[n,:] = xi.reshape(self.feat_size)
            xjs[n,:] = xj.reshape(self.feat_size)
            xks[n,:] = xk.reshape(self.feat_size)
            delta_js[n,0] = delta_j
            delta_ks[n,0] = delta_k
            n += 1
	#print xis
	#print xjs
	#print '--------------------------------------------------------'
        return attrs, xis, xjs, xks, delta_js, delta_ks
