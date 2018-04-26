#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
sys.path.append('./caffe-ssd/python')
import argparse
import numpy as np
from PIL import Image, ImageDraw
from sklearn import preprocessing
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, 
        mean_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        mean_value = open(mean_file, 'r').readline().split()
        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

        self.mean_value = np.array([float(mean_value[0]), float(mean_value[1]), float(mean_value[2])]) # mean pixel

        

    def detect(self, image_file, dataset, pred_bbox=''):
        '''
        Dist computation
        '''
        # set net to batch size of 1
        # image_resize = 300
        img = Image.open(image_file)
	img = img.convert('RGB')

        if dataset == 'test':
            with open(pred_bbox, 'r') as f:
                bbox_value = f.readlines()[1].split()
                img_size = np.array([int(bbox_value[0]), int(bbox_value[1])])
                assert(img.size[0]==img_size[0])
                assert(img.size[1]==img_size[1])
                xmin = max(int(bbox_value[2]), 0)
                ymin = max(int(bbox_value[3]), 0)
                xmax = min(int(bbox_value[4]), img_size[0]-1)
                ymax = min(int(bbox_value[5]), img_size[1]-1)
        else:
                xmin = int(pred_bbox[1])
                ymin = int(pred_bbox[0])
                xmax = int(pred_bbox[3])
                ymax = int(pred_bbox[2])

        #crop
        region = img.crop((xmin, ymin, xmax, ymax))
            
        region = region.resize([self.image_resize, self.image_resize], Image.BILINEAR)
        in_ = np.array(region, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_value
        in_ = in_.transpose((2,0,1))
    
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)

        #Run the net and examine the top_k results
        self.net.blobs['data'].data[...] = in_.copy()

        outputs = self.net.forward()
        feat = outputs['pool5/7x7_s1'].reshape((1,1024))       

        return feat

def normlize_feat(feats, imgs, labels, args):
    norm_feats = preprocessing.scale(feats)
    norm_feats_file = open(args.train_file+'/norm_feats.txt', 'w')
    feats_file = open(args.train_file+'/feats.txt', 'w')

    norm_feats_file.write('mean: [ ')
    for value in feats.mean(axis=0):
        norm_feats_file.write('{} '.format(str(value)))
    norm_feats_file.write(']\n')

    norm_feats_file.write('std: [ ')
    for value in feats.std(axis=0):
        norm_feats_file.write('{} '.format(str(value)))
    norm_feats_file.write(']\n')

    known_labels = []
    for idx in range(len(imgs)):        
        label = labels[idx]
        img = imgs[idx]
        if not label in known_labels:
            known_labels.append(label)

        norm_feats_file.write(img+', '+label+', [ ')
        for value in norm_feats[idx,:]:
            norm_feats_file.write('{} '.format(str(value)))
        norm_feats_file.write(']\n')

        feats_file.write(img+', '+label+', [ ')
        for value in feats[idx,:]:
            feats_file.write('{} '.format(str(value)))
        feats_file.write(']\n')

    feats_file.close()
    norm_feats_file.close()
    print norm_feats.mean(axis=0), norm_feats.std(axis=0)
    return known_labels

def normlize_attr(known_labels, args):

    # normlize attribute
    idx = 0
    kl = []
    ukl = dict()
    with open(args.train_file+'/'+args.attribute_file, 'r') as f:
        for line in f.readlines():
            al = line.split(',')
            lab = al[0]
            spos = al[1].find('[')
            epos = al[1].find(']')
            attr_str = al[1][spos+1:epos].split()
            if lab in known_labels:
                
                attr = np.zeros((1,len(attr_str)), dtype=np.float32)
                for i in range(len(attr_str)):
                    attr[0,i] = float(attr_str[i])
                if idx == 0:
                    attrs = attr
                    idx = idx+1
                else:
                    attrs = np.vstack((attrs, attr))

                kl.append(lab)
            else:
                attr = np.zeros((1,len(attr_str)), dtype=np.float32)
		for i in range(len(attr_str)):
			attr[0,i] = float(attr_str[i])
		ukl[lab] = attr
    norm_attrs = preprocessing.scale(attrs)
    with open(args.train_file+'/norm_attrs.txt', 'w') as f:
        f.write('mean: [ ')
        for value in attrs.mean(axis=0):
            f.write('{} '.format(str(value)))
        f.write(']\n')
        f.write('std: [ ')
        for value in attrs.std(axis=0):
            f.write('{} '.format(str(value)))
        f.write(']\n')
        for i in range(len(kl)):
            f.write('{}: [ '.format(kl[i]))
            for value in norm_attrs[i,:]:
                f.write('{} '.format(str(value)))
            f.write(']\n')
    print norm_attrs.mean(axis=0), norm_attrs.std(axis=0)

    mean = attrs.mean(axis=0)
    std = attrs.std(axis=0)
    with open(args.train_file+'/norm_unknown_attrs.txt', 'w') as f:
	for lab in ukl.keys():
		f.write(lab+': ')
		attr = ukl[lab]
		f.write('[ ')
		print attr
		print attr.size
		for i in range(attr.size):
			norm_v = (attr[0,i]-mean[i])/(std[i]+1e-6)
			f.write('{} '.format(str(norm_v)))
		f.write(']\n')
def normlize(feats, imgs, labels, args):
    
    known_labels = normlize_feat(feats, imgs, labels, args)
    normlize_attr(known_labels, args)        

def main(args):
    '''main '''
    

    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.mean_file)   
    
    if args.dataset == 'train':
        

        #normlize feature
        feats = np.zeros((1, 1024), dtype=np.float32)
        idx = 0
        imgs = []
        labels = []
        with open(args.train_file+'/'+args.label_file, 'r') as f:
            for line in f.read().splitlines():
                img_info = line.replace(' ', '').split(',')
                spos = line.find('[')
                epos = line.find(']')
                bbox = line[spos+1:epos].split(',')
                
                feat = detection.detect(args.train_file+'/'+img_info[len(img_info)-1], args.dataset, bbox)

                feat = np.array(feat, dtype=np.float32)
                if idx == 0:
                    feats = feat
                    idx = idx + 1
                else:
                    feats = np.vstack((feats, feat))
                imgs.append(img_info[len(img_info)-1])
                labels.append(img_info[1])
        normlize(feats, imgs, labels, args)
        
    elif args.dataset == 'test':
        
    	img_list = os.listdir(args.images_file)
	
	with open(args.train_file+'/norm_feats.txt', 'r') as f:
		line = f.readline()
		spos = line.find('[')
		epos = line.find(']')
		mean_str = line[spos+1:epos].split()
		mean = []
		for v in mean_str:
			mean.append(float(v))
		line = f.readline()
		spos = line.find('[')
		epos = line.find(']')
		std_str = line[spos+1:epos].split()
		std = []
		for v in std_str:
			std.append(float(v))
    	prediction = open(args.images_file+'/norm_feat.txt', 'w')
    	for img_name in img_list:
        	if os.path.splitext(img_name)[1] == ".jpg":

            		feat = detection.detect(args.images_file+'/'+img_name, args.dataset, args.pred_bbox_file+'/'+img_name.replace('jpg', 'txt'))
			# normlize
			feat_norm = np.zeros_like(feat, dtype=np.float32)
			for i in range(feat.size):
				feat_norm[0,i] = (feat[0,i]-mean[i])/(std[i]+1e-6) 
			prediction.write(img_name+': ')
			prediction.write('[ ')
			for i in range(feat.size):
				prediction.write('{} '.format(str(feat_norm[0,i])))
			prediction.write(']\n')
	prediction.close()
	
    else:
        img_list = os.listdir(args.images_file)
        imgdir_list = os.listdir(args.images_file)
        feats = np.zeros((1, 1024), dtype=np.float32)
        idx = 0
        for img_name in img_list:
            feat = detection.detect(args.images_file+'/'+img_name, args.dataset)
            if idx == 0:
		feats = feat
                idx = idx + 1
            else:
                feats = np.vstack((feats, feat))
        norm_feat = preprocessing.scale(feats)
        print norm_feat.mean(axis=0), norm_feat.std(axis=0)
    

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='models/VGGNet/VOC0712/ZSL_ORI_ATTR_300x300/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='models/VGGNet/VOC0712/ZSL_ATTR_ORI_300x300/'
                        'VGG_VOC0712_ZSL_ATTR_ORI_300x300_iter_120000.caffemodel')
    parser.add_argument('--images_file', default='examples/images')
    parser.add_argument('--attribute_file', default='attributes.txt')
    parser.add_argument('--mean_file', default='bgr_mean.txt')
    parser.add_argument('--result_file', default='result.txt')
    parser.add_argument('--pred_bbox_file', default='detect_results/')
    parser.add_argument('--dataset', default='test')
    parser.add_argument('--label_file', default='labels.txt')
    parser.add_argument('--train_file', default='images/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
