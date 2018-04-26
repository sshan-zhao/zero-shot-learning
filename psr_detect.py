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
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def detect(self, attr_file):
        '''
        Dist computation
        '''

        lab2embed = dict()
        with open(attr_file, 'r') as f:
            for line in f.read().splitlines():
                lab = line.replace(' ','').split(':')[0]
                attr_str = line.split(':')[1].replace('[','').replace(']','').split()
		attr = np.zeros([1, len(attr_str)], dtype=np.float32)
		for i in range(len(attr_str)):
			attr[0,i] = float(attr_str[i])

                self.net.blobs['attr'].reshape(1, attr.size)

                #Run the net and examine the top_k results
                self.net.blobs['attr'].data[...] = attr.reshape(1, attr.size)

                outputs = self.net.forward()
                embed = outputs['attr_embed']

                lab2embed[lab] = embed.copy()
		print embed
	print lab2embed
        return lab2embed

def cossim(x, y):

  dot_result = 0.0
  normx_result = 0.0
  normy_result = 0.0
  for a,b in zip(x,y):
    dot_result += a*b
    normx_result += a**2
    normy_result += b**2

  return dot_result/((normx_result*normy_result)**0.5+1e-6)

def main(args):
    '''main '''
    

    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights)
    lab2embed = detection.detect(args.attribute_file)
    result = open(args.images_file+'/'+args.result_file, 'w')
    with open(args.images_file+'/'+args.feat_file, 'r') as f:
        for line in f.read().splitlines():
            img_name = line.split(':')[0].replace(' ','')
            result.write(img_name+' ')
            feat_str = line.split('[')[1].replace(']','').split()
            feat = np.zeros([1, len(feat_str)], dtype=np.float32)
            lab2sim = dict()
            for i in range(len(feat_str)):
                feat[0,i] = float(feat_str[i])

            for lab in lab2embed.keys():
                lab2sim[lab] = cossim(lab2embed[lab].reshape(feat.size,), feat.reshape(feat.size,))
		#print lab
		#print lab2embed[lab]
		#print feat
		#print lab2sim[lab]

            lab2sim = sorted(lab2sim.items(), lambda x, y:cmp(x[1], y[1]), reverse=True)

            #print lab2sim

            result.write(lab2sim[0][0]+'\n')
    result.close()
    

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='models/VGGNet/VOC0712/ZSL_ORI_ATTR_300x300/deploy.prototxt')
    parser.add_argument('--model_weights',
                        default='models/VGGNet/VOC0712/ZSL_ATTR_ORI_300x300/'
                        'VGG_VOC0712_ZSL_ATTR_ORI_300x300_iter_120000.caffemodel')
    parser.add_argument('--images_file', default='examples/images')
    parser.add_argument('--feat_file', default='norm_feat.txt')
    parser.add_argument('--attribute_file', default='attributes.txt')
    parser.add_argument('--result_file', default='results.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
