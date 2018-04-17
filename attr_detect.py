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
    def __init__(self, gpu_id, model_def, model_weights, image_resize, 
        mean_file, attribute_file, label_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        mean_value = open(mean_file, 'r').readline().split()
        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)

        self.mean_value = np.array([float(mean_value[0]), float(mean_value[1]), float(mean_value[2])]) # mean pixel

        self.known_labels = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                img_info = line.replace(' ', '').split(',')
                label = img_info[1]
                if not label in self.known_labels:
		    print label
                    self.known_labels.append(label)
        print len(self.known_labels)
        idx = 0
        self.attr_lab_set = dict()
        self.attr_size = 0
        with open(attribute_file, 'r') as f:
            for line in f.readlines():
                al = line.split(',')
                lab = al[0]
                spos = al[1].find('[')
                epos = al[1].find(']')
                attrs = al[1][spos+1:epos].split()
                if not lab in self.known_labels:
                    self.attr_lab_set[lab] = np.zeros((len(attrs), 1, 1), dtype=np.float32)
                    self.attr_size = len(attrs)
                    for idx in range(0, len(attrs)):
                        self.attr_lab_set[lab][idx] = float(attrs[idx])
                else:
                    pass

    def detect(self, image_file, pred_bbox_file):
        '''
        Dist computation
        '''
        # set net to batch size of 1
        # image_resize = 300
        img = Image.open(image_file)

        with open(pred_bbox_file, 'r') as f:
            bbox_value = f.readlines()[1].split()
            img_size = np.array([int(bbox_value[0]), int(bbox_value[1])])
            assert(img.size[0]==img_size[0])
            assert(img.size[1]==img_size[1])
            xmin = max(int(bbox_value[2]), 0)
            ymin = max(int(bbox_value[3]), 0)
            xmax = min(int(bbox_value[4]), img_size[0]-1)
            ymax = min(int(bbox_value[5]), img_size[1]-1)

        #crop
        region = img.crop((xmin, ymin, xmax, ymax))
            
        region = region.resize([self.image_resize, self.image_resize], Image.BILINEAR)
        in_ = np.array(region, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean_value
        in_ = in_.transpose((2,0,1))
        #ims[1,:,:,:] = in_.copy()

        self.net.blobs['img'].reshape(1, 3, self.image_resize, self.image_resize)

        #Run the net and examine the top_k results
        self.net.blobs['img'].data[...] = in_.copy()

        lab_dist = dict()
        for lab in self.attr_lab_set:

            self.net.blobs['attr'].data[...] = self.attr_lab_set[lab]
            outputs = self.net.forward()
            pred = outputs['pred']
            gt = self.attr_lab_set[lab]
            dist = np.sum((pred-gt)**2)
            lab_dist[lab] = dist
        return sorted(lab_dist.items(), lambda x, y: cmp(x[1], y[1])), np.array([xmin, ymin, xmax, ymax])

def main(args):
    '''main '''
    

    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.mean_file, args.attribute_file,
                               args.label_file)

    img_list = os.listdir(args.images_file)

    prediction = open(args.save_file+'/'+args.result_file, 'w')
    for img_name in img_list:
        if os.path.splitext(img_name)[1] == ".jpg":

            result, bbox = detection.detect(args.images_file+'/'+img_name, args.pred_bbox_file+'/'+img_name.replace('jpg', 'txt'))

            img = Image.open(args.images_file+'/'+img_name)
            draw = ImageDraw.Draw(img)
            width, height = img.size
            print width, height
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 0, 0))
            for i in range(0,5):
                draw.text([bbox[0]+i*30, bbox[1]+i*30], result[i][0]+str(result[i][1]), (0, 0, 255))
            print [bbox[0], bbox[1]], result[i][0]
	    prediction.write("{} {}\n".format(img_name, result[0][0]))
            img.save("{}/{}.jpg".format(args.save_file, img_name))
    prediction.close()

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
    parser.add_argument('--save_file', default='detect_results/images')
    parser.add_argument('--attribute_file', default='attributes.txt')
    parser.add_argument('--mean_file', default='bgr_mean.txt')
    parser.add_argument('--label_file', default='labels.txt')
    parser.add_argument('--result_file', default='result.txt')
    parser.add_argument('--pred_bbox_file', default='detect_results/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
