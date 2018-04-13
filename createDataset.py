import os
import sys
import random
import cv2 as cv
from PIL import Image, ImageDraw, ImageStat
from itertools import islice
from xml.dom.minidom import Document

def bbox(bbox_str):
	xy = bbox_str.split(',')
    
	return list([int(xy[1]), int(xy[0]), int(xy[3]), int(xy[2])])

def insertObject(doc, xy):
	obj = doc.createElement('object')
   	name = doc.createElement('name')
    	name.appendChild(doc.createTextNode('object'))
    	obj.appendChild(name)
    	pose = doc.createElement('pose')
    	pose.appendChild(doc.createTextNode('Unspecified'))
    	obj.appendChild(pose)
    	truncated = doc.createElement('truncated')
    	truncated.appendChild(doc.createTextNode(str(0)))
    	obj.appendChild(truncated)
    	difficult = doc.createElement('difficult')
    	difficult.appendChild(doc.createTextNode(str(0)))
    	obj.appendChild(difficult)
    	bndbox = doc.createElement('bndbox')
    
   	xmin = doc.createElement('xmin')
    	xmin.appendChild(doc.createTextNode(str(xy[0])))
    	bndbox.appendChild(xmin)
    
    	ymin = doc.createElement('ymin')                
    	ymin.appendChild(doc.createTextNode(str(xy[1])))
    	bndbox.appendChild(ymin)                
   	xmax = doc.createElement('xmax')                
    	xmax.appendChild(doc.createTextNode(str(xy[2])))
    	bndbox.appendChild(xmax)                
    	ymax = doc.createElement('ymax')    
    	ymax.appendChild(doc.createTextNode(str(xy[3])))
   	bndbox.appendChild(ymax)
    	obj.appendChild(bndbox)                
    	return obj

def save2xml(anno_name, xy, img_name):

	anno_file = open(anno_name, 'w')

	doc = Document()
	annotation = doc.createElement('annotation')
	doc.appendChild(annotation)

	folder = doc.createElement('folder')
	folder.appendChild(doc.createTextNode(root_path))
	annotation.appendChild(folder)

	filename = doc.createElement('filename')
	filename.appendChild(doc.createTextNode(img_name))
	annotation.appendChild(filename)

	source = doc.createElement('source')
	database = doc.createElement('database')
	database.appendChild(doc.createTextNode('{}-{}'.format('AI-CHALLENGE',cls.upper())))
	source.appendChild(database)
	source_annotation = doc.createElement('annotation')
	source_annotation.appendChild(doc.createTextNode(root_path))
	source.appendChild(source_annotation)

	image = doc.createElement('image')
	image.appendChild(doc.createTextNode(cls))
	source.appendChild(image)
	flickrid = doc.createElement('flickrid')
	flickrid.appendChild(doc.createTextNode('NULL'))
	source.appendChild(flickrid)
	annotation.appendChild(source)

	owner = doc.createElement('owner')
	owner.appendChild(flickrid)
	name = doc.createElement('name')
	name.appendChild(doc.createTextNode('AI-CHALLENGE'))
	owner.appendChild(name)
	annotation.appendChild(owner)

	size = doc.createElement('size')
	width = doc.createElement('width')
	width.appendChild(doc.createTextNode(str(resize)))
	size.appendChild(width)
	height = doc.createElement('height')
	height.appendChild(doc.createTextNode(str(resize)))
	size.appendChild(height)
	depth = doc.createElement('depth')
	depth.appendChild(doc.createTextNode(str(3)))
	size.appendChild(depth)
	annotation.appendChild(size)

	segmented = doc.createElement('segmented')
	segmented.appendChild(doc.createTextNode(str(0)))
	annotation.appendChild(segmented)
	annotation.appendChild(insertObject(doc, xy))

	try:
		anno_file.write(doc.toprettyxml(indent = '   '))
		anno_file.close()
	except:
		pass

def create():
	
	label_file = open('{}/{}'.format(sourcefile_path, labelfile_name))
	train_file = open('{}/{}'.format(imgset_path, trainfile_name), 'w')
	val_file = open('{}/{}'.format(imgset_path, valfile_name), 'w')
	mean_file = open('{}/{}'.format(root_path, meanfile_name), 'w')

	idx = 0

	imgs = label_file.readlines()
	img_num = len(imgs)
	img_list = range(0, img_num)
	random.shuffle(img_list)
	val_list = img_list[0:val_num]

	mean = list([0,0,0])

	for img_info in imgs:
		
		img_info = img_info.replace(' ', '')
		print  '{}--{}'.format(idx, img_info)
		spos = img_info.index('[')
		epos = img_info.index(']')
		xy = bbox(img_info[spos+1:epos])

		img_name = img_info[epos+2:-1]
		img = Image.open('{}/{}'.format(sourcefile_path, img_name))

		width = img.size[0]
		height = img.size[1]

		img = img.resize((resize, resize), Image.BILINEAR)

		xy[0] = min(int(round(float(resize)/float(width)*xy[0])), resize-1)
		xy[1] = min(int(round(float(resize)/float(height)*xy[1])), resize-1)
		xy[2] = min(int(round(float(resize)/float(width)*xy[2])), resize-1)
		xy[3] = min(int(round(float(resize)/float(height)*xy[3])), resize-1)
		"""
		iii = cv.imread('{}/{}'.format(sourcefile_path, img_name))
		iii = cv.resize(iii, (resize, resize))
		cv.line(iii, (xy[0],xy[1]), (xy[2],xy[3]), (0,255,0), 5)
		cv.imshow('ddd', iii)
		"""
		pos = img_name.index('/')
		subfile_name = img_name[:pos]
		if not os.path.exists('{}/{}'.format(imgfile_path, subfile_name)):
			os.makedirs('{}/{}'.format(imgfile_path, subfile_name))
		img.save('{}/{}'.format(imgfile_path, img_name))


		anno_name = img_name.replace('.jpg', '.xml')
		if not os.path.exists('{}/{}'.format(annofile_path, subfile_name)):
			os.makedirs('{}/{}'.format(annofile_path, subfile_name))
		save2xml('{}/{}'.format(annofile_path, anno_name), xy, img_name)

		stat = ImageStat.Stat(img)
		mean_ = [i/(resize*resize*img_num) for i in stat.sum]
		mean = [mean[i]+mean_[i] for i in range(0,3)]

		if idx in val_list:
			val_file.write('{}/{} {}/{}\n'.format(imgfile, img_name, annofile, anno_name))
		else:
			train_file.write('{}/{} {}/{}\n'.format(imgfile, img_name, annofile, anno_name))
	
		idx = idx+1

	mean_file.write(str(mean[2])+' '+str(mean[1])+' '+str(mean[0]))

	mean_file.close()
	val_file.close()
	train_file.close()
	label_file.close()

imgfile = 'JPEGImages'
annofile = 'Annotations'
imgset = 'ImageSets'
meanfile_name = 'bgr_mean.txt'
trainfile_name = 'train.txt'
valfile_name = 'val.txt'
targetpath = 'ssd_dataset'
resize = 300
val_num = 120

dataset = sys.argv[1]
cls = sys.argv[2]
sourcefile_path = sys.argv[3]
labelfile_name = sys.argv[4]

root_path = '{}/{}/{}_{}'.format(targetpath, dataset, cls, str(resize))
imgfile_path = '{}/{}'.format(root_path, imgfile)
imgset_path = '{}/{}'.format(root_path, imgset)
annofile_path = '{}/{}'.format(root_path, annofile)

if __name__ == '__main__':

	if not os.path.exists(root_path):
		os.makedirs(root_path)
	if not os.path.exists(annofile_path):
		os.makedirs(annofile_path)
	if not os.path.exists(imgset_path):
		os.makedirs(imgset_path)
	if not os.path.exists(imgfile_path):
		os.makedirs(imgfile_path)
	create()
