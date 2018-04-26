from __future__ import print_function
import sys
import os
sys.path.append('./caffe-ssd/python')
import caffe
from model_libs import *
from google.protobuf import text_format

import math
import shutil
import stat
import subprocess

def AddExtractImgFeatLayers(net, lr_mult=1, dropout=True):

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
    net.conv3_new1 = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    net.relu3_new1 = L.ReLU(net.conv3_new1, in_place=True)
    net.conv3_new2 = L.Convolution(net.relu3_new1, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    net.relu3_new2 = L.ReLU(net.conv3_new2, in_place=True)
    net.conv4_new = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    net.relu4_new = L.ReLU(net.conv4_new, in_place=True)
    net.concat = L.Concat(net.relu5_3, net.relu4_new, net.relu3_new2)
    net.conv6 = L.Convolution(net.concat, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    net.relu6 = L.ReLU(net.conv6, in_place=True)
    net.img_fc1 = L.InnerProduct(net.relu6, num_output=1024, **kwargs)
    net.img_relu1 = L.ReLU(net.img_fc1, in_place=True)
    if dropout:
      net.img_drop1 = L.Dropout(net.img_relu1, dropout_ratio=0.2, in_place=True)
    
    net.img_fc2 = L.InnerProduct(net.img_relu1, num_output=1024, **kwargs)
    net.img_relu2 = L.ReLU(net.img_fc2, in_place=True)
    if dropout:
      net.img_drop2 = L.Dropout(net.img_relu2, dropout_ratio=0.2, in_place=True)

def AddAttrEmbeddingLayers(net, attr_type='ori_attr', num_output=1000, lr_mult=1, dropout=True):

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
    from_layer = net.attr
    if attr_type=='ori_attr':
        net.attr_fc1 = L.InnerProduct(net.attr, num_output=512, **kwargs)
        net.attr_relu1 = L.ReLU(net.attr_fc1, in_place=True)
        if dropout:
            net.attr_drop1 = L.Dropout(net.attr_relu1, dropout_ratio=0.5, in_place=True)
        net.attr_fc2 = L.InnerProduct(net.attr_relu1, num_output=1024, **kwargs)
        net.attr_relu2 = L.ReLU(net.attr_fc2, in_place=True)
        if dropout:
            net.attr_drop2 = L.Dropout(net.attr_relu2, dropout_ratio=0.5, in_place=True)
	
        net.attr_fc3 = L.InnerProduct(net.attr_relu2, num_output=512, **kwargs)
        net.attr_relu3 = L.ReLU(net.attr_fc3, in_place=True)
        if dropout:
            net.attr_drop3 = L.Dropout(net.attr_relu3, dropout_ratio=0.5, in_place=True)
        net.attr_pred = L.InnerProduct(net.attr_relu3, num_output=num_output, **kwargs)

    elif attr_type=='ori_attr2':
	        
        net.attr_fc1 = L.InnerProduct(net.attr, num_output=1024, **kwargs)
        net.attr_relu1 = L.ReLU(net.attr_fc1, in_place=True)
        if dropout:
            net.attr_drop1 = L.Dropout(net.attr_relu1, dropout_ratio=0.5, in_place=True)
        net.attr_fc2 = L.InnerProduct(net.attr_relu1, num_output=1024, **kwargs)
        net.attr_relu2 = L.ReLU(net.attr_fc2, in_place=True)
        if dropout:
            net.attr_drop2 = L.Dropout(net.attr_relu2, dropout_ratio=0.5, in_place=True)
    else:

        net.attr_conv1 = L.Convolution(net.attr, num_output=64, pad=1, kernel_size=3, **kwargs)

        net.attr_relu1 = L.ReLU(net.attr_conv1, in_place=True)
        net.attr_conv2 = L.Convolution(net.attr_relu1, num_output=64, pad=1, kernel_size=3, **kwargs)
        net.attr_relu2 = L.ReLU(net.attr_conv2, in_place=True)
        net.attr_fc1 = L.InnerProduct(net.attr_relu2, num_output=4096)
        net.attr_relu3 = L.ReLU(net.attr_fc1, in_place=True)
        if dropout:
            net.attr_drop1 = L.Dropout(net.attr_relu3, dropout_ratio=0.5, in_place=True)
        net.attr_fc2 = L.InnerProduct(net.attr_relu3, num_output=4096)
        net.attr_relu4 = L.ReLU(net.attr_fc2, in_place=True)
        if dropout:
            net.attr_drop2 = L.Dropout(net.attr_relu4, dropout_ratio=0.5, in_place=True)

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = 'caffe-ssd'

dataset = sys.argv[1]
cls = sys.argv[2]
attr_type = sys.argv[3]
gpus = sys.argv[4]
  
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = False
# If true, Remove old model files.
remove_old_models = True

root_dir = os.getcwd()

# The database file for training data. Created by data/VOC0712/create_data.sh
train_data_dir = "{}/data/{}/{}_train/images".format(root_dir, dataset, cls, cls)

# Specify the batch sampler.
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)

python_module = "{}".format(root_dir)
mean_file = "{}/lmdb-dataset/{}/{}_{}/bgr_mean.txt".format(root_dir, dataset, cls, str(resize_width))

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.0001

# Modify the job name if you want.
job_name = "ZSL_{}_{}".format(attr_type.upper(), resize)
# The name of the model. Modify it if you want.
model_name = "{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "{}/models/{}/{}_{}/{}".format(root_dir, dataset, cls, str(resize_width), model_name)
# Directory which stores the snapshot of models.
snapshot_dir = "{}/snapshot".format(save_dir)
# Directory which stores the job script and log file.
job_dir = "{}/jobs/{}/{}_{}/{}".format(root_dir, dataset, cls, str(resize_width), job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

pretrain_model = "{}/models/{}/{}_{}/ATTR_PRED2_300x300/snapshot/ATTR_PRED2_300x300_iter_80000.caffemodel".format(root_dir, dataset, cls, str(resize_width))
# Solver parameters.
# Defining which GPUs to use.
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 8
accum_batch_size = 8
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [40000, 80000, 120000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 120000,
    'snapshot': 80000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    }

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data_dir)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.img, net.attr = L.Python(python_param=dict(
        module="datareader_layers",
        layer="OriDataReaderLayer",
        param_str="{{\'train_data_dir\': \'{}\',\'mean_file\': \'{}\', \'label_file\': \'{}\', \'attr_pc_file\': \'{}\', \'batch_size\': \'{}\', \'resize_height\': \'{}\', \'resize_width\': \'{}\'}}".format(train_data_dir, mean_file, 'labels.txt', 'attributes_per_class.txt', str(batch_size), str(resize_height), str(resize_width))),
        ntop=2)

with open('{}/{}'.format(train_data_dir, 'attributes_per_class.txt')) as f:
	line = f.readline()
	spos = line.find('[')
	epos = line.find(']')
	attr = line[spos+1:epos].split()
	attr_size = len(attr)
VGGNetBody(net, from_layer='img', need_fc=False, lr_mult=0)

AddExtractImgFeatLayers(net, lr_mult=0, dropout=False)

AddAttrEmbeddingLayers(net, attr_type=attr_type, lr_mult=lr_mult, dropout=False, num_output=attr_size)

net.loss1 = L.SmoothL1Loss(net.img_relu2, net.attr_relu2, propagate_down = [False, True])
if attr_type == 'ori_attr':
	net.loss2 = L.SmoothL1Loss(net.attr_pred, net.attr)


with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['img', 'attr'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width]),
	caffe_pb2.BlobShape(dim=[1, attr_size, 1, 1])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('export PYTHONPATH={}:$PYTHONPATH\n'.format(python_module))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)
# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
