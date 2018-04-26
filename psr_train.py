from __future__ import print_function
import sys
import os
sys.path.append('./caffe-ssd/python')
import caffe
from model_libs import *
from google.protobuf import text_format
import numpy as np
import math
import shutil
import stat
import subprocess
def ReadFeat(feat_file):

  data_info = dict()
  with open(feat_file, 'r') as f:
    for line in f.read().splitlines()[2:]:
      info = line.split(',')
      img = info[0].replace(' ','')
      label = info[1].replace(' ','')
      feat_str = info[2].replace('[','').replace(']','').split()
      feat = np.zeros([1, len(feat_str)], dtype=np.float32)
      for i in range(len(feat_str)):
        feat[0,i] = float(feat_str[i])
      if label in data_info.keys():
        data_info[label][img] = feat
      else:
        data_info[label] = dict()
        data_info[label][img] = feat
      
  return data_info

def ReadAttr(attr_file):
  attr2label = dict()
  with open(attr_file, 'r') as f:
    for line in f.read().splitlines()[2:]:
      info = line.split(':')
      label = info[0].replace(' ','')
      attr_str = info[1].replace('[','').replace(']','').split()
      attr = np.zeros([1, len(attr_str)], dtype=np.float32)
      for i in range(len(attr_str)):
        attr[0,i] = float(attr_str[i])
      attr2label[label] = attr
  return attr2label

def cossim(x, y):

  dot_result = 0.0
  normx_result = 0.0
  normy_result = 0.0
  for a,b in zip(x,y):
    dot_result += a*b
    normx_result += a**2
    normy_result += b**2

  return dot_result/((normx_result*normy_result)**0.5+1e-6)

def GenerateSample(sim_matrix, data_info, lab_set, embed, label, all_num_img, th_low, th_high):
  max_loss = 0.0
  delta = 0.0
  img = ''
  for lb in lab_set:
  	num_img = len(data_info[lb])
        num_samp = round(float(num_img)/float(all_num_img)*30)
        samp = np.random.randint(0,int(num_img),int(num_samp))
        idx = 0
        for img_ in data_info[lb].keys():
        	if idx in samp:
          		feat = data_info[lb][img_]
	  		sim_embed_feat = cossim(embed.reshape(feat.size,), feat.reshape(feat.size,))
          		loss = max(0.0,threshold-sim_embed_feat)+max(0.0,sim_embed_feat-sim_matrix[label][lb])
          		if loss > max_loss:
            			max_loss = loss
            			img = img_
            			delta = sim_matrix[label][lb]
        	idx += 1

  return img, delta
def GenerateNewQuadrupletList(model_weights, model_def, feat_file, 
  attr_file, save_file, attr_size, threshold=0.0, gpu_id=0):
  caffe.set_device(gpu_id)
  caffe.set_mode_gpu()
        
  net = caffe.Net(model_def,      # defines the structure of the model
                  model_weights,  # contains the trained weights
                  caffe.TEST)
  net.blobs['attr'].reshape(1, attr_size, 1, 1)

  data_info = ReadFeat(feat_file)
  attr2label = ReadAttr(attr_file)

  sim_matrix = dict()
  for lab1 in attr2label.keys():
    sim_matrix[lab1] = dict()
    for lab2 in attr2label.keys():
      if lab1 == lab2:
        sim_matrix[lab1][lab2] = 1.0
      else:
        sim_matrix[lab1][lab2] = cossim(attr2label[lab1].reshape(attr_size,), attr2label[lab2].reshape(attr_size,))

  with open(save_file, 'w') as f:
    for label in attr2label.keys():
      for img in data_info[label].keys():
        attr = attr2label[label]
        net.blobs['attr'].data[...] = attr.reshape(1, attr_size, 1, 1)

        outputs = net.forward()
        embed = outputs['attr_embed']

        # write label
        f.write(label+', ')
        # write sample x_i
        f.write(img+', ')
        
        num_sim_img = 0
        num_dissim_img = 0
	sim_lab_set = []
	dissim_lab_set = []
        for lb in sim_matrix[label].keys():
          if sim_matrix[label][lb] >= threshold and sim_matrix[label][lb] < 1.0:
            #num_sim_img += len(data_info[lb])
	    sim_lab_set.append(lb)
          elif sim_matrix[label][lb] < threshold:
            #num_dissim_img += len(data_info[lb])
	    dissim_lab_set.append(lb)
	sim_rand_ord = np.random.randint(0,len(sim_lab_set),int(math.ceil(len(sim_lab_set)*0.5)))
	dissim_rand_ord = np.random.randint(0,len(dissim_lab_set),int(math.ceil(len(dissim_lab_set)*0.5)))
	sim_lab_set = [sim_lab_set[x] for x in sim_rand_ord]
	dissim_lab_set = [dissim_lab_set[x] for x in dissim_rand_ord]
	for lb in sim_lab_set:
		num_sim_img += len(data_info[lb])
	for lb in dissim_lab_set:
		num_dissim_img += len(data_info[lb])
        img_xj, delta_j = GenerateSample(sim_matrix, data_info, sim_lab_set, embed, label, num_sim_img, threshold, 1.0)
        img_xk, delta_k = GenerateSample(sim_matrix, data_info, dissim_lab_set, embed, label, num_dissim_img, -1.0, threshold)

        # write sample x_j
        f.write(img_xj+', ')
        # write sample x_k
        f.write(img_xk+', ')
        # wirte delta
        f.write('{}, '.format(str(delta_j)))
        f.write('{}\n'.format(str(delta_k)))

def AddAttrEnDecoderLayers(net, num_output=1000, lr_mult=1):

    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_filler': dict(type='constant', value=0)}

    net.attr_en1 = L.InnerProduct(net.attr, num_output=512, **kwargs)
    net.attr_relu1 = L.ReLU(net.attr_en1, in_place=True)
    
    net.attr_en2 = L.InnerProduct(net.attr_en1, num_output=1024, **kwargs)
    net.attr_relu2 = L.ReLU(net.attr_en2, in_place=True)

    net.attr_embed = L.InnerProduct(net.attr_en2, num_output=1024, **kwargs)
    net.attr_embedelu = L.ELU(net.attr_embed, in_place=True)

    net.attr_de1 = L.InnerProduct(net.attr_embed, num_output=1024, **kwargs)
    net.attr_elu1 = L.ELU(net.attr_de1, in_place=True)

    net.attr_de2 = L.InnerProduct(net.attr_de1, num_output=512, **kwargs)
    net.attr_elu2 = L.ELU(net.attr_de2, in_place=True)

    net.attr_pred = L.InnerProduct(net.attr_de2, num_output=num_output, **kwargs)

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = 'caffe-ssd'

dataset = sys.argv[1]
cls = sys.argv[2]
threshold = float(sys.argv[3])
loss_weight = float(sys.argv[4])
gpus = sys.argv[5]
  
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

root_dir = os.getcwd()

# The database file for training data. Created by data/VOC0712/create_data.sh
train_data_dir = "{}/data/{}/{}_train/images".format(root_dir, dataset, cls, cls)


python_module = "{}".format(root_dir)

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00001

# Modify the job name if you want.
job_name = "PSR_{}_{}".format(threshold, loss_weight)
# The name of the model. Modify it if you want.
model_name = "{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "{}/models/{}/{}/{}".format(root_dir, dataset, cls, model_name)
# Directory which stores the snapshot of models.
snapshot_dir = "{}/snapshot".format(save_dir)
# Directory which stores the job script and log file.
job_dir = "{}/jobs/{}/{}/{}".format(root_dir, dataset, cls, job_name)

# Directory which stores the generated quadruplet list file
quadruplet_dir = "{}/quadruplet".format(save_dir)
# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Solver parameters.
# Defining which GPUs to use.
gpulist = gpus.split(",")
num_gpus = len(gpulist)

#threshold = 0.1
loss_weight1 = loss_weight
loss_weight2 = 0.01
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

with open('{}/{}'.format(train_data_dir, 'attributes_per_class.txt')) as f:
  line = f.readline()
  spos = line.find('[')
  epos = line.find(']')
  attr = line[spos+1:epos].split()
  attr_size = len(attr)
with open('{}/{}'.format(train_data_dir, 'labels.txt')) as f:
  num_sample = len(f.readlines())

max_epoch = 10
iter_per_epoch = int(math.ceil(float(num_sample) / float(batch_size)))

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.00005,
    'lr_policy': "step",
    'stepsize': iter_per_epoch*5,
    'gamma': 0.5,
    'momentum': 0.9,
    'momentum2': 0.999,
    'iter_size': iter_size,
    'snapshot': iter_per_epoch,
    'display': 10,
    #'type': "ADAM",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    }

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data_dir)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)
make_if_not_exist(quadruplet_dir)
# Create train net.
net = caffe.NetSpec()
net.attr, net.iden, net.sim, net.dissim, net.delta_sim, net.delta_dissim = L.Python(python_param=dict(
        module="datareader_layers",
        layer="QuadrupletDataReaderLayer",
        param_str="{{\'train_data_dir\': \'{}\', \'attr_file\': \'{}\',"
        "\'feat_file\': \'{}\', \'batch_size\': \'{}\', \'quadruplet_list_dir\': \'{}\'}}".format(train_data_dir, 'norm_attrs.txt', 'norm_feats.txt', str(batch_size), quadruplet_dir)),
        ntop=6)

AddAttrEnDecoderLayers(net, attr_size)

#net.embed_loss, net.identical_loss, net.similar_loss, net.dissimilar_loss = L.Python(net.attr_embed, net.iden, net.sim, net.dissim, net.delta_sim, net.delta_dissim, loss_weight=[1,1,1,1], python_param=dict(module="loss",layer="QuadrupletLoss",param_str="{{\'threshold\': {}, \'loss_weight\': {}}}".format(str(threshold), str(loss_weight1))), ntop=4)
net.recon_loss = L.EuclideanLoss(net.attr_pred, net.attr, loss_weight=loss_weight2)
net.embed_loss = L.Python(net.attr_embed, net.iden, net.sim, net.dissim, net.delta_sim, net.delta_dissim,loss_weight=1, python_param=dict(module="loss",layer="QuadrupletLoss",param_str="{{\'threshold\': {}, \'loss_weight\': {}}}".format(str(threshold), str(loss_weight1))), ntop=1)

solver_str = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

with open(solver_file, 'w') as f:
    print(solver_str, file=f)
shutil.copy(solver_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-7:]
  
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['attr'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, attr_size, 1, 1])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('export PYTHONPATH={}:$PYTHONPATH\n'.format(python_module))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  #f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)
os.chmod(job_file, stat.S_IRWXU)
#remove old quadruplet list

for file in os.listdir(quadruplet_dir):
	if file.endswith(".txt"):
		os.remove(quadruplet_dir+'/'+file)

caffe.set_device(device_id)
caffe.set_mode_gpu()

for file in os.listdir(snapshot_dir):
	if file.endswith("caffemodel"):
		os.remove(snapshot_dir+'/'+file)
	if file.endswith("solverstate"):
		os.remove(snapshot_dir+'/'+file)
solver = caffe.AdamSolver(solver_file)

# save the initial model
solver.net.save("{}_iter_0.caffemodel".format(snapshot_prefix))

max_iter = 0
for epoch in range(max_epoch):

  # Find most recent snapshot.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if iter > max_iter:
        max_iter = iter
  
  if max_iter % iter_per_epoch == 0:
  	model_file = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)
  	GenerateNewQuadrupletList(model_file, deploy_net_file, 
  		  train_data_dir+'/norm_feats.txt', train_data_dir+'/norm_attrs.txt', 
  		  '{}/quadruplet_list_{}.txt'.format(quadruplet_dir,epoch), attr_size, threshold, device_id)
  
  solver.step(iter_per_epoch)


  
