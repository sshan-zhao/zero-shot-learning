import sys
sys.path.append('/home/zss/zsl/caffe-ssd/python')
import caffe

import numpy as np
from PIL import Image
import scipy.io as sco
import scipy.misc as smc
import random

class QuadrupletLoss(caffe.Layer):
	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.threshold = float(params['threshold'])
		self.loss_weight = float(params['loss_weight'])
		self.eps = 1e-6

		if len(bottom) != 6:
			raise Exception("Need six inputs: attr, identical class x_i, similar class x_j, dissimilar class x_k,"
				"delta_j, delta_k!")

	def reshape(self, bottom, top):

		if bottom[0].count != bottom[1].count or bottom[1].count != bottom[2].count or bottom[2].count != bottom[3].count:
			raise Exception("Inputs must have the same dimension.")

		self.y_dot_xi = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.y_dot_xj = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.y_dot_xk = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.normy_prod_normxi = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.normy_prod_normxj = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.normy_prod_normxk = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.cos_yxi = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.cos_yxj = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.cos_yxk = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		self.normy = np.zeros([bottom[0].data.shape[0], 1], dtype=np.float32)
		top[0].reshape(1)
#		top[1].reshape(1)
#		top[2].reshape(1)
#		top[3].reshape(1)

	def forward(self, bottom, top):

		identical_loss = 0.0
		similar_loss = 0.0
		dissimilar_loss = 0.0
		for n in range(0, bottom[0].data.shape[0]):
			self.cos_yxi[n,:], self.y_dot_xi[n,:], self.normy_prod_normxi[n,:], self.normy[n,:] = _cossim(bottom[0].data[n,:], bottom[1].data[n,:])
			self.cos_yxj[n,:], self.y_dot_xj[n,:], self.normy_prod_normxj[n,:], _  = _cossim(bottom[0].data[n,:], bottom[2].data[n,:])
			self.cos_yxk[n,:], self.y_dot_xk[n,:], self.normy_prod_normxk[n,:], _  = _cossim(bottom[0].data[n,:], bottom[3].data[n,:])

			identical_loss += (-self.cos_yxi[n,0])
#			print self.cos_yxi[n]
#			print bottom[1].data[n,:]
#			print bottom[2].data[n,:]
#			print bottom[3].data[n,:]
#			print bottom[4].data[n,0]
#			print bottom[5].data[n,0]
#			print n
#			print '-----------------------------------'
			similar_loss += (max(0.0, self.threshold-self.cos_yxj[n,0]) + max(0.0,self.cos_yxj[n,0]-bottom[4].data[n,0]))
			dissimilar_loss += (self.threshold-bottom[5].data[n,0])*self.cos_yxk[n,0]
		print '{:<20}: {}'.format('identical_loss', str(identical_loss/bottom[0].data.shape[0]))
		print '{:<20}: {}'.format('similar_loss', str(similar_loss*self.loss_weight/bottom[0].data.shape[0]))
		print '{:<20}: {}'.format('dissimilar_loss', str(dissimilar_loss/bottom[0].data.shape[0]))
		#print '{:<20}: {}'.format('overall_loss', str((identical_loss+dissimilar_loss+similar_loss)/bottom[0].data.shape[0]))
		top[0].data[...] =  (identical_loss+similar_loss*self.loss_weight+dissimilar_loss)/bottom[0].data.shape[0]	
#		top[1].data[...] =  identical_loss/bottom[0].data.shape[0]
#		top[2].data[...] =  similar_loss/bottom[0].data.shape[0]*self.loss_weight
#		top[3].data[...] =  dissimilar_loss/bottom[0].data.shape[0]
#		print self.cos_yxi
#		print self.cos_yxj
#		print self.cos_yxk
#		print '----------------------------------------------'
	def backward(self, top, propagate_down, bottom):
#		print 'fsafjasldfjl;sjdfl'
		for n in range(bottom[0].data.shape[0]):
			for m in range(bottom[0].data.shape[1]):
				diff_xi = self.cos_yxi[n,0]*bottom[0].data[n,m]/(self.normy[n,0]+self.eps)-bottom[1].data[n,m]/(self.normy_prod_normxi[n,0]+self.eps)
				diff_xk = (self.threshold-bottom[5].data[n,0])*(bottom[3].data[n,m]/(self.normy_prod_normxk[n,0]+self.eps)-self.cos_yxk[n,0]*bottom[0].data[n,m]/(self.normy[n,0]+self.eps))
				diff_xj = 0.0
				if self.threshold-self.cos_yxj[n,0] > 0.0:
					diff_xj += (self.cos_yxj[n,0]*bottom[0].data[n,m]/(self.normy[n,0]+self.eps)-bottom[2].data[n,m]/(self.normy_prod_normxj[n,0]+self.eps))
				if self.cos_yxj[n,0]-bottom[4].data[n,0] > 0.0:
					diff_xj += (bottom[2].data[n,m]/(self.normy_prod_normxj[n,0]+self.eps)-self.cos_yxj[n,0]*bottom[0].data[n,m]/(self.normy[n,0]+self.eps))

				bottom[0].diff[n,m] = diff_xi+diff_xj*self.loss_weight+diff_xk
#			print bottom[0].diff[n,:]
#		print '------------------------------------------------------'
		

def _cossim(x, y):
	dot_result = 0.0
	cossim_result = 0.0
	norm_prod_result = 0.0
	normy_result = 0.0
	normx_result = 0.0
	for a,b in zip(x, y):
		dot_result += a*b
		normy_result += b**2
		normx_result += a**2

	norm_prod_result = (normx_result*normy_result)**0.5
	cossim_result = dot_result / (norm_prod_result+1e-6)

	return cossim_result, dot_result, norm_prod_result, normx_result
