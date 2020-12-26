from layers.worker_layers import *
from tensorflow.keras import layers as tfkl
from param import *
import numpy as np

class State:
  layers = {}
 
  # misc
  layers['KL'] = tf.Variable(0,trainable = False,dtype = 'float32')
  layers['zeros'] = tf.Variable(np.zeros((BS*BUCKETS,2,2,1)),trainable = False,dtype = 'float32')

  # backbone
  layers['z0_expand_to_z1_expand'],layers['z1_expand_to_z2_expand'],layers['z2_expand_to_z3_expand'],layers['z3_expand_to_z4_expand'],layers['z4_expand_to_z5_expand'] = get_backbone()

  # z crush layers
  layers['z1_expand_to_z1'] = FilterCrush(32)
  layers['z2_expand_to_z2'] = FilterCrush(64)
  layers['z3_expand_to_z3'] = FilterCrush(128)
  layers['z4_expand_to_z4'] = FilterCrush(256)
  layers['z5_expand_to_z5'] = FilterCrush(512)

  # z expand layers
  layers['z1_to_z1_expand'] = FilterExpand(32)
  layers['z2_to_z2_expand'] = FilterExpand(64)
  layers['z3_to_z3_expand'] = FilterExpand(128)
  layers['z4_to_z4_expand'] = FilterExpand(256)
  layers['z5_to_z5_expand'] = FilterExpand(512)

  # z_n_expand to z_n-1 infer
  layers['z2_expand_to_z1_expand'] = Infer(32)
  layers['z3_expand_to_z2_expand'] = Infer(64)
  layers['z4_expand_to_z3_expand'] = Infer(128)
  layers['z5_expand_to_z4_expand'] = Infer(256)
  
  # decoder
  layers['decoder'] = Decoder()
  
  # visual
  layers['alpha_visual'] = Visual()
  layers['beta_visual'] = Visual()
 
  # x distribution layers
  layers['visual_to_x_dist_param_alpha'] = tfkl.Conv2D(1,7,padding = 'same')
  layers['visual_to_x_dist_param_beta'] = tfkl.Conv2D(1,7,padding = 'same')

