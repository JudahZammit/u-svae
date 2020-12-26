from layers.gaussian_dist import * 
from layers.x_dist import *
from layers.state import State
from param import *
l = State.layers
import tensorflow.keras as tfk
import numpy as np

def get_qhat_znt_expand_direct(zn_1_expand,n):
  
  zn_expand = l['z{}_expand_to_z{}_expand'.format(n-1,n)](zn_1_expand)

  return zn_expand


def get_qhat_zt_expand_direct(x):
    
  qhat_zt_expand_direct = {}
   
  zn_1t_expand = x
  for n in range(1,6):
      zn_1t_expand = get_qhat_znt_expand_direct(zn_1t_expand,n)
      qhat_zt_expand_direct['qhat_zt_expand_direct{}'.format(n)] = zn_1t_expand

  return qhat_zt_expand_direct

def get_all_qhat_z(all_expands):
  
  qhat_z = []

  for i in range(len(all_expands)):
    qhat_z_layer = l['z{}_expand_to_z{}'.format(5-i,5-i)](all_expands['qhat_zt_expand_direct{}'.format(5-i)])
    qhat_z.append(qhat_z_layer)

  return qhat_z

def get_qhat_z(x):
  
  all_expands = get_qhat_zt_expand_direct(x)
 
  qhat_z = get_all_qhat_z(all_expands) 
    
  return qhat_z


def combine_params(q_mean,q_logvar,p_mean,p_logvar):
  
  q_var = K.exp(q_logvar)
  q_var_inv = 1/q_var

  p_var = K.exp(p_logvar)
  p_var_inv = 1/p_var

  var = 1/(p_var_inv + q_var_inv)
  logvar = K.log(var)

  mean_numerator = q_mean*q_var_inv + p_mean*p_var_inv
  mean_denominator = (p_var_inv + q_var_inv)
        
  mean = mean_numerator/mean_denominator

  return mean,logvar

def get_level_info(qhat_zn_1,zn_expanded,zn_sample,level,gen):
    
  if level == 5:
    p_zn_1 = get_unit_gaussian_dist()
  else:
    zn_1_expanded = l['z{}_expand_to_z{}_expand'.format(level+1,level)](zn_expanded)
    p_zn_1 = l['z{}_expand_to_z{}'.format(level,level)](zn_1_expanded)   

  
  if gen:
    q_zn_1 = p_zn_1
  else:
    q_zn_1 = combine_params(qhat_zn_1[0],qhat_zn_1[1],
                            p_zn_1[0],p_zn_1[1])
    
  zn_1_sample = gaussian_sample(q_zn_1[0],q_zn_1[1])
  zn_1_expanded = l['z{}_to_z{}_expand'.format(level,level)](zn_1_sample)    
   

  return p_zn_1,q_zn_1,zn_1_expanded,zn_1_sample


def z_information(qhat_z,gen = False):
  
  p_z = []
  q_z = []
  z_expanded = []
  z_sample = []

  level_5 = get_level_info(qhat_z[0],None,None,5-0,gen)
  p_z.append(level_5[0])
  q_z.append(level_5[1])
  z_expanded.append(level_5[2])
  z_sample.append(level_5[3])
  for i in range(1,5):
    level = 5 - i
    level_n_1 = get_level_info(qhat_z[i],z_expanded[-1],z_sample[-1],level,gen)
    p_z.append(level_n_1[0])
    q_z.append(level_n_1[1])
    z_expanded.append(level_n_1[2])
    z_sample.append(level_n_1[3])

  out={}
  out['p_z'] = p_z
  out['q_z'] = q_z
  out['z_expanded'] = z_expanded
  out['z_sample'] = z_sample

  return out


def get_decoded_z(z_expanded):
    return l['decoder'](z_expanded)

def get_visuals(decoded_z):
    
    alpha_visual = l['alpha_visual'](decoded_z)
    beta_visual = l['beta_visual'](decoded_z)
    
    return alpha_visual,beta_visual

def create_output_dict(z_sample,x_reconstructed):
  
  out = {}
  out['z5_sample'] = z_sample[0]
  out['z4_sample'] = z_sample[1]
  out['z3_sample'] = z_sample[2]
  out['z2_sample'] = z_sample[3]
  out['z1_sample'] = z_sample[4]
  out['x_reconstructed'] = x_reconstructed

  return out

def create_loss_dict(xent,z_sample,p_z,q_z):
  
  # get losses
  loss_dict = {}

  # x recon loss
  loss_dict['XENT'] = xent  

  # p_z loss 
  for i in range(5):
    loss_dict['p_z{}'.format(5-i)] = -gaussian_ll(z_sample[i],p_z[i][0],p_z[i][1]) 
  
  # q_z loss 
  for i in range(5):
    loss_dict['q_z{}'.format(5-i)] = gaussian_ll(z_sample[i],q_z[i][0],q_z[i][1])
  
  loss = 0
  for x in loss_dict.values():
    loss += x
  
  loss_dict['loss'] = loss
  loss_dict['KL'] = loss-loss_dict['XENT'] 

  return loss_dict

def predict(inputs,gen):
  t0,t1,t2,t3 = inputs 
  x = tf.concat([t0,t1,t2,t3],axis = 0) 
 
  qhat_z = get_qhat_z(x)
  
  z_info = z_information(qhat_z,gen = gen)
   
  all_out = {}
  all_loss = {'KL':0,'loss':0,'XENT':0}

  z_expand = z_info['z_expanded']   
 
  all_decoded = get_decoded_z(z_expand)
  all_alpha_visual,all_beta_visual = get_visuals(all_decoded)
  
  p_x = visual_to_x_dist(all_alpha_visual,all_beta_visual)
  x_reconstructed = dist_to_x(p_x)
  xent = -x_ll(x,p_x)
 
  out = create_output_dict(z_info['z_sample'],x_reconstructed)
  all_out.update(out)
   
  loss_dict = create_loss_dict(xent,z_info['z_sample'],
                                z_info['p_z'],z_info['q_z'])
  for key in all_loss:
      all_loss[key] += loss_dict[key] 

  return all_out,all_loss

 
class myModel(tfk.Model):
    def __init__(self):
        super(myModel,self).__init__()
        
        self.l = State.layers

    def call(self,inputs,gen = False):
        out,loss_dict = predict(inputs,gen = gen)
        out['loss'] = loss_dict['loss'] 
        self.add_loss(loss_dict['loss'])
        self.add_metric(loss_dict['XENT'],name = 'XENT',aggregation = 'mean')          
        self.add_metric(loss_dict['KL']/l['KL'],name = 'Actual KL',aggregation = 'mean')       
        self.add_metric(loss_dict['KL'],name = 'Scaled KL',aggregation = 'mean')       

        return out


