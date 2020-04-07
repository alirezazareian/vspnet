## from vsp_model_037.py: bigger images

import numpy as np
import tensorflow as tf

from ..util.util import get_variables_starting_with, get_tensor_shape

class AbstractModel:
    def __init__(self, debugging):
        self.debug_ops = {}
        self.debug_summary_ops = {}
        self.debug_summary_hist_ops = {}
        self.data_placeholders = {}
        self.hyperparam_placeholders = {}
        self.debugging = debugging
        

    def add_to_log(self, tensor, name, summary=True, summary_hist=False):
        get_stats = lambda tensor: {
            'min': tf.reduce_min(tensor), 
            'mean': tf.reduce_mean(tensor), 
            'std': tf.math.reduce_std(tensor), 
            'max': tf.reduce_max(tensor), 
            'norm': tf.norm(tensor),
        }
        
        self.debug_ops[name] = tensor
        if summary_hist:
            self.debug_summary_hist_ops[name] = tensor
        stats = get_stats(tensor)
        if summary:
            for key, val in stats.items():
                self.debug_summary_ops[name + '__' + key] = val
        if self.debugging:
            pr_dict = {}
            for key, val in stats.items():
                pr_dict[name + ':' + key] = val                
            pr_dict[name + ':shape'] = tf.shape(val)
            with tf.control_dependencies([tf.print(pr_dict)]):
                temp = tf.identity(tensor)
                tensor.__init__(
                    temp.op,
                    temp.value_index,
                    temp.dtype
                )
        
        
    def get_hyperparam(self, key, shape=(), dtype=tf.float32):
        if key not in self.hyperparam_placeholders:
            self.hyperparam_placeholders[key] = tf.placeholder(dtype, shape=shape, name=key)
        return self.hyperparam_placeholders[key]
        
