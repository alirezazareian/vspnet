import os
import time
import sys
import traceback

import numpy as np
import tensorflow as tf

from ..util.util import get_variables_starting_with
from ...util.util import tic, toc

def load_params(sess, filename, replace_string_dict={}):
    var_dict = {}
    for var in tf.global_variables():
        name = var.name.split(':')[0]
        flag = False
        for key, val in replace_string_dict.items():
            if key in name:
                if val is None:
                    flag = True
                else:
                    name = name.replace(key, val)
        if not flag:
            var_dict[name] = var
    saver = tf.train.Saver(var_dict, reshape=False)
    saver.restore(sess, filename)
    

def save_params(sess, filename=None, var_dict=None, global_step=None):
    if var_dict is not None:
        saver = tf.train.Saver(var_dict)
    else:
        saver = tf.train.Saver()
    saver.save(sess, filename, global_step=global_step)


def deploy(sess, model, batcher, collector, op,
           hyperparams = {},
           verbose_rate=None,
           ignore_errors=False,
           debug_mode=False,
          ):

    if verbose_rate is not None:
        print('Deploying')
    batcher.reset()
    start_time = tic()
    rt_hist = []
    pt_hist = []
    wt_hist = []
    for step in range(batcher.num_batch):
        read_flag = False
        write_flag = False
        try:
            # loading data batch into GPU
            batcher_cursor = batcher.cursor
            s = tic()
            data_batch = batcher.next_batch()
            rt = toc(s)
            rt_hist.append(rt)

            feed_dict = {model.data_placeholders[key] : data_batch[key] for key in data_batch if key in model.data_placeholders}
            # setting hyperparameters
            hp_feed_dict = {model.hyperparam_placeholders[key] : hyperparams[key] for key in hyperparams}
            feed_dict.update(hp_feed_dict)

            # running the computational graph
            s = tic()
            res = sess.run(
                (op, model.debug_info) if debug_mode else op,
                feed_dict=feed_dict
            )
            pt = toc(s)
            pt_hist.append(pt)

            if debug_mode:
                values, debug_info = res
            else:
                values = res
            
        except:
            print(traceback.format_exc())
            print('Batch %d/%d' % (step+1, batcher.num_batch))
            if not ignore_errors:
                raise
            collector.skip_batch()
            
        try:
            # saving batch results
            s = tic()
            collector.collect_batch(values)
            wt = toc(s)
            wt_hist.append(wt)
            write_flag = True
        except:
            print('Batch %d/%d' % (step+1, batcher.num_batch))
            raise

        # printing status once every verbose_rate steps
        if verbose_rate is not None and (step % verbose_rate == 0 or step == batcher.num_batch - 1):        
            print('Batch %d/%d done in %d seconds. Spent time for read=%.2f (avg %.2f), processing=%.2f (avg %.2f), write=%.2f (avg %.2f) seconds.' 
                  % (step+1, batcher.num_batch, toc(start_time), rt, np.mean(rt_hist), pt, np.mean(pt_hist), wt, np.mean(wt_hist)))
             
            if debug_mode:
                print('DEBUG INFO: ', debug_info)
                    

                
    if verbose_rate is not None:
        print('Done in %d seconds' % toc(start_time))


    

def convert(batcher, collector, verbose_rate=None, ignore_errors=False):

    if verbose_rate is not None:
        print('Converting')
    start_time = tic()
    
    for step in range(batcher.num_batch):
        try:
            # load a batch
            s = tic()
            data_batch = batcher.next_batch()
            rt = toc(s)

            # write the batch
            s = tic()
            collector.collect_batch(data_batch)
            wt = toc(s)

            # print logs
            if verbose_rate is not None and step % verbose_rate == 0:        
                print('Batch %d/%d done in %d seconds. Spent time for read=%.2f, write=%.2f seconds.' 
                      % (step+1, batcher.num_batch, toc(start_time), rt, wt))

        except:
            print(traceback.format_exc())
            print('Batch %d/%d' % (step+1, batcher.num_batch))
            if not ignore_errors:
                raise
            
    if verbose_rate is not None:
        print('All done in %d secs' % toc(start_time))








