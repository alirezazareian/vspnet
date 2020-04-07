import os
import time
from numbers import Number
import traceback
from copy import deepcopy

import tensorflow as tf

from ...util.util import tic, toc, round_floats
from ...eval.eval_utils import evaluate
from .util import deploy 

class Trainer (object):
    def __init__ (self, model, sess,
                  train_batcher,
                  val_batcher,
                  train_evaluator,
                  val_evaluator,
                  logdir,
                  modeldir,
                  debug_val=False,
                  initialize='all',
                  initial_step=0,
                 ):

        self.model = model
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self.train_evaluator = train_evaluator
        self.val_evaluator = val_evaluator
        self.modeldir = modeldir
        self.debug_val = debug_val

        self.train_evaluator.reset()
        self.val_evaluator.reset()
        self.training_step = initial_step
        
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        if initialize is 'all':
            self.sess.run(tf.global_variables_initializer())
        if initialize is 'trainables':
            self.sess.run(tf.variables_initializer(tf.trainable_variables()))

        if logdir is not None:
            self.summary = tf.summary.merge_all()
            self.summary_writer_train_main = tf.summary.FileWriter(os.path.join(logdir, 'main', 'train'))
            self.summary_writer_train_debug = tf.summary.FileWriter(os.path.join(logdir, 'debug', 'train'), self.sess.graph)
            self.summary_writer_val_main = tf.summary.FileWriter(os.path.join(logdir, 'main', 'val'))
            if self.debug_val:
                self.summary_writer_val_debug = tf.summary.FileWriter(os.path.join(logdir, 'debug', 'val'), self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=100)#var_list = tf.trainable_variables())

        
    def train (self,
               max_steps,
               train_eval_rate, validation_rate, verbose_rate, summary_rate, checkpoint_rate,
               hyperparams = {},
               validation_hyperparams = {},
               debug_mode=False,
               ignore_errors=False,
               save_last_checkpoint=False,
               stop_loss_ratio=None,
              ):

        start_time = tic()
        validation_updated = False
        train_eval_updated = False
        
        last_loss = 1000.
        
        results = {}
        
        for step in range(max_steps):
            try:               
                self.training_step += 1
                
                data_batch = self.train_batcher.next_batch()
                feed_dict = {self.model.data_placeholders[key] : data_batch[key] for key in data_batch if key in self.model.data_placeholders}
                hp_feed_dict = {self.model.hyperparam_placeholders[key] : hyperparams[key] for key in hyperparams}
                feed_dict.update(hp_feed_dict)

                # gathering all the operations that should be run in this step
                ops_to_run = {
                    'loss': self.model.loss,
                }

                ops_to_run['train_update'] = self.model.train_op
                        
                if train_eval_rate is not None:
                    ops_to_run['train_eval'] = self.model.eval_ops
                    
                if summary_rate is not None and self.training_step % summary_rate == 0:
                    ops_to_run['summary_values'] = self.summary
                    
                if debug_mode and verbose_rate is not None and self.training_step % verbose_rate == 0:
                    ops_to_run['debug_info'] = self.model.debug_info

                # running one training step
                results = self.sess.run(ops_to_run, feed_dict=feed_dict)
                        
                # accumulating training evaluation results to pass to evaluate() function at once
                if train_eval_rate is not None:
                    feed_dict.update(results['train_eval'])
                    self.train_evaluator.collect_batch(feed_dict)
                
                # do evaluation on training data
                if train_eval_rate is not None and self.training_step % train_eval_rate == 0:
                    # Run evaluation on accumulated results of last training batches.
                    train_eval_results = self.train_evaluator.evaluate()
                    
                    # manually adding training score to tensorboard summary file
                    train_eval_summary_str = tf.Summary(
                        value=[tf.Summary.Value(tag='eval/'+key, simple_value=train_eval_results[key]) 
                               for key in train_eval_results if isinstance(train_eval_results[key], Number)]
                    ) 
                    self.summary_writer_train_main.add_summary(train_eval_summary_str,
                                                    global_step=self.training_step)
                    self.summary_writer_train_main.flush()
                        
                    # tell the log printing module to show evaluation results 
                    train_eval_updated = True
                    # empty the values for the next cycle
                    self.train_evaluator.reset()

                # do validation evaluation
                if validation_rate is not None and self.training_step % validation_rate == 0:
                    validation_results = self.validate(hyperparams=validation_hyperparams, verbose=False, ignore_errors=ignore_errors, debug_mode=debug_mode)
                    
                    # manually adding validation score to tensorboard summary file
                    validation_summary_str = tf.Summary(
                        value=[tf.Summary.Value(tag='eval/'+key, simple_value=validation_results[key]) 
                               for key in validation_results if isinstance(validation_results[key], Number)]
                    ) 
                    self.summary_writer_val_main.add_summary(validation_summary_str,
                                                    global_step=self.training_step)
                    self.summary_writer_val_main.flush()
                        
                    # tell the log printing module to show validation results 
                    validation_updated = True
                    # empty the values for the next cycle

                    
                # print log
                if verbose_rate is not None and self.training_step % verbose_rate == 0:
                    duration = toc(start_time)

                    print('Step {}: loss = {} ({:.3f} sec)'.format(self.training_step,
                                                                   round_floats(results['loss'], 3),
                                                                   duration))
                    
                    if train_eval_updated:
                        print('Training Batch Evaluation Results:', round_floats(train_eval_results, 3))
                        # don't print it again until updated
                        train_eval_updated = False
                    if validation_updated:
                        print('Validation Results:', round_floats(validation_results, 3))
                        # don't print it again until updated
                        validation_updated = False

                    if debug_mode:
                        print('DEBUG INFO: ', results['debug_info'])
                        
                    
                # write model summaries to tensorboard summary file
                if summary_rate is not None and self.training_step % summary_rate == 0:
                    self.summary_writer_train_debug.add_summary(results['summary_values'],
                                                    global_step=self.training_step)
                    self.summary_writer_train_debug.flush()
                    
                    if self.debug_val:
                        data_batch = self.val_batcher.next_batch()
                        feed_dict = {self.model.data_placeholders[key] : data_batch[key] for key in data_batch if key in self.model.data_placeholders}
                        hp_feed_dict = {self.model.hyperparam_placeholders[key] : hyperparams[key] for key in validation_hyperparams}
                        feed_dict.update(hp_feed_dict)
                        sv = self.sess.run(self.summary, feed_dict=feed_dict)

                        self.summary_writer_val_debug.add_summary(sv, global_step=self.training_step)
                        self.summary_writer_val_debug.flush()
                    

                # save model parameters as checkpoint
                if checkpoint_rate is not None and self.training_step % checkpoint_rate == 0:
                    self.save_model()

                if stop_loss_ratio is not None:
                    if last_loss - results['loss'] <= stop_loss_ratio:
                        break
                    last_loss = results['loss']
                    
            except:
                print(traceback.format_exc())
                print('Batch {}/{}'.format (
                    ((self.training_step-1) % self.train_batcher.num_batch)+1,
                    self.train_batcher.num_batch,
                ))
                if not ignore_errors:
                    raise
            
        if save_last_checkpoint:
            self.save_model()

    def save_model(self, name=None):
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        if name is None:
            name = os.path.join(self.modeldir, 'ckpt')

        self.saver.save(self.sess, name, global_step=self.training_step if name is None else None)                

    def save_model(self, name=None):
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
        global_step = None
        if name is None:
            name = 'ckpt'
            global_step = self.training_step
        path = os.path.join(self.modeldir, name)

        self.saver.save(self.sess, path, global_step=global_step)

        
    def load_model(self, name=None):
        if name is None:
            path = tf.train.latest_checkpoint(self.modeldir)
        else:
            path = os.path.join(self.modeldir, name)

        self.saver.restore(self.sess, path)


    def validate(self, hyperparams={}, verbose=True, ignore_errors=False, debug_mode=False):
        self.val_evaluator.reset()
        deploy(self.sess, self.model, self.val_batcher, self.val_evaluator, self.model.eval_ops,
            hyperparams=hyperparams,
            ignore_errors=ignore_errors,
            debug_mode=debug_mode
        )
        validation_results = self.val_evaluator.evaluate()
        self.val_evaluator.reset()
        if verbose:
            print(validation_results)
        return validation_results
        

    def run_debug_ops(self, hyperparams = {}, step=None, verbose=False):        
        data_batch = self.train_batcher.next_batch(keep_cursor=True)
        feed_dict = {self.model.data_placeholders[key] : data_batch[key] for key in data_batch if key in self.model.data_placeholders}

        hp_feed_dict = {self.model.hyperparam_placeholders[key] : hyperparams[key] for key in hyperparams}

        feed_dict.update(hp_feed_dict)

        debug_info = self.sess.run(self.model.debug_ops, feed_dict=feed_dict)

        debug_info['input_data'] = data_batch
        
        if verbose:
            print(debug_info)
        
        return debug_info
    
    
    def get_runtime_stats(self, hyperparams = {}):   
        data_batch = self.train_batcher.next_batch(keep_cursor=True)
        feed_dict = {self.model.data_placeholders[key] : data_batch[key] for key in data_batch if key in self.model.data_placeholders}

        hp_feed_dict = {self.model.hyperparam_placeholders[key] : hyperparams[key] for key in hyperparams}

        feed_dict.update(hp_feed_dict)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        self.sess.run(self.model.train_op, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        
        self.summary_writer_train_debug.add_run_metadata(run_metadata, f'step_{self.training_step}')
        self.summary_writer_train_debug.flush()
    
    