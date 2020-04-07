## from vsp_model_002.py using GT labels for predcls

import numpy as np
from scipy.optimize import linear_sum_assignment

import tensorflow as tf

from ..util.util import nd_batch_proc, relaxed_softmax, act_fn_dict, mask_keep_topk, get_variables_starting_with, get_tensor_shape
from ..util.util import MLPClass as MLP

from .abstract_model import AbstractModel

slim = tf.contrib.slim

class Model(AbstractModel):
    def __init__(self, 
                 emb_dict_ent,
                 emb_dict_pred,
                 
                 num_mp_iter, 
                 num_align_iter,
                 num_proposals,
                 max_num_pred,  
                 
                 dim_state_ent,
                 dim_state_pred,
                 dim_init_head_ent,
                 dim_init_head_pred,
                 dim_emb_head_ent,
                 dim_conf_head_ent,
                 dim_conf_head_pred,
                 dim_emb_head_pred,
                 dim_att_head_role_ent,
                 dim_att_head_role_pred,
                 dim_message_send_head_ent2pred,
                 dim_message_pool_head_ent2pred,
                 dim_message_receive_head_ent2pred,
                 dim_message_send_head_pred2ent,
                 dim_message_pool_head_pred2ent,
                 dim_message_receive_head_pred2ent,
                 
                 dim_proposal_feat,
                 #dim_image_feat,
                 #image_size,
                 #num_channels,
                 #backbone,
                 #trainable_backbone,

                 init_state_type_pred,
                 role_edge_mode,
                 stop_grad_role_edge,

                 alignment_order='best',
                 heat_role=1.0,
                 heat_emb_mil=1.0,
                 num_roles=2,
                 #suppress_duplicates=False,
                 default_act_fn_name='leaky_relu',
                 null_att_logit=0.0,
                 #optimizer_type_backbone=tf.train.GradientDescentOptimizer,
                 optimizer_type_generator=tf.train.AdamOptimizer,
                 debugging=False,
                 clip_loss_terms=False,
                 eps=1e-10,
                ):
        super().__init__(debugging)
        
        #emb_dim = dim_emb_head_ent[-1]
        default_act_fn = act_fn_dict[default_act_fn_name]
        
        self.data_placeholders = {
            'image_id': tf.placeholder(tf.int32, (None,), 'image_id'),
            #'image_features': tf.placeholder(tf.float32, (None, dim_image_feat), 'image_features'),
            'proposal_features': tf.placeholder(tf.float32, (None, num_proposals, dim_proposal_feat), 'proposal_features'),
            #'image': tf.placeholder(tf.float32, (None, image_size, image_size, num_channels), 'image'),
            'proposal_boxes': tf.placeholder(tf.float32, (None, num_proposals, 4), 'proposal_boxes'),
            #'ent_emb': tf.placeholder(tf.float32, (None, None, emb_dim), 'ent_emb'),
            'ent_lbl': tf.placeholder(tf.int32, (None, None), 'ent_lbl'),
            #'pred_emb': tf.placeholder(tf.float32, (None, None, emb_dim), 'pred_emb'),
            'ent_box': tf.placeholder(tf.float32, (None, None, 4), 'ent_box'),
            'pred_lbl': tf.placeholder(tf.int32, (None, None), 'pred_lbl'),
            'num_ent': tf.placeholder(tf.int32, (None,), 'num_ent'),
            'num_pred': tf.placeholder(tf.int32, (None,), 'num_pred'),
            'pred_roles': tf.placeholder(tf.bool, (None, num_roles, None, None), 'pred_roles'),
            #'proposal_prior_lbl': tf.placeholder(tf.float32, (None, num_proposals, emb_dict_ent.shape[0]), 'ent_prior_lbl'),
        }
        
        batch_size = get_tensor_shape(self.data_placeholders['image_id'], 0)
        if num_proposals == None:
            num_proposals = get_tensor_shape(self.data_placeholders['proposal_boxes'], 1)

        with tf.variable_scope('graph_generator'):
            # Defining recurrent cells
            entity_recurrent_cell = tf.nn.rnn_cell.GRUCell(dim_state_ent, activation=default_act_fn)
            predicate_recurrent_cell = tf.nn.rnn_cell.GRUCell(dim_state_pred, activation=default_act_fn)
            with tf.variable_scope('mlp_bank'):
                # Defining MLP heads            
                # To map a GRU state to word embedding:
                emb_head_ent = MLP(
                    num_unit_list=dim_emb_head_ent, 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name='emb_head_ent', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )
                emb_head_ent_cls = tf.get_variable('emb_head_ent_cls', initializer=emb_dict_ent, trainable=True)

                emb_head_pred = MLP(
                    num_unit_list=dim_emb_head_pred, 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name='emb_head_pred', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )
                emb_head_pred_cls = tf.get_variable('emb_head_pred_cls', initializer=emb_dict_pred, trainable=True)

                # To predict confidence of nodes
                conf_head_ent = MLP(
                    num_unit_list=dim_conf_head_ent + [1], 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name='conf_head_ent', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )
                conf_head_pred = MLP(
                    num_unit_list=dim_conf_head_pred + [1], 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name='conf_head_pred', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )
                
                # To send messages
                message_send_head_ent2pred = MLP(
                        num_unit_list=dim_message_send_head_ent2pred, 
                        act_fn_hid=default_act_fn, 
                        act_fn_last=default_act_fn, 
                        name=f'message_send_head_ent2pred', 
                        reuse=tf.AUTO_REUSE, 
                        dropout=self.get_hyperparam('dropout_rate_g'),
                        dropout_last=True,
                    )

                message_send_head_pred2ent = MLP(
                        num_unit_list=dim_message_send_head_pred2ent, 
                        act_fn_hid=default_act_fn, 
                        act_fn_last=default_act_fn, 
                        name=f'message_send_head_pred2ent', 
                        reuse=tf.AUTO_REUSE, 
                        dropout=self.get_hyperparam('dropout_rate_g'),
                        dropout_last=True,
                    )
                
                # To pool messages across nodes
                message_pool_head_ent2pred_list = []
                message_pool_head_pred2ent_list = []
                for r in range(num_roles):
                    message_pool_head_ent2pred_list.append(MLP(
                            num_unit_list=dim_message_pool_head_ent2pred, 
                            act_fn_hid=default_act_fn, 
                            act_fn_last=default_act_fn, 
                            name=f'message_pool_head_ent2pred_{r}', 
                            reuse=tf.AUTO_REUSE, 
                            dropout=self.get_hyperparam('dropout_rate_g'),
                            dropout_last=True,
                        ))

                    message_pool_head_pred2ent_list.append(MLP(
                            num_unit_list=dim_message_pool_head_pred2ent, 
                            act_fn_hid=default_act_fn, 
                            act_fn_last=default_act_fn, 
                            name=f'message_pool_head_pred2ent_{r}', 
                            reuse=tf.AUTO_REUSE, 
                            dropout=self.get_hyperparam('dropout_rate_g'),
                            dropout_last=True,
                        ))
                
                # To receive messages
                message_receive_head_ent2pred = MLP(
                        num_unit_list=dim_message_receive_head_ent2pred, 
                        act_fn_hid=default_act_fn, 
                        act_fn_last=default_act_fn, 
                        name=f'message_receive_head_ent2pred', 
                        reuse=tf.AUTO_REUSE, 
                        dropout=self.get_hyperparam('dropout_rate_g'),
                        dropout_last=True,
                    )

                message_receive_head_pred2ent = MLP(
                        num_unit_list=dim_message_receive_head_pred2ent, 
                        act_fn_hid=default_act_fn, 
                        act_fn_last=default_act_fn, 
                        name=f'message_receive_head_pred2ent', 
                        reuse=tf.AUTO_REUSE, 
                        dropout=self.get_hyperparam('dropout_rate_g'),
                        dropout_last=True,
                    )
                
                # To compare two GRU states and compute attention:
                assert(dim_att_head_role_ent[-1] % num_roles == 0)
                att_head_role_ent = MLP(
                    num_unit_list=dim_att_head_role_ent, 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name=f'att_head_role_ent', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )
                assert(dim_att_head_role_pred[-1] % num_roles == 0)
                att_head_role_pred = MLP(
                    num_unit_list=dim_att_head_role_pred, 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=None, 
                    name=f'att_head_role_pred', 
                    reuse=tf.AUTO_REUSE, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )

                # To initialize node states using image/region features
                init_head_ent = MLP(
                    num_unit_list=dim_init_head_ent + [dim_state_ent], 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=default_act_fn, 
                    name=f'init_head_ent', 
                    reuse=None, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )                
                init_head_ent_coord = MLP(
                    num_unit_list=dim_init_head_ent + [dim_state_ent], 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=default_act_fn, 
                    name=f'init_head_ent_coord', 
                    reuse=None, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )                
                init_head_ent_lbl = MLP(
                    num_unit_list=dim_init_head_ent + [dim_state_ent], 
                    act_fn_hid=default_act_fn, 
                    act_fn_last=default_act_fn, 
                    name=f'init_head_ent_lbl', 
                    reuse=None, 
                    dropout=self.get_hyperparam('dropout_rate_g'),
                    dropout_last=False,
                )                
                if init_state_type_pred == 'mlp':
                    init_head_pred = MLP(
                        num_unit_list=dim_init_head_pred + [max_num_pred * dim_state_pred], 
                        act_fn_hid=default_act_fn, 
                        act_fn_last=default_act_fn, 
                        name=f'init_head_pred', 
                        reuse=None, 
                        dropout=self.get_hyperparam('dropout_rate_g'),
                        dropout_last=False,
                    )                                
                                
                                                
            ent_gru_state = (nd_batch_proc(self.data_placeholders['proposal_features'], init_head_ent) +
                             nd_batch_proc(tf.one_hot(self.data_placeholders['ent_lbl'], emb_dict_ent.shape[0]), init_head_ent_lbl) +
                             nd_batch_proc(self.data_placeholders['proposal_boxes'], init_head_ent_coord))
                
            # Initial state of predicate nodes can have different types
            with tf.variable_scope(f'state_init_pred'):
                if init_state_type_pred == 'zeros':
                    pred_gru_state = tf.zeros((batch_size, max_num_pred, dim_state_pred))
                elif init_state_type_pred == 'random':
                    pred_gru_state = tf.random.uniform((batch_size, max_num_pred, dim_state_pred))
                elif init_state_type_pred == 'trainable':
                    pred_gru_state = tf.get_variable('init_gru_state_pred', 
                        initializer=tf.random.uniform((1, max_num_pred, dim_state_pred)), 
                        trainable=True
                    )
                    pred_gru_state = tf.tile(pred_gru_state, (batch_size, 1, 1))
                elif init_state_type_pred == 'fixed_random':
                    pred_gru_state = tf.constant(np.random.uniform((1, max_num_pred, dim_state_pred)), dtype=np.float32)
                    pred_gru_state = tf.tile(pred_gru_state, (batch_size, 1, 1))
                else:
                    raise NotImplementedError
                
            
                
            # initializing some things
            entity_nodes_conf_logits = tf.ones((batch_size, num_proposals))

            # Main loop: iterations of message passing 
            for step in range(num_mp_iter):
                with tf.variable_scope(f'mp_iter_{step}'): 
                    with tf.variable_scope(f'mp_update_role_edges'):
                        ent_keys = nd_batch_proc(ent_gru_state, att_head_role_ent)
                        pred_keys = nd_batch_proc(pred_gru_state, att_head_role_pred)
                        
                        ent_keys = tf.reshape(ent_keys, (batch_size, num_proposals, num_roles, -1))
                        pred_keys = tf.reshape(pred_keys, (batch_size, max_num_pred, num_roles, -1))

                        ent_keys = tf.transpose(ent_keys, (0, 2, 3, 1))
                        pred_keys = tf.transpose(pred_keys, (0, 2, 1, 3))
                        
                        ent_keys = tf.reshape(ent_keys, (batch_size * num_roles, -1, num_proposals))
                        pred_keys = tf.reshape(pred_keys, (batch_size * num_roles, max_num_pred, -1))                        
                        
                        att_weights_role = tf.matmul(pred_keys, ent_keys)
                        att_weights_role = tf.reshape(att_weights_role, (batch_size, num_roles, max_num_pred, num_proposals))
                        
                        self.add_to_log(att_weights_role, f'att_weights_role_logits_step_{step}')

                        att_weights_role = att_weights_role + (tf.stop_gradient(entity_nodes_conf_logits)[:, tf.newaxis, tf.newaxis, :] * self.get_hyperparam('confidence_weight_on_role'))
                        self.add_to_log(att_weights_role, f'att_weights_role_logits_modulated_by_confidence_step_{step}')
                        
                        cc_shape = get_tensor_shape(att_weights_role)
                        cc_shape[1] = 1
                        att_weights_role = tf.concat((att_weights_role, null_att_logit * tf.ones(cc_shape, dtype=tf.float32)), axis=1)
                        cc_shape = get_tensor_shape(att_weights_role)
                        cc_shape[3] = 1
                        att_weights_role = tf.concat((att_weights_role, null_att_logit * tf.ones(cc_shape, dtype=tf.float32)), axis=3)
                        att_weights_role = att_weights_role / heat_role
                        
                        att_weights_role_normalized_wrt_role = tf.nn.softmax(att_weights_role, axis=1)
                        att_weights_role_normalized_wrt_ent = tf.nn.softmax(att_weights_role, axis=3)

                        log_att_weights_role_normalized_wrt_role = tf.nn.log_softmax(att_weights_role, axis=1)
                        log_att_weights_role_normalized_wrt_ent = tf.nn.log_softmax(att_weights_role, axis=3)
                        
                        att_weights_role_normalized_wrt_role = att_weights_role_normalized_wrt_role[:, :-1, :, :-1]
                        att_weights_role_normalized_wrt_ent = att_weights_role_normalized_wrt_ent[:, :-1, :, :-1]
                        log_att_weights_role_normalized_wrt_role = log_att_weights_role_normalized_wrt_role[:, :-1, :, :-1]
                        log_att_weights_role_normalized_wrt_ent = log_att_weights_role_normalized_wrt_ent[:, :-1, :, :-1]

                        att_weights_role = att_weights_role_normalized_wrt_role * att_weights_role_normalized_wrt_ent                        
                        #att_weights_role = att_weights_role_normalized_wrt_role
                        log_att_weights_role = log_att_weights_role_normalized_wrt_role + log_att_weights_role_normalized_wrt_ent                        
                        #log_att_weights_role = log_att_weights_role_normalized_wrt_role 
                        
                        self.add_to_log(att_weights_role_normalized_wrt_role, f'att_weights_role_normalized_wrt_grid_step_{step}')
                        self.add_to_log(att_weights_role_normalized_wrt_ent, f'att_weights_role_normalized_wrt_ent_step_{step}')
                        self.add_to_log(att_weights_role, f'att_weights_role_step_{step}')
                        self.add_to_log(log_att_weights_role_normalized_wrt_role, f'log_att_weights_role_normalized_wrt_grid_step_{step}')
                        self.add_to_log(log_att_weights_role_normalized_wrt_ent, f'log_att_weights_role_normalized_wrt_ent_step_{step}')
                        self.add_to_log(log_att_weights_role, f'log_att_weights_role_step_{step}')
                                                      
                        if role_edge_mode == 'full_soft':
                            att_weights_mp = att_weights_role
                        elif role_edge_mode == 'one_per_role':
                            att_weights_mp = mask_keep_topk(att_weights_role, 1, True)
                        elif role_edge_mode.startswith('one_per_top_'):
                            att_weights_sum = tf.transpose(tf.reduce_sum(att_weights_role, axis=-1), (0, 2, 1))
                            att_weights_sum = mask_keep_topk(att_weights_sum, int(role_edge_mode[12:]), True)
                            att_weights_sum = tf.transpose(att_weights_sum, (0, 2, 1))[:, :, :, tf.newaxis]
                            att_weights_mp = mask_keep_topk(att_weights_role, 1, True) * att_weights_sum
                        else:
                            raise NotImplementedError
                        if stop_grad_role_edge:
                            att_weights_mp = tf.stop_gradient(att_weights_mp)
                        
                    with tf.variable_scope(f'mp_comp_messages_role'):
                        message_sent_ent2pred = nd_batch_proc(ent_gru_state, message_send_head_ent2pred)
                        message_pooled_ent2pred = tf.reshape(tf.matmul(
                            tf.reshape(tf.transpose(att_weights_mp, (0, 2, 1, 3)), (batch_size, max_num_pred * num_roles, num_proposals)),
                            message_sent_ent2pred
                        ), (batch_size, max_num_pred, num_roles, get_tensor_shape(message_sent_ent2pred, -1)))
                        message_pooled_ent2pred = tf.add_n([nd_batch_proc(message_pooled_ent2pred[:, :, r, :], message_pool_head_ent2pred_list[r]) for r in range(num_roles)])
                        message_received_ent2pred = nd_batch_proc(message_pooled_ent2pred, message_receive_head_ent2pred)

                        message_sent_pred2ent = nd_batch_proc(pred_gru_state, message_send_head_pred2ent)
                        message_pooled_pred2ent = tf.reshape(tf.matmul(
                            tf.reshape(tf.transpose(att_weights_mp, (0, 3, 1, 2)), (batch_size, num_proposals * num_roles, max_num_pred)),
                            message_sent_pred2ent
                        ), (batch_size, num_proposals, num_roles, get_tensor_shape(message_sent_pred2ent, -1)))
                        message_pooled_pred2ent = tf.add_n([nd_batch_proc(message_pooled_pred2ent[:, :, r, :], message_pool_head_pred2ent_list[r]) for r in range(num_roles)])
                        message_received_pred2ent = nd_batch_proc(message_pooled_pred2ent, message_receive_head_pred2ent)
                                            
                    if step > 0:
                        with tf.variable_scope(f'mp_update_ent'):
                            message_received_ent = message_received_pred2ent
                            ent_gru_input = tf.reshape(message_received_ent, [batch_size * num_proposals, get_tensor_shape(message_received_ent, -1)])
                            ent_gru_state = tf.reshape(ent_gru_state, [batch_size * num_proposals, dim_state_ent])
                            _, ent_gru_state_new = entity_recurrent_cell(ent_gru_input, ent_gru_state)
                            self.add_to_log(tf.norm(ent_gru_state_new - ent_gru_state, axis=-1) / (tf.norm(ent_gru_state, axis=-1) + eps), f'ent_gru_state_diff_step_{step}')
                            ent_gru_state = tf.reshape(ent_gru_state_new, [batch_size, num_proposals, dim_state_ent])
                                
                    with tf.variable_scope(f'mp_update_pred'):
                        message_received_pred = message_received_ent2pred
                        pred_gru_input = tf.reshape(message_received_pred, [batch_size * max_num_pred, get_tensor_shape(message_received_pred, -1)])
                        pred_gru_state = tf.reshape(pred_gru_state, [batch_size * max_num_pred, dim_state_pred])
                        _, pred_gru_state_new = predicate_recurrent_cell(pred_gru_input, pred_gru_state)
                        self.add_to_log(tf.norm(pred_gru_state_new - pred_gru_state, axis=-1) / (tf.norm(pred_gru_state, axis=-1) + eps), f'pred_gru_state_diff_step_{step}')
                        pred_gru_state = tf.reshape(pred_gru_state_new, [batch_size, max_num_pred, dim_state_pred])

                    with tf.variable_scope(f'mp_output_heads'):
                        entity_nodes_emb = nd_batch_proc(ent_gru_state, emb_head_ent)
                        entity_nodes_emb = nd_batch_proc(entity_nodes_emb, lambda x: tf.matmul(x, emb_head_ent_cls, transpose_b=True))
                        entity_nodes_dist = tf.nn.softmax(tf.stop_gradient(entity_nodes_emb))

                        predicate_nodes_emb = nd_batch_proc(pred_gru_state, emb_head_pred)            
                        predicate_nodes_emb = nd_batch_proc(predicate_nodes_emb, lambda x: tf.matmul(x, emb_head_pred_cls, transpose_b=True))
                        predicate_nodes_dist = tf.nn.softmax(tf.stop_gradient(predicate_nodes_emb))
                        
                        entity_nodes_conf_logits = tf.squeeze(nd_batch_proc(ent_gru_state, conf_head_ent), axis=-1)
                        entity_nodes_conf = tf.nn.sigmoid(entity_nodes_conf_logits)
                        self.add_to_log(entity_nodes_conf, f'entity_nodes_conf_step_{step}')
                        
                        predicate_nodes_conf_logits = tf.squeeze(nd_batch_proc(pred_gru_state, conf_head_pred), axis=-1)
                        predicate_nodes_conf = tf.nn.sigmoid(predicate_nodes_conf_logits)
                        self.add_to_log(predicate_nodes_conf, f'predicate_nodes_conf_step_{step}')
                        
                        all_pred_classes = tf.tile(tf.reduce_sum(tf.stop_gradient(predicate_nodes_emb), axis=1, keepdims=True), [1, num_proposals, 1])
                        
                            
                        
                        
                        
                        
        with tf.variable_scope('objective'):
            with tf.variable_scope('triplet_mil'):
                self.get_hyperparam('loss_factor_g_mil_ent')
                self.get_hyperparam('loss_factor_g_mil_pred')
                self.get_hyperparam('loss_factor_g_mil_role')
                self.get_hyperparam('loss_factor_g_mil_box')
                self.get_hyperparam('mil_iou_offset')
                                
                def cost_fn(inputs):
                    (   output_ent, target_ent, 
                        output_pred, target_pred,
                        log_output_roles, target_roles, 
                        output_box, target_box,
                        output_ent_conf, output_pred_conf,
                        num_target_ent, num_target_pred
                    ) = inputs
                                        
                    target_ent = tf.slice(target_ent, [0], [num_target_ent])
                    target_pred = tf.slice(target_pred, [0], [num_target_pred])
                    target_box = tf.slice(target_box, [0, 0], [num_target_ent, 4])
                    
                    target_roles = tf.slice(target_roles, [0, 0, 0], [-1, num_target_pred, num_target_ent])
                    target_roles = tf.cast(target_roles, tf.float32)                                                                    
                        
                    ## BEGIN Computing pairwise costs
                    cost_mtx_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.tile(output_ent[:, tf.newaxis, :], [1, num_target_ent, 1]),
                        labels=tf.tile(target_ent[tf.newaxis, :], [num_proposals, 1])
                    )
                    #cost_mtx_ent = tf.Print(cost_mtx_ent, [tf.reduce_mean(cost_mtx_ent)], message=f'cost_mtx_ent')
                    cost_mtx_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.tile(output_pred[:, tf.newaxis, :], [1, num_target_pred, 1]),
                        labels=tf.tile(target_pred[tf.newaxis, :], [max_num_pred, 1])
                    )
                    #cost_mtx_pred = tf.Print(cost_mtx_pred, [tf.reduce_mean(cost_mtx_pred)], message=f'cost_mtx_pred')

                    pw_iou = iou(
                        tf.tile(output_box[:, tf.newaxis, :], [1, num_target_ent, 1]),
                        tf.tile(target_box[tf.newaxis, :, :], [num_proposals, 1, 1])
                    )
                    cost_mtx_box = - tf.log(pw_iou + self.get_hyperparam('mil_iou_offset'))
                    
                    #cost_mtx_ent_nrmls = cost_mtx_ent - offset_ent
                    #cost_mtx_ent_nrmls = cost_mtx_ent_nrmls / (tf.reduce_max(cost_mtx_ent_nrmls) + eps)
                    #cost_mtx_pred_nrmls = cost_mtx_pred - offset_pred
                    #cost_mtx_pred_nrmls = cost_mtx_pred_nrmls / (tf.reduce_max(cost_mtx_pred_nrmls) + eps)                    
                    
                    #output_roles = tf.Print(output_roles, [tf.reduce_mean(output_roles)], message=f'output_roles')
                    
                    cost_mtx_roles = - tf.reduce_sum(log_output_roles[:, :, :, tf.newaxis, tf.newaxis] * target_roles[:, tf.newaxis, tf.newaxis, :, :], axis=0)
                    ## END Computing pairwise costs
                    
                    #cost_mtx_roles = tf.Print(cost_mtx_roles, [tf.reduce_mean(cost_mtx_roles)], message=f'cost_mtx_roles')

                    # Some images may have no entity or no predicate, in which case loss becomes NaN. Here I replace those losses with zero.
                    zero_ents = tf.equal(num_target_ent, 0)
                    zero_preds = tf.equal(num_target_pred, 0)

                    ## BEGIN Finding an approximation for predicate alignment to start with
                    offset_ent = tf.reduce_min(cost_mtx_ent)
                    offset_pred = tf.reduce_min(cost_mtx_pred)
                    
                    pw_sim_ent_emb_u = tf.exp((offset_ent - cost_mtx_ent) / heat_emb_mil)
                    pw_sim_pred_emb_u = tf.exp((offset_pred - cost_mtx_pred) / heat_emb_mil)
                    pw_sim_ent_emb = pw_sim_ent_emb_u / (tf.reduce_sum(pw_sim_ent_emb_u, axis=0, keepdims=True) + eps)
                    pw_sim_pred_emb = pw_sim_pred_emb_u / (tf.reduce_sum(pw_sim_pred_emb_u, axis=0, keepdims=True) + eps)
                    '''                
                    stats = []
                    for item in [(offset_ent - cost_mtx_ent), (offset_pred - cost_mtx_pred), pw_sim_ent_emb_u, pw_sim_pred_emb_u, pw_sim_ent_emb, pw_sim_pred_emb, cost_mtx_roles, output_roles, neg_log_output_roles, neg_log_inv_output_roles]:
                        stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                    cost_mtx_roles = tf.Print(cost_mtx_roles, stats, message=f'stats')
                    '''
                    
                    
                    #pred_align_mtx = pw_sim_pred_emb 
                    pred_align_mtx = tf.zeros_like(pw_sim_pred_emb)
                    ## END Finding an approximation for predicate alignment to start with
                    
                    ent_align_mtx_per_iter_entfirst = []
                    pred_align_mtx_per_iter_entfirst = []
                    ent_loss_per_iter_entfirst = []
                    ent_conf_loss_per_iter_entfirst = []
                    pred_conf_loss_per_iter_entfirst = []
                    pred_loss_per_iter_entfirst = []
                    role_loss_per_iter_entfirst = []
                    ent_align_mtx_diff_per_iter_entfirst = []
                    pred_align_mtx_diff_per_iter_entfirst = []
                    for it in range(num_align_iter):
                        ## BEGIN Finding optimal entity alignment given current predicate alignment
                        cost_mtx_ent_roles = tf.reduce_sum(pred_align_mtx[:, tf.newaxis, :, tf.newaxis] * cost_mtx_roles, axis=[0, 2])   
                        cost_mtx_ent_roles = cost_mtx_ent_roles / (float(num_roles) * tf.reduce_sum(pred_align_mtx) + eps)
                        pw_ent_loss = ((self.get_hyperparam('loss_factor_g_mil_ent') * cost_mtx_ent) +
                                       (self.get_hyperparam('loss_factor_g_mil_role') * cost_mtx_ent_roles) +
                                       (self.get_hyperparam('loss_factor_g_mil_box') * cost_mtx_box)) 
                        '''
                        stats = []
                        for item in [cost_mtx_ent, cost_mtx_ent_roles, pw_ent_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        pw_ent_loss = tf.Print(pw_ent_loss, stats, message=f'alignment_entfirst_ent_it_{it}')
                        '''
                        output_ind_ent, target_ind_ent = tf.py_func(linear_sum_assignment, [pw_ent_loss], (tf.int64, tf.int64))   
                        ent_align_mtx = tf.scatter_nd(tf.stack((output_ind_ent, target_ind_ent), axis=-1), 
                                                      tf.ones(tf.shape(output_ind_ent), dtype=tf.float32), 
                                                      (num_proposals, num_target_ent))
                        ent_align_mtx_per_iter_entfirst.append(ent_align_mtx)
                        ## END Finding optimal entity alignment given current predicate alignment
                        
                        ## BEGIN Finding optimal predicate alignment given current entity alignment
                        cost_mtx_pred_roles = tf.reduce_sum(ent_align_mtx[tf.newaxis, :, tf.newaxis, :] * cost_mtx_roles, axis=[1, 3])                    
                        cost_mtx_pred_roles = cost_mtx_pred_roles / (float(num_roles) * tf.reduce_sum(ent_align_mtx) + eps)
                        pw_pred_loss = ((self.get_hyperparam('loss_factor_g_mil_pred') * cost_mtx_pred) +
                                        (self.get_hyperparam('loss_factor_g_mil_role') * cost_mtx_pred_roles))
                        '''
                        stats = []
                        for item in [cost_mtx_pred, cost_mtx_pred_roles, pw_pred_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        pw_pred_loss = tf.Print(pw_pred_loss, stats, message=f'alignment_entfirst_pred_it_{it}')
                        '''
                        output_ind_pred, target_ind_pred = tf.py_func(linear_sum_assignment, [pw_pred_loss], (tf.int64, tf.int64))               
                        pred_align_mtx = tf.scatter_nd(tf.stack((output_ind_pred, target_ind_pred), axis=-1), 
                                                       tf.ones(tf.shape(output_ind_pred), dtype=tf.float32), 
                                                       (max_num_pred, num_target_pred))
                        pred_align_mtx_per_iter_entfirst.append(pred_align_mtx)
                        ## END Finding optimal predicate alignment given current entity alignment
                        
                        stats = []
                        for item in [cost_mtx_ent, cost_mtx_ent_roles, pw_ent_loss,
                                     cost_mtx_pred, cost_mtx_pred_roles, pw_pred_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        stats.append(tf.reduce_sum(ent_align_mtx))
                        stats.append(tf.reduce_sum(pred_align_mtx))
                        if it > 0:
                            diff = tf.reduce_sum(tf.abs(ent_align_mtx_per_iter_entfirst[-1] - ent_align_mtx_per_iter_entfirst[-2]))
                            ent_align_mtx_diff_per_iter_entfirst.append(diff)
                            stats.append(diff)
                            diff = tf.reduce_sum(tf.abs(pred_align_mtx_per_iter_entfirst[-1] - pred_align_mtx_per_iter_entfirst[-2]))
                            pred_align_mtx_diff_per_iter_entfirst.append(diff)
                            stats.append(diff)
                        #pred_align_mtx = tf.Print(pred_align_mtx, stats, message=f'alignment_entfirst_it_{it}')
                        
                        ## BEGIN Computing loss based on final alignment
                        ent_loss_1 = tf.gather_nd(cost_mtx_ent, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                        ent_loss_1 = tf.reduce_mean(ent_loss_1)
                        pred_loss_1 = tf.gather_nd(cost_mtx_pred, tf.stack((output_ind_pred, target_ind_pred), axis=1))
                        pred_loss_1 = tf.reduce_mean(pred_loss_1)                    
                        cost_mtx_ent_roles_1 = tf.reduce_sum(pred_align_mtx[:, tf.newaxis, :, tf.newaxis] * cost_mtx_roles, axis=[0, 2])                    
                        role_loss_1 = tf.reduce_sum(ent_align_mtx * cost_mtx_ent_roles_1)
                        role_loss_1 = role_loss_1 / (float(num_roles) * tf.reduce_sum(ent_align_mtx) * tf.reduce_sum(pred_align_mtx) + eps)
                                                
                        ent_conf_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_ent_conf, labels=tf.reduce_sum(ent_align_mtx, axis=1)))
                        pred_conf_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_pred_conf, labels=tf.reduce_sum(pred_align_mtx, axis=1)))
                        ## END Computing final loss based tfon alignment
                        
                        ent_loss_1 = tf.where(zero_ents, tf.cast(0.0, tf.float32), ent_loss_1)
                        pred_loss_1 = tf.where(zero_preds, tf.cast(0.0, tf.float32), pred_loss_1)
                        role_loss_1 = tf.where(tf.logical_or(zero_ents, zero_preds), tf.cast(0.0, tf.float32), role_loss_1)
                        
                        #ent_loss_1 = tf.Print(ent_loss_1, [ent_loss_1, pred_loss_1, role_loss_1], message=f'final_loss_entfirst_it_{it}')
                        
                        ent_loss_per_iter_entfirst.append(ent_loss_1)
                        ent_conf_loss_per_iter_entfirst.append(ent_conf_loss_1)
                        pred_conf_loss_per_iter_entfirst.append(pred_conf_loss_1)
                        pred_loss_per_iter_entfirst.append(pred_loss_1)
                        role_loss_per_iter_entfirst.append(role_loss_1)
                                        
                    output_top_conf_1 = mask_keep_topk(output_ent_conf, tf.cast(tf.reduce_sum(ent_align_mtx), tf.int32), True)
                    conf_accuracy_1 = tf.div_no_nan(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(output_top_conf_1, 1.0), tf.equal(tf.reduce_sum(ent_align_mtx, axis=1), 1.0)), tf.float32)), tf.reduce_sum(ent_align_mtx))
                    pred_output_top_conf_1 = mask_keep_topk(output_pred_conf, tf.cast(tf.reduce_sum(pred_align_mtx), tf.int32), True)
                    pred_conf_accuracy_1 = tf.div_no_nan(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred_output_top_conf_1, 1.0), tf.equal(tf.reduce_sum(pred_align_mtx, axis=1), 1.0)), tf.float32)), tf.reduce_sum(pred_align_mtx))
                    box_loss_1 = tf.gather_nd(cost_mtx_box, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                    box_loss_1 = tf.reduce_mean(box_loss_1)
                    iou_1 = tf.gather_nd(pw_iou, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                    iou_1 = tf.reduce_mean(iou_1)
                            
                    # Repeating the same process but this time starting from predicate instead
                    #ent_align_mtx = pw_sim_ent_emb                    
                    ent_align_mtx = tf.zeros_like(pw_sim_ent_emb)
                    
                    ent_align_mtx_per_iter_predfirst = []
                    pred_align_mtx_per_iter_predfirst = []
                    ent_loss_per_iter_predfirst = []
                    ent_conf_loss_per_iter_predfirst = []
                    pred_conf_loss_per_iter_predfirst = []
                    pred_loss_per_iter_predfirst = []
                    role_loss_per_iter_predfirst = []
                    ent_align_mtx_diff_per_iter_predfirst = []
                    pred_align_mtx_diff_per_iter_predfirst = []
                    for it in range(num_align_iter):
                        ## BEGIN Finding optimal predicate alignment given current entity alignment
                        cost_mtx_pred_roles = tf.reduce_sum(ent_align_mtx[tf.newaxis, :, tf.newaxis, :] * cost_mtx_roles, axis=[1, 3])                    
                        cost_mtx_pred_roles = cost_mtx_pred_roles / (float(num_roles) * tf.reduce_sum(ent_align_mtx) + eps)
                        pw_pred_loss = ((self.get_hyperparam('loss_factor_g_mil_pred') * cost_mtx_pred) +
                                        (self.get_hyperparam('loss_factor_g_mil_role') * cost_mtx_pred_roles))
                        '''
                        stats = []
                        for item in [cost_mtx_pred, cost_mtx_pred_roles, pw_pred_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        pw_pred_loss = tf.Print(pw_pred_loss, stats, message=f'alignment_predfirst_pred_it_{it}')
                        '''
                        output_ind_pred, target_ind_pred = tf.py_func(linear_sum_assignment, [pw_pred_loss], (tf.int64, tf.int64))               
                        pred_align_mtx = tf.scatter_nd(tf.stack((output_ind_pred, target_ind_pred), axis=-1), 
                                                       tf.ones(tf.shape(output_ind_pred), dtype=tf.float32), 
                                                       (max_num_pred, num_target_pred))
                        pred_align_mtx_per_iter_predfirst.append(pred_align_mtx)
                        ## END Finding optimal predicate alignment given current entity alignment

                        ## BEGIN Finding optimal entity alignment given current predicate alignment
                        cost_mtx_ent_roles = tf.reduce_sum(pred_align_mtx[:, tf.newaxis, :, tf.newaxis] * cost_mtx_roles, axis=[0, 2])                    
                        cost_mtx_ent_roles = cost_mtx_ent_roles / (float(num_roles) * tf.reduce_sum(pred_align_mtx) + eps)
                        pw_ent_loss = ((self.get_hyperparam('loss_factor_g_mil_ent') * cost_mtx_ent) +
                                       (self.get_hyperparam('loss_factor_g_mil_role') * cost_mtx_ent_roles) +
                                       (self.get_hyperparam('loss_factor_g_mil_box') * cost_mtx_box)) 
                        '''
                        stats = []
                        for item in [cost_mtx_ent, cost_mtx_ent_roles, pw_ent_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        pw_ent_loss = tf.Print(pw_ent_loss, stats, message=f'alignment_predfirst_ent_it_{it}')
                        '''
                        output_ind_ent, target_ind_ent = tf.py_func(linear_sum_assignment, [pw_ent_loss], (tf.int64, tf.int64))   
                        ent_align_mtx = tf.scatter_nd(tf.stack((output_ind_ent, target_ind_ent), axis=-1), 
                                                      tf.ones(tf.shape(output_ind_ent), dtype=tf.float32), 
                                                      (num_proposals, num_target_ent))
                        ent_align_mtx_per_iter_predfirst.append(ent_align_mtx)
                        ## END Finding optimal entity alignment given current predicate alignment
                        
                        stats = []
                        for item in [cost_mtx_ent, cost_mtx_ent_roles, pw_ent_loss,
                                     cost_mtx_pred, cost_mtx_pred_roles, pw_pred_loss,]:
                            stats += [tf.reduce_min(item), tf.reduce_max(item), tf.reduce_mean(item)]
                        stats.append(tf.reduce_sum(ent_align_mtx))
                        stats.append(tf.reduce_sum(pred_align_mtx))
                        if it > 0:
                            diff = tf.reduce_sum(tf.abs(ent_align_mtx_per_iter_predfirst[-1] - ent_align_mtx_per_iter_predfirst[-2]))
                            ent_align_mtx_diff_per_iter_predfirst.append(diff)
                            stats.append(diff)
                            diff = tf.reduce_sum(tf.abs(pred_align_mtx_per_iter_predfirst[-1] - pred_align_mtx_per_iter_predfirst[-2]))
                            pred_align_mtx_diff_per_iter_predfirst.append(diff)
                            stats.append(diff)
                        #pred_align_mtx = tf.Print(pred_align_mtx, stats, message=f'alignment_predfirst_it_{it}')
                        
                        ## BEGIN Computing loss based on final alignment
                        ent_loss_2 = tf.gather_nd(cost_mtx_ent, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                        ent_loss_2 = tf.reduce_mean(ent_loss_2)
                        pred_loss_2 = tf.gather_nd(cost_mtx_pred, tf.stack((output_ind_pred, target_ind_pred), axis=1))
                        pred_loss_2 = tf.reduce_mean(pred_loss_2)                    
                        cost_mtx_ent_roles_2 = tf.reduce_sum(pred_align_mtx[:, tf.newaxis, :, tf.newaxis] * cost_mtx_roles, axis=[0, 2])                    
                        role_loss_2 = tf.reduce_sum(ent_align_mtx * cost_mtx_ent_roles_2)
                        role_loss_2 = role_loss_2 / (float(num_roles) * tf.reduce_sum(ent_align_mtx) * tf.reduce_sum(pred_align_mtx) + eps)

                        ent_conf_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_ent_conf, labels=tf.reduce_sum(ent_align_mtx, axis=1)))
                        pred_conf_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_pred_conf, labels=tf.reduce_sum(pred_align_mtx, axis=1)))
                        ## END Computing final loss based on alignment
                        
                        ent_loss_2 = tf.where(zero_ents, tf.cast(0.0, tf.float32), ent_loss_2)
                        pred_loss_2 = tf.where(zero_preds, tf.cast(0.0, tf.float32), pred_loss_2)
                        role_loss_2 = tf.where(tf.logical_or(zero_ents, zero_preds), tf.cast(0.0, tf.float32), role_loss_2)

                        #ent_loss_2 = tf.Print(ent_loss_2, [ent_loss_2, pred_loss_2, role_loss_2], message=f'final_loss_predfirst_it_{it}')
                        #ent_loss_2 = tf.Print(ent_loss_2, [tf.reduce_sum(ent_align_mtx), tf.reduce_sum(pred_align_mtx)], message=f'num_aligned_predfirst_it_{it}')
                        
                        ent_loss_per_iter_predfirst.append(ent_loss_2)
                        ent_conf_loss_per_iter_predfirst.append(ent_conf_loss_2)
                        pred_conf_loss_per_iter_predfirst.append(pred_conf_loss_2)
                        pred_loss_per_iter_predfirst.append(pred_loss_2)
                        role_loss_per_iter_predfirst.append(role_loss_2)
                    
                    output_top_conf_2 = mask_keep_topk(output_ent_conf, tf.cast(tf.reduce_sum(ent_align_mtx), tf.int32), True)
                    conf_accuracy_2 = tf.div_no_nan(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(output_top_conf_2, 1.0), tf.equal(tf.reduce_sum(ent_align_mtx, axis=1), 1.0)), tf.float32)), tf.reduce_sum(ent_align_mtx))
                    pred_output_top_conf_2 = mask_keep_topk(output_pred_conf, tf.cast(tf.reduce_sum(pred_align_mtx), tf.int32), True)
                    pred_conf_accuracy_2 = tf.div_no_nan(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred_output_top_conf_2, 1.0), tf.equal(tf.reduce_sum(pred_align_mtx, axis=1), 1.0)), tf.float32)), tf.reduce_sum(pred_align_mtx))
                    box_loss_2 = tf.gather_nd(cost_mtx_box, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                    box_loss_2 = tf.reduce_mean(box_loss_2)
                    iou_2 = tf.gather_nd(pw_iou, tf.stack((output_ind_ent, target_ind_ent), axis=1))
                    iou_2 = tf.reduce_mean(iou_2)
                    
                    ent_loss_per_iter_entfirst = tf.stack(ent_loss_per_iter_entfirst)
                    ent_conf_loss_per_iter_entfirst = tf.stack(ent_conf_loss_per_iter_entfirst)
                    pred_conf_loss_per_iter_entfirst = tf.stack(pred_conf_loss_per_iter_entfirst)
                    pred_loss_per_iter_entfirst = tf.stack(pred_loss_per_iter_entfirst)
                    role_loss_per_iter_entfirst = tf.stack(role_loss_per_iter_entfirst)

                    ent_loss_per_iter_predfirst = tf.stack(ent_loss_per_iter_predfirst)
                    ent_conf_loss_per_iter_predfirst = tf.stack(ent_conf_loss_per_iter_predfirst)
                    pred_conf_loss_per_iter_predfirst = tf.stack(pred_conf_loss_per_iter_predfirst)
                    pred_loss_per_iter_predfirst = tf.stack(pred_loss_per_iter_predfirst)
                    role_loss_per_iter_predfirst = tf.stack(role_loss_per_iter_predfirst)

                    ent_align_mtx_diff_per_iter_entfirst = tf.stack(ent_align_mtx_diff_per_iter_entfirst)
                    pred_align_mtx_diff_per_iter_entfirst = tf.stack(pred_align_mtx_diff_per_iter_entfirst)
                    ent_align_mtx_diff_per_iter_predfirst = tf.stack(ent_align_mtx_diff_per_iter_predfirst)
                    pred_align_mtx_diff_per_iter_predfirst = tf.stack(pred_align_mtx_diff_per_iter_predfirst)
                    
                    return (ent_loss_per_iter_entfirst, pred_loss_per_iter_entfirst, role_loss_per_iter_entfirst, ent_conf_loss_per_iter_entfirst, pred_conf_loss_per_iter_entfirst,
                            ent_loss_per_iter_predfirst, pred_loss_per_iter_predfirst, role_loss_per_iter_predfirst, ent_conf_loss_per_iter_predfirst, pred_conf_loss_per_iter_predfirst,
                            ent_align_mtx_diff_per_iter_entfirst, pred_align_mtx_diff_per_iter_entfirst, ent_align_mtx_diff_per_iter_predfirst, pred_align_mtx_diff_per_iter_predfirst,
                            conf_accuracy_1, conf_accuracy_2, pred_conf_accuracy_1, pred_conf_accuracy_2,
                            box_loss_1, box_loss_2,
                            iou_1, iou_2,
                           ) 
                (
                    ent_loss_per_iter_entfirst, pred_loss_per_iter_entfirst, role_loss_per_iter_entfirst, ent_conf_loss_per_iter_entfirst, pred_conf_loss_per_iter_entfirst,
                    ent_loss_per_iter_predfirst, pred_loss_per_iter_predfirst, role_loss_per_iter_predfirst, ent_conf_loss_per_iter_predfirst, pred_conf_loss_per_iter_predfirst,
                    ent_align_mtx_diff_per_iter_entfirst, pred_align_mtx_diff_per_iter_entfirst, ent_align_mtx_diff_per_iter_predfirst, pred_align_mtx_diff_per_iter_predfirst,
                    conf_accuracy_entfirst, conf_accuracy_predfirst, pred_conf_accuracy_entfirst, pred_conf_accuracy_predfirst,
                    box_loss_entfirst, box_loss_predfirst,
                    iou_entfirst, iou_predfirst,
                ) = tf.map_fn(cost_fn, [
                    entity_nodes_emb, self.data_placeholders['ent_lbl'],
                    predicate_nodes_emb, self.data_placeholders['pred_lbl'], 
                    log_att_weights_role, self.data_placeholders['pred_roles'],                      
                    self.data_placeholders['proposal_boxes'], self.data_placeholders['ent_box'],                      
                    entity_nodes_conf_logits,
                    predicate_nodes_conf_logits,
                    self.data_placeholders['num_ent'],
                    self.data_placeholders['num_pred'],
                ], dtype=(tf.float32,) * 22)

                ent_loss_per_iter_entfirst = tf.reduce_mean(ent_loss_per_iter_entfirst, axis=0)
                ent_conf_loss_per_iter_entfirst = tf.reduce_mean(ent_conf_loss_per_iter_entfirst, axis=0)
                pred_conf_loss_per_iter_entfirst = tf.reduce_mean(pred_conf_loss_per_iter_entfirst, axis=0)
                pred_loss_per_iter_entfirst = tf.reduce_mean(pred_loss_per_iter_entfirst, axis=0)
                role_loss_per_iter_entfirst = tf.reduce_mean(role_loss_per_iter_entfirst, axis=0)
                        
                ent_loss_per_iter_predfirst = tf.reduce_mean(ent_loss_per_iter_predfirst, axis=0)
                ent_conf_loss_per_iter_predfirst = tf.reduce_mean(ent_conf_loss_per_iter_predfirst, axis=0)
                pred_conf_loss_per_iter_predfirst = tf.reduce_mean(pred_conf_loss_per_iter_predfirst, axis=0)
                pred_loss_per_iter_predfirst = tf.reduce_mean(pred_loss_per_iter_predfirst, axis=0)
                role_loss_per_iter_predfirst = tf.reduce_mean(role_loss_per_iter_predfirst, axis=0)                                                
                
                loss_generator_entfirst = {
                    'loss_factor_g_mil_ent': ent_loss_per_iter_entfirst,
                    'loss_factor_g_mil_ent_conf': ent_conf_loss_per_iter_entfirst,
                    'loss_factor_g_mil_pred_conf': pred_conf_loss_per_iter_entfirst,
                    'loss_factor_g_mil_pred': pred_loss_per_iter_entfirst,
                    'loss_factor_g_mil_role': role_loss_per_iter_entfirst,
                }

                loss_generator_predfirst = {
                    'loss_factor_g_mil_ent': ent_loss_per_iter_predfirst,
                    'loss_factor_g_mil_ent_conf': ent_conf_loss_per_iter_predfirst,
                    'loss_factor_g_mil_pred_conf': pred_conf_loss_per_iter_predfirst,
                    'loss_factor_g_mil_pred': pred_loss_per_iter_predfirst,
                    'loss_factor_g_mil_role': role_loss_per_iter_predfirst,
                }
                effective_loss_entfirst = tf.add_n([self.get_hyperparam(key) * val for key, val in loss_generator_entfirst.items()])
                effective_loss_predfirst = tf.add_n([self.get_hyperparam(key) * val for key, val in loss_generator_predfirst.items()])

                if alignment_order == 'entfirst':
                    effective_loss = effective_loss_entfirst[-1]
                elif alignment_order == 'predfirst':
                    effective_loss = effective_loss_predfirst[-1]
                elif alignment_order == 'best':
                    effective_loss = tf.minimum(effective_loss_entfirst[-1], effective_loss_predfirst[-1])
                else:
                    raise NotImplementedError

                self.add_to_log(conf_accuracy_entfirst, f'conf_accuracy_entfirst')
                self.add_to_log(conf_accuracy_predfirst, f'conf_accuracy_predfirst')
                self.add_to_log(pred_conf_accuracy_entfirst, f'pred_conf_accuracy_entfirst')
                self.add_to_log(pred_conf_accuracy_predfirst, f'pred_conf_accuracy_predfirst')
                self.add_to_log(box_loss_entfirst, f'box_loss_entfirst')
                self.add_to_log(box_loss_predfirst, f'box_loss_predfirst')
                self.add_to_log(iou_entfirst, f'iou_entfirst')
                self.add_to_log(iou_predfirst, f'iou_predfirst')
                    
            with tf.variable_scope('aux_emb'):
                loss_aux_emb_ent = tf.reduce_mean(tf.reduce_sum(tf.square(emb_dict_ent - emb_head_ent_cls), axis=-1))
                loss_aux_emb_pred = tf.reduce_mean(tf.reduce_sum(tf.square(emb_dict_pred - emb_head_pred_cls), axis=-1))
                
            
            self.loss = effective_loss + (self.get_hyperparam('loss_factor_g_aux_emb_ent') * loss_aux_emb_ent) + (self.get_hyperparam('loss_factor_g_aux_emb_pred') * loss_aux_emb_pred)
            
            tf.summary.scalar(f'effective_loss', effective_loss)
            for k, v in loss_generator_entfirst.items():
                tf.summary.scalar(f'{k}_entfirst', v[-1])
            for k, v in loss_generator_predfirst.items():
                tf.summary.scalar(f'{k}_predfirst', v[-1])
            tf.summary.scalar('loss_factor_g_aux_emb_ent', loss_aux_emb_ent)
            tf.summary.scalar('loss_factor_g_aux_emb_pred', loss_aux_emb_pred)
            
            self.eval_ops = {
                'image_id': self.data_placeholders['image_id'],
                'out_ent_prob': entity_nodes_dist,
                'out_pred_prob': predicate_nodes_dist,
                'out_pred_roles': att_weights_role,
                'out_pred_conf': predicate_nodes_conf,
                'out_num_ent': tf.fill((batch_size,), num_proposals),
                'out_num_pred': tf.fill((batch_size,), max_num_pred),
                #'ent_box': self.data_placeholders['proposal_boxes'],
            }

            self.debug_ops.update(self.eval_ops)

            
        def build_multiterm_grads(loss_factors, var_list):
            grads = {}
            for key in loss_factors:
                grads[key] = tf.gradients(loss_factors[key], var_list)
                grads[key] = [tf.clip_by_norm(grad, self.get_hyperparam('grad_clipping')) if grad != None else None for grad in grads[key]]
            grads_combined = []
            for i in range(len(var_list)):
                grad = 0.0
                for key in grads:
                    if grads[key][i] != None:
                        grad = grad + (grads[key][i] * self.get_hyperparam(key))
                grads_combined.append(grad)
            return grads_combined, grads
                        
        with tf.variable_scope('training'):
            grad_relative_norm = lambda grad_list, var_list: tf.reduce_mean([tf.norm(grad) / tf.norm(var) for grad, var in zip(grad_list, var_list) if grad is not None])            
            grad_avg_val = lambda grad_list: tf.reduce_mean([tf.reduce_mean(tf.abs(item)) for item in grad_list if item is not None])            
            
            var_list_generator = get_variables_starting_with('graph_generator', trainables_only=True)
            var_list = list(var_list_generator)
            
            for var in tf.trainable_variables():
                if var not in var_list:
                    print(f'WARNING: variable {var.name} is trainable but will not be trained')            
            
            if clip_loss_terms:
                print('WARNING: computing gradients for each loss term is computationally expensive')
                grads, grads_mt = build_multiterm_grads(self.loss, var_list)
                for key in grads_mt:
                    self.debug_summary_ops['grads_' + key] = grad_avg_val(grads_mt[key])
            else:
                grads = tf.gradients(self.loss, var_list)
                grads = [tf.clip_by_norm(grad, self.get_hyperparam('grad_clipping')) if grad != None else None for grad in grads]
                self.debug_summary_ops['grad_avg_val'] = grad_avg_val(grads)
                self.debug_summary_ops['grad_relative_norm'] = grad_relative_norm(grads, var_list)
                     
            grads_generator = grads
                    
            optimizer_generator = optimizer_type_generator(self.get_hyperparam('learning_rate_generator'))
                    
            additional_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if debugging:
                print_op = tf.print(self.debug_ops)
                additional_update_ops.append(print_op)
                
            with tf.control_dependencies(additional_update_ops):
                train_op_generator = optimizer_generator.apply_gradients(zip(grads_generator, var_list_generator))
                self.train_ops = [train_op_generator]
                
                self.train_op = tf.group(self.train_ops)
                
        self.embeddings = {
        }
                
        with tf.variable_scope('other_stats'):
            #tf.summary.histogram(f'att_weights_role', att_weights_role)
            #tf.summary.histogram(f'sum_att_weights_role', tf.reduce_sum(att_weights_role, axis=-1))
            [tf.summary.scalar(f'effective_loss_entfirst_step_{i}', effective_loss_entfirst[i]) for i in range(num_align_iter)]
            [tf.summary.scalar(f'effective_loss_predfirst_step_{i}', effective_loss_predfirst[i]) for i in range(num_align_iter)]
            #[tf.summary.histogram(f'ent_align_mtx_diff_per_iter_entfirst_step_{i}', ent_align_mtx_diff_per_iter_entfirst[:, i]) for i in range(num_align_iter - 1)]
            #[tf.summary.histogram(f'pred_align_mtx_diff_per_iter_entfirst_step_{i}', pred_align_mtx_diff_per_iter_entfirst[:, i]) for i in range(num_align_iter - 1)]
            #[tf.summary.histogram(f'ent_align_mtx_diff_per_iter_predfirst_step_{i}', ent_align_mtx_diff_per_iter_predfirst[:, i]) for i in range(num_align_iter - 1)]
            #[tf.summary.histogram(f'pred_align_mtx_diff_per_iter_predfirst_step_{i}', pred_align_mtx_diff_per_iter_predfirst[:, i]) for i in range(num_align_iter - 1)]
            [tf.summary.scalar(key, val) for key, val in self.debug_summary_ops.items()]
            [tf.summary.histogram(key, val) for key, val in self.debug_summary_hist_ops.items()]

            

def iou(box1, box2):
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    intsec = (tf.maximum(0.0, tf.minimum(box1[..., 2], box2[..., 2]) - tf.maximum(box1[..., 0], box2[..., 0])) * 
              tf.maximum(0.0, tf.minimum(box1[..., 3], box2[..., 3]) - tf.maximum(box1[..., 1], box2[..., 1])))
    iou = tf.div_no_nan(intsec, (area1 + area2 - intsec))
    return iou
            