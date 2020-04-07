# from graph_util_10.py: new interface

import numpy as np
import cv2
from .eval_utils import heatmap_iou_batch, HEATMAP_RESIZE_SHAPE, pw_iou
import time


def evaluation_preproc(model_output, args={}):  
    graph_loader = args['graph_loader']
    heat = args.get('heat', 100.)
    edge_thresh = args.get('edge_thresh', None)
    heat_thresh = args.get('heat_thresh', None)
    heat_thresh_is_relative = args.get('heat_thresh_is_relative', True)
    verbose_rate = args.get('verbose_rate', None)
    verbose = args.get('verbose', False)
    topk = args.get('top_k', 1)
    match_zero = args.get('match_zero', True)
    add_full_proposal = args.get('add_full_proposal', False)
    num_proposals = args.get('num_proposals', None)
    gt_entities = args.get('gt_entities', False)
    
    freq_prior_mtx = args.get('freq_prior', None)
    freq_prior_weight = args.get('freq_prior_weight', 0.0)
    filter_nonoverlap = args.get('filter_nonoverlap', False)
    self_confidence = args.get('self_confidence', False)
    rank_triplet_confidence = args.get('rank_triplet_confidence', True)

    assert(not match_zero)
    noun_prior = args.get('noun_prior', [1.0])
    verb_prior = args.get('verb_prior', [1.0])
    
    # reading output graph
    pred_conf = None
    image_id = model_output['image_id']    
    ent_prob = [item[:n] for item, n in zip(model_output['out_ent_prob'], model_output['out_num_ent'])]
    pred_prob = [item[:n] for item, n in zip(model_output['out_pred_prob'], model_output['out_num_pred'])]
    if self_confidence:
        pred_conf = [item[:n] for item, n in zip(model_output['out_pred_conf'], model_output['out_num_pred'])]
    pred_roles = [item[:, :np, :ne] for item, np, ne in zip(model_output['out_pred_roles'], model_output['out_num_pred'], model_output['out_num_ent'])]

    # Sparsifying the graph
    pred_roles_idx = []
    for i in range(len(pred_roles)):
        #pred_roles[i] = pred_roles[i] >= pred_roles[i].max(axis=-1, keepdims=True) if edge_thresh is None else edge_thresh
        if edge_thresh is None:
            max_idx = np.argmax(pred_roles[i], axis=-1)
            pred_roles[i] = np.zeros(pred_roles[i].shape, dtype='bool')
            pr_idx = np.zeros((pred_roles[i].shape[1], pred_roles[i].shape[0]), dtype=np.int32)
            for role in range(max_idx.shape[0]):
                for j in range(max_idx.shape[1]):
                    pred_roles[i][role, j, max_idx[role, j]] = True
                    pr_idx[j, role] = max_idx[role, j]
            pred_roles_idx.append(pr_idx)
        else:
            pred_roles[i] = pred_roles[i] >= edge_thresh

            
    ent_box = []
    for i in range(len(image_id)):
        prop = np.copy(args['proposals'][image_id[i]])
        ent_box.append(prop)
    num_prop = max([prop.shape[0] for prop in ent_box])
    box_array = np.zeros((len(image_id), num_prop, 4))
    for i, box in enumerate(ent_box):
        box_array[i, :box.shape[0]] = box
    ent_box = list(box_array)
            
    if verbose:
        print('Classifying embeddings...')
    
    # Fetching the ground truth graph
    idx = np.asarray([graph_loader.imgid2idx[i] for i in image_id])
    gt_graph = graph_loader.get_gt_batch(idx, pack=False)        
        
        
    # Classifying nodes
    ent_lbl = []
    pred_lbl = []
    ent_score = []
    pred_score = []
    for i in range(len(image_id)):
        if gt_entities:
            nouns_label = np.copy(gt_graph['ent_lbl'][i]).reshape((gt_graph['ent_lbl'][i].shape[0], 1))
            nouns_score = np.ones((gt_graph['ent_lbl'][i].shape[0],))
        else:
            if topk > 1:
                nouns_label = np.argsort(-ent_prob[i], axis=-1)[:, :topk]
            else:
                nouns_label = np.argmin(-ent_prob[i], axis=-1)[:, np.newaxis]
            nouns_score = np.asarray([[ent_prob[i][j, nouns_label[j, k]] for k in range(topk)] for j in range(nouns_label.shape[0])])            
            nouns_score = np.sum(nouns_score, axis=-1)
            
        # Late-fusion with frequency prior        
        if freq_prior_weight > 0.0:
            n = nouns_label.shape[0]
            pred_dist = np.zeros((n * n, pred_prob[i].shape[1]))
            pred_dist[pred_roles_idx[i][:, 0] * n + pred_roles_idx[i][:, 1]] = (1 - freq_prior_weight) * pred_prob[i]
            temp = np.tile(np.arange(n)[:, np.newaxis], (1, n))
            pred_roles_idx[i] = np.stack((temp, temp.T), axis=-1).reshape((n * n, 2))
            pred_dist += freq_prior_weight * freq_prior_mtx[nouns_label[pred_roles_idx[i][:, 0], 0], nouns_label[pred_roles_idx[i][:, 1], 0]]
            pred_roles[i] = np.eye(n, dtype=np.bool)[pred_roles_idx[i].T, :]
        else:
            pred_dist = pred_prob[i]
            
        if topk > 1:
            preds_label = np.argsort(-pred_dist, axis=-1)[:, :topk]
        else:
            preds_label = np.argmin(-pred_dist, axis=-1)[:, np.newaxis]
        if self_confidence:
            nouns_score = np.ones((nouns_label.shape[0],))            
            preds_score = np.copy(pred_conf[i])
        else:
            preds_score = np.asarray([[pred_dist[j, preds_label[j, k]] for k in range(topk)] for j in range(preds_label.shape[0])])
            preds_score = np.sum(preds_score, axis=-1)
        
        if filter_nonoverlap:
            overlap = (pw_iou(ent_box[i], ent_box[i]) > 0.0).astype(np.float32)
            preds_score *= overlap[pred_roles_idx[i][:, 0], pred_roles_idx[i][:, 1]]
        
        ent_lbl.append(nouns_label)
        pred_lbl.append(preds_label)
        ent_score.append(nouns_score)
        pred_score.append(preds_score)
    
    # sorting ...
    for i in range(len(image_id)):
        ts = np.copy(pred_score[i])
        if rank_triplet_confidence:
            for r in range(pred_roles[i].shape[0]):
                ts *= np.matmul(pred_roles[i][r], ent_score[i])
        sort_idx_pred = np.argsort(-ts)
        pred_lbl[i] = pred_lbl[i][sort_idx_pred]
        pred_score[i] = pred_score[i][sort_idx_pred]
        pred_roles[i] = pred_roles[i][:, sort_idx_pred, :]
    
    # Here is the final detected graph
    det_graph = {
        'ent_lbl': ent_lbl,
        'ent_score': ent_score,
        'ent_box': ent_box,
        'pred_lbl': pred_lbl,
        'pred_score': pred_score,
        'pred_roles': pred_roles,
    }
    
    return det_graph, gt_graph
    
    
def evaluation_fetch_gt(model_output, args={}):  
    graph_loader = args['graph_loader']
    
    image_id = []
    for batch in model_output:
        image_id += list(batch['image_id'])
    
    # Fetching the ground truth graph
    idx = np.asarray([graph_loader.imgid2idx[i] for i in image_id])
    gt_graph = graph_loader.get_gt_batch(idx, pack=False)        
    
    return gt_graph
    
    
