# from graph_eval_2.py: using confidence, adding topk for triplet eval
import numpy as np
import cv2
from .eval_utils import iou, is_inside, attention_correctness, HEATMAP_RESIZE_SHAPE


def eval_ent_prec_rec_simple(det_graph, gt_graph, args={}):                
    verbose_rate = args.get('verbose_rate', None)
    verbose = args.get('verbose', False)
    localization_method = args.get('localization_method', None) # options are None, 'iou', 'peak', 'attention_correctness'
    iou_thresh = args.get('iou_thresh', None)
    ac_thresh = args.get('ac_thresh', None)
    image_size = args.get('image_size', None)
                    
    per_image_ent_precision = []
    per_image_ent_recall = []
    
    for i in range(len(gt_graph['image_id'])):        
        pos = 0
        gt_matched = set()
        det_matched = set()        
        for k in range(det_graph['ent_lbl'][i].shape[0]):
            labels = []
            for l in range(det_graph['ent_lbl'][i].shape[1]):
                if det_graph['ent_lbl'][i][k, l] == 0:
                    break
                labels.append(det_graph['ent_lbl'][i][k, l])
            if len(labels) == 0:
                continue
            pos += 1
            for j in range(gt_graph['ent_lbl'][i].size):
                if gt_graph['ent_lbl'][i][j] not in labels:
                    continue
                if localization_method == 'peak':
                    if not is_inside(det_graph['ent_peak'][i][k], gt_graph['ent_box'][i][j]):
                        continue
                elif localization_method == 'iou':
                    if iou(gt_graph['ent_box'][i][j], det_graph['ent_box'][i][k]) < iou_thresh:
                        continue
                elif localization_method == 'attention_correctness':
                    imgsize = image_size[gt_graph['image_id'][i]] if image_size is not None else HEATMAP_RESIZE_SHAPE
                    hm = cv2.resize(det_graph['ent_heatmap'][i][k], imgsize, interpolation=cv2.INTER_CUBIC)
                    if attention_correctness(hm, gt_graph['ent_box'][i][j]) < ac_thresh:
                        continue
                elif localization_method == None:
                    pass
                else:
                    raise NotImplementedError
                gt_matched.add(j)
                det_matched.add(k)
                                        
        per_image_ent_precision.append(np.divide(len(det_matched), pos))
        per_image_ent_recall.append(np.divide(len(gt_matched), len(gt_graph['ent_lbl'][i])))            
        
        if verbose and i % verbose_rate == 0:
            print('{} images processed out of {}'.format(i+1, len(gt_graph["image_id"])))
    
    ret_dict = {
        'PerImageEntMP': np.nanmean(per_image_ent_precision),
        'PerImageEntMR': np.nanmean(per_image_ent_recall),
    }
    if args.get('return_per_image'):
        ret_dict['PerImageEntPrecision'] = per_image_ent_precision
        ret_dict['PerImageEntRecall'] = per_image_ent_recall

    return ret_dict

def eval_pred_prec_rec_simple(det_graph, gt_graph, args={}):                
    verbose_rate = args.get('verbose_rate', None)
    verbose = args.get('verbose', False)
    
    per_image_pred_precision = []
    per_image_pred_recall = []
    
    for i in range(len(gt_graph['image_id'])):        
        pos = 0
        gt_matched = set()
        det_matched = set()                
        for k in range(det_graph['pred_lbl'][i].shape[0]):
            labels = []
            for l in range(det_graph['pred_lbl'][i].shape[1]):
                if det_graph['pred_lbl'][i][k, l] == 0:
                    break
                labels.append(det_graph['pred_lbl'][i][k, l])
            if len(labels) == 0:
                continue
            pos += 1
            for j in range(gt_graph['pred_lbl'][i].size):
                if gt_graph['pred_lbl'][i][j] not in labels:
                    continue
                gt_matched.add(j)
                det_matched.add(k)

        per_image_pred_precision.append(np.divide(len(det_matched), pos))
        per_image_pred_recall.append(np.divide(len(gt_matched), len(gt_graph['pred_lbl'][i])))
        
        if verbose and i % verbose_rate == 0:
            print('{} images processed out of {}'.format(i+1, len(gt_graph["image_id"])))
                
    ret_dict = {
        'PerImagePredMP': np.nanmean(per_image_pred_precision),
        'PerImagePredMR': np.nanmean(per_image_pred_recall),
    }
    if args.get('return_per_image'):
        ret_dict['PerImagePredPrecision'] = per_image_pred_precision
        ret_dict['PerImagePredRecall'] = per_image_pred_recall

    return ret_dict
    
    
def eval_triple_prec_rec_simple(det_graph, gt_graph, args={}):                
    verbose_rate = args.get('verbose_rate', None)
    verbose = args.get('verbose', False)
    localization_method = args.get('localization_method', None) # options are None, 'iou', 'peak', 'attention_correctness'
    iou_thresh = args.get('iou_thresh', None)
    ac_thresh = args.get('ac_thresh', None)
    image_size = args.get('image_size', None)
    topk = args.get('topk', None)
    
    per_image_precision = []
    per_image_recall = []
    
    for i in range(len(gt_graph['image_id'])):        
        #print(gt_graph['image_id'][i])
        pos = 0
        gt_matched = set()
        det_matched = set()                
        num_pred = det_graph['pred_lbl'][i].shape[0]
        if topk != None:
            num_pred = min(num_pred, topk)
        for k in range(num_pred):
            labels = []
            for l in range(det_graph['pred_lbl'][i].shape[1]):
                if det_graph['pred_lbl'][i][k, l] == 0:
                    break
                labels.append(det_graph['pred_lbl'][i][k, l])
            if len(labels) == 0:
                continue
            pos += 1
            for j in range(gt_graph['pred_lbl'][i].size):
                if gt_graph['pred_lbl'][i][j] not in labels:
                    continue
                if localization_method == 'phrase_iou':
                    det_arg_boxes = []
                    gt_arg_boxes = []
                flag = False   
                for role in range(gt_graph['pred_roles'][i].shape[0]):
                    gt_role_idx = np.where(gt_graph['pred_roles'][i][role, j, :])[0]
                    if gt_role_idx.size != 1:
                        raise NotImplementedError
                    gt_role_idx = gt_role_idx[0]
                    
                    det_role_idx = np.where(det_graph['pred_roles'][i][role, k, :])[0]
                    if det_role_idx.size != 1:
                        raise NotImplementedError
                    det_role_idx = det_role_idx[0]
                    
                    
                    labels_ent = []
                    for l in range(det_graph['ent_lbl'][i].shape[1]):
                        if det_graph['ent_lbl'][i][det_role_idx, l] == 0:
                            break
                        labels_ent.append(det_graph['ent_lbl'][i][det_role_idx, l])
                    if gt_graph['ent_lbl'][i][gt_role_idx] not in labels_ent:
                        flag = True
                    if localization_method == 'peak':
                        if not is_inside(det_graph['ent_peak'][i][det_role_idx], gt_graph['ent_box'][i][gt_role_idx]):
                            flag = True
                    elif localization_method == 'iou':
                        if iou(gt_graph['ent_box'][i][gt_role_idx], det_graph['ent_box'][i][det_role_idx]) < iou_thresh:
                            flag = True
                    elif localization_method == 'phrase_iou':
                        det_arg_boxes.append(det_graph['ent_box'][i][det_role_idx])
                        gt_arg_boxes.append(gt_graph['ent_box'][i][gt_role_idx])
                    elif localization_method == 'attention_correctness':
                        imgsize = image_size[gt_graph['image_id'][i]] if image_size is not None else HEATMAP_RESIZE_SHAPE
                        hm = cv2.resize(det_graph['ent_heatmap'][i][det_role_idx], imgsize, interpolation=cv2.INTER_CUBIC)
                        if attention_correctness(hm, gt_graph['ent_box'][i][gt_role_idx]) < ac_thresh:
                            flag = True
                    elif localization_method == None:
                        pass
                    else:
                        raise NotImplementedError
                if localization_method == 'phrase_iou':
                    det_phr_box = [
                        min([b[0] for b in det_arg_boxes]),
                        min([b[1] for b in det_arg_boxes]),
                        max([b[2] for b in det_arg_boxes]),
                        max([b[3] for b in det_arg_boxes]),
                    ]
                    gt_phr_box = [
                        min([b[0] for b in gt_arg_boxes]),
                        min([b[1] for b in gt_arg_boxes]),
                        max([b[2] for b in gt_arg_boxes]),
                        max([b[3] for b in gt_arg_boxes]),
                    ]
                    if iou(gt_phr_box, det_phr_box) < iou_thresh:
                        flag = True
                if flag:
                    continue
                    
                gt_matched.add(j)
                det_matched.add(k)
                '''
                print('hi', j, k, len(gt_matched), gt_graph['pred_lbl'][i][j])
                for role in range(gt_graph['pred_roles'][i].shape[0]):
                    gt_role_idx = np.where(gt_graph['pred_roles'][i][role, j, :])[0]
                    if gt_role_idx.size != 1:
                        raise NotImplementedError
                    gt_role_idx = gt_role_idx[0]
                    det_role_idx = np.where(det_graph['pred_roles'][i][role, k, :])[0]
                    if det_role_idx.size != 1:
                        raise NotImplementedError
                    det_role_idx = det_role_idx[0]                    
                    print(gt_graph['ent_lbl'][i][gt_role_idx], det_graph['ent_lbl'][i][det_role_idx], iou(gt_graph['ent_box'][i][gt_role_idx], det_graph['ent_box'][i][det_role_idx]))
                '''    
        #print('total', len(gt_graph['pred_lbl'][i]))
        per_image_precision.append(np.divide(len(det_matched), pos))
        per_image_recall.append(np.divide(len(gt_matched), len(gt_graph['pred_lbl'][i])))
        
        if verbose and i % verbose_rate == 0:
            print('{} images processed out of {}'.format(i+1, len(gt_graph["image_id"])))
    
    ret_dict = {
        'PerImageOneHopMP': np.nanmean(per_image_precision),
        'PerImageOneHopMR': np.nanmean(per_image_recall),
    }
    if args.get('return_per_image'):
        ret_dict['PerImageOneHopPrecision'] = per_image_precision
        ret_dict['PerImageOneHopRecall'] = per_image_recall

    return ret_dict
    
    
def eval_loc_pred_prec_rec_simple(det_graph, gt_graph, args={}):                
    verbose_rate = args.get('verbose_rate', None)
    verbose = args.get('verbose', False)
    localization_method = args.get('localization_method', None) # options are None, 'iou', 'peak', 'attention_correctness'
    iou_thresh = args.get('iou_thresh', None)
    ac_thresh = args.get('ac_thresh', None)
    image_size = args.get('image_size', None)
    topk = args.get('topk', None)
    
    per_image_precision = []
    per_image_recall = []
    
    for i in range(len(gt_graph['image_id'])):        
        pos = 0
        gt_matched = set()
        det_matched = set()                
        num_pred = det_graph['pred_lbl'][i].shape[0]
        if topk != None:
            num_pred = min(num_pred, topk)
        for k in range(num_pred):
            labels = []
            for l in range(det_graph['pred_lbl'][i].shape[1]):
                if det_graph['pred_lbl'][i][k, l] == 0:
                    break
                labels.append(det_graph['pred_lbl'][i][k, l])
            if len(labels) == 0:
                continue
            pos += 1
            for j in range(gt_graph['pred_lbl'][i].size):
                if gt_graph['pred_lbl'][i][j] not in labels:
                    continue
                flag = False   
                for role in range(gt_graph['pred_roles'][i].shape[0]):
                    gt_role_idx = np.where(gt_graph['pred_roles'][i][role, j, :])[0]
                    if gt_role_idx.size != 1:
                        raise NotImplementedError
                    gt_role_idx = gt_role_idx[0]
                    
                    det_role_idx = np.where(det_graph['pred_roles'][i][role, k, :])[0]
                    if det_role_idx.size != 1:
                        raise NotImplementedError
                    det_role_idx = det_role_idx[0]
                                        
                    if localization_method == 'peak':
                        if not is_inside(det_graph['ent_peak'][i][det_role_idx], gt_graph['ent_box'][i][gt_role_idx]):
                            flag = True
                    elif localization_method == 'iou':
                        if iou(gt_graph['ent_box'][i][gt_role_idx], det_graph['ent_box'][i][det_role_idx]) < iou_thresh:
                            flag = True
                    elif localization_method == 'attention_correctness':
                        imgsize = image_size[gt_graph['image_id'][i]] if image_size is not None else HEATMAP_RESIZE_SHAPE
                        hm = cv2.resize(det_graph['ent_heatmap'][i][det_role_idx], imgsize, interpolation=cv2.INTER_CUBIC)
                        if attention_correctness(hm, gt_graph['ent_box'][i][gt_role_idx]) < ac_thresh:
                            flag = True
                    elif localization_method == None:
                        pass
                    else:
                        raise NotImplementedError
                if flag:
                    continue
                    
                gt_matched.add(j)
                det_matched.add(k)
                    
        per_image_precision.append(np.divide(len(det_matched), pos))
        per_image_recall.append(np.divide(len(gt_matched), len(gt_graph['pred_lbl'][i])))
        
        if verbose and i % verbose_rate == 0:
            print('{} images processed out of {}'.format(i+1, len(gt_graph["image_id"])))
    
    ret_dict = {
        'PerImageLocPredMP': np.nanmean(per_image_precision),
        'PerImageLocPredMR': np.nanmean(per_image_recall),
    }
    if args.get('return_per_image'):
        ret_dict['PerImageLocPredPrecision'] = per_image_precision
        ret_dict['PerImageLocPredRecall'] = per_image_recall

    return ret_dict
    
    
    
    
    