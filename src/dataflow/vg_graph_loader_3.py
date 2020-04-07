# from vg_graph_loader_1.py: loading stanford VG metadata instead of original

import os
import json
import pickle
import cv2

from hdf5storage import savemat, loadmat
import numpy as np
import tensorflow as tf

class VGGraphLoader:
    def __init__(self, scene_graph_path, embedding_path, images_path=None):
        self.images_path = images_path
        with open(scene_graph_path, 'rb') as fin:
            sg_dict = pickle.load(fin)    
            
        self.image_id = sg_dict['img_ids']
        self.entity_label = sg_dict['entity_label']
        self.entity_box_coords = sg_dict['entity_box_coords']
        self.pred_label = sg_dict['pred_label']
        #self.pred_edges = sg_dict['pred_edges']
        self.pred2subj = sg_dict['pred2subj']
        self.pred2obj = sg_dict['pred2obj']
        self.triples = sg_dict['triples']
        
        with open(embedding_path, 'rb') as fin:
            self.scene_graph_meta, self.noun_emb_dict, self.pred_emb_dict = pickle.load(fin)
                                        
        self.emb_dim = self.noun_emb_dict.shape[-1]        
        self.size = self.image_id.size
        self.imgid2idx = {imgid: idx for idx, imgid in enumerate(self.image_id)}
        
        self.entity_emb = [self.noun_emb_dict[ents] if ents.size > 0 else np.zeros((0, self.emb_dim)) for ents in self.entity_label]
        self.pred_emb = [self.pred_emb_dict[preds] if preds.size > 0 else np.zeros((0, self.emb_dim)) for preds in self.pred_label]
            
            
    def get_gt_batch(self, idx_list, pack):
        image_id = [self.image_id[idx] for idx in idx_list]
        ent_lbl = [self.entity_label[idx] for idx in idx_list]
        pred_lbl = [self.pred_label[idx] for idx in idx_list]
        ent_box = [self.entity_box_coords[idx] for idx in idx_list]
        pred_roles = [np.stack((self.pred2subj[idx], self.pred2obj[idx])) for idx in idx_list]
        ent_emb = [self.entity_emb[idx] for idx in idx_list]
        pred_emb = [self.pred_emb[idx] for idx in idx_list]
        
        gt_graph = {
            'image_id': image_id,
            'ent_lbl': ent_lbl,
            'ent_box': ent_box,
            'pred_lbl': pred_lbl,
            'pred_roles': pred_roles,
            'ent_emb': ent_emb,
            'pred_emb': pred_emb,            
        }
        
        if pack:        
            num_entities = []
            num_preds = []
            for i in range(len(idx_list)):
                num_entities.append(gt_graph['ent_emb'][i].shape[0])
                num_preds.append(gt_graph['pred_emb'][i].shape[0])
            num_entities = np.asarray(num_entities, dtype='int32')
            num_preds = np.asarray(num_preds, dtype='int32') 
            max_n_ent = np.max(num_entities)
            max_n_pred = np.max(num_preds)

            ent_lbl = np.zeros((len(idx_list), max_n_ent,), dtype='int32')
            pred_lbl = np.zeros((len(idx_list), max_n_pred,), dtype='int32')
            ent_emb = np.zeros((len(idx_list), max_n_ent, self.emb_dim), dtype='float32')
            pred_emb = np.zeros((len(idx_list), max_n_pred, self.emb_dim), dtype='float32')
            ent_box = np.zeros((len(idx_list), max_n_ent, 4), dtype='float32')
            pred_roles = np.zeros((len(idx_list), gt_graph['pred_roles'][0].shape[0], max_n_pred, max_n_ent), dtype='bool')

            for i in range(len(idx_list)):
                if max_n_ent > 0:
                    if num_entities[i] > 0:
                        ent_lbl[i, :num_entities[i]] = gt_graph['ent_lbl'][i]
                        ent_emb[i, :num_entities[i]] = gt_graph['ent_emb'][i]
                        ent_box[i, :num_entities[i]] = gt_graph['ent_box'][i]
                if max_n_pred > 0:
                    assert(max_n_ent > 0)
                    if num_preds[i] > 0:
                        pred_lbl[i, :num_preds[i]] = gt_graph['pred_lbl'][i]
                        pred_emb[i, :num_preds[i]] = gt_graph['pred_emb'][i]
                        if num_entities[i] > 0:
                            pred_roles[i, :, :num_preds[i], :num_entities[i]] = gt_graph['pred_roles'][i]

            image_id = np.asarray(gt_graph['image_id'])
                
            return {
                'image_id': image_id,
                'ent_lbl': ent_lbl,
                'ent_box': ent_box,
                'pred_lbl': pred_lbl,
                'pred_roles': pred_roles,
                'ent_emb': ent_emb,
                'pred_emb': pred_emb,
                'num_ent': num_entities,
                'num_pred': num_preds,
            }
        
        return gt_graph

