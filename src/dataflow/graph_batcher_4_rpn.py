#from graph_batcher_2_rpn: pad proposals to make them constant

import os
import lmdb
import numpy as np
from .util import load_preproc_image


class GraphBatcher:
    
    def __init__(self, loader, batch_size, proposals, lmdb_path, dim_feats, seed, min_num_proposals=0):
        self.loader = loader
        self.batch_size = batch_size
        
        self.cursor = 0
        self.subset_idx = np.arange(loader.size)
        self.size = loader.size
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        
        self.seed = seed
        self.rand_gen = np.random.RandomState(seed)
        self.rand_samp = self.rand_gen.choice
        
        self.env = lmdb.open(lmdb_path, map_size=1e12, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        
        self.dim_feats = dim_feats 
        self.proposals = proposals
        self.min_num_proposals = min_num_proposals

        
    def set_subset(self, idx):
        self.subset_idx = np.asarray(idx, dtype='int32').flatten()
        self.size = self.subset_idx.shape[0]
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        
        
    def shuffle(self):
        self.rand_gen.shuffle(self.subset_idx)
        
        
    def reset(self):
        self.rand_gen.seed(self.seed)
        self.cursor = 0
        
        
    def next_batch(self, keep_cursor=False):
        new_cursor = min(self.cursor + self.batch_size, self.subset_idx.shape[0])
        idx_idx = np.arange(self.cursor, new_cursor)
        idx = self.subset_idx[idx_idx]
        
        gt_graph = self.loader.get_gt_batch(idx, pack=True)          
        feed_dict = dict(gt_graph)
                           
        features = []
        for img_id in gt_graph['image_id']:
            ft = self.txn.get(str(img_id).encode('utf-8'))
            ft = np.frombuffer(ft, 'float32')
            ft = np.reshape(ft, (-1, self.dim_feats))
            features.append(ft)
        
        num_prop = max([self.min_num_proposals] + [ft.shape[0] for ft in features])
        feat_array = np.zeros((len(features), num_prop, features[0].shape[-1]))
        for i, ft in enumerate(features):
            feat_array[i, :ft.shape[0]] = ft
            
        feed_dict['proposal_features'] = feat_array
        
        prop_box = []
        for img_id in gt_graph['image_id']:
            prop_box.append(self.proposals[img_id])
        
        box_array = np.zeros((len(features), num_prop, 4))
        for i, box in enumerate(prop_box):
            box_array[i, :box.shape[0]] = box
                
        feed_dict['proposal_boxes'] = box_array
        
        if not keep_cursor:
            self.cursor = new_cursor
            if self.cursor >= self.size:
                self.cursor = 0        
        
        return feed_dict
        
    

    
    