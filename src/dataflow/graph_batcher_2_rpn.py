import os
import lmdb
import numpy as np
from .util import load_preproc_image


class GraphBatcher:
    
    def __init__(self, loader, batch_size, proposals, lmdb_path, dim_feats, seed, num_proposals=None):
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
        self.num_proposals = num_proposals

        self.key_type = type(list(self.proposals.keys())[0])
        
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
            if self.num_proposals != None: 
                if ft.shape[0] > self.num_proposals:
                    ft = ft[:self.num_proposals]
                elif ft.shape[0] < self.num_proposals:
                    ft = np.concatenate((ft, np.zeros((self.num_proposals - ft.shape[0], self.dim_feats), dtype=ft.dtype)), axis=0)
            features.append(ft)
        features = np.stack(features)  
        feed_dict['proposal_features'] = features
        
        prop_box = []
        for img_id in gt_graph['image_id']:
            pb = self.proposals[self.key_type(img_id)]
            if self.num_proposals != None: 
                if pb.shape[0] > self.num_proposals:
                    pb = pb[:self.num_proposals]
                elif pb.shape[0] < self.num_proposals:
                    pb = np.concatenate((pb, np.zeros((self.num_proposals - pb.shape[0], 4), dtype=pb.dtype)), axis=0)
            prop_box.append(pb)
        prop_box = np.asarray(prop_box)
        feed_dict['proposal_boxes'] = prop_box
        
        if not keep_cursor:
            self.cursor = new_cursor
            if self.cursor >= self.size:
                self.cursor = 0        
        
        return feed_dict
        
    

    
    