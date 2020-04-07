from .graph_util_new_02 import evaluation_preproc    
from .graph_eval_3 import eval_triple_prec_rec_simple, eval_loc_pred_prec_rec_simple
    
class Evaluator(object):
    def __init__(self, args):
        self.data = {
            'image_id': [],
            'out_num_ent': [],
            'out_num_pred': [],
            'out_ent_prob': [],
            'out_pred_prob': [],
            'out_pred_roles': [],
            'out_pred_conf': [],
        }
        self.stats = {}
        self.args = dict(args)
        
    def reset(self):
        for val in self.data.values():
            val.clear()
        self.stats = {}
        
    def collect_batch(self, batch):
        for key in self.data:
            if key in batch:
                self.data[key] += list(batch[key])
            
    
    def evaluate(self):
        det_graph_val, gt_graph_val = evaluation_preproc(self.data, {
            'graph_loader': self.args['graph_loader'],
            'match_zero': False,
            'top_k': 1, 
            'proposals': self.args['proposals'],
            'num_proposals': self.args.get('num_proposals'),
            'add_full_proposal': self.args.get('add_full_proposal', False),
            'freq_prior': self.args.get('add_full_proposal', None),
            'freq_prior_weight': self.args.get('freq_prior_weight', 0.0),
            'filter_nonoverlap': self.args.get('filter_nonoverlap', False),
            'self_confidence': self.args.get('self_confidence', False),
            'rank_triplet_confidence': self.args.get('rank_triplet_confidence', True),
        })

        res = eval_triple_prec_rec_simple(
            det_graph_val,      
            gt_graph_val,      
            args={
                'localization_method': 'iou',
                'iou_thresh': 0.5,
                'topk': 100,
            })
        for key, val in res.items():
            self.stats[f'{key}_at100_iou_0.5'] = val

        res = eval_triple_prec_rec_simple(
            det_graph_val,      
            gt_graph_val,      
            args={
                'localization_method': 'iou',
                'iou_thresh': 0.5,
                'topk': 50,
            })
        for key, val in res.items():
            self.stats[f'{key}_at50_iou_0.5'] = val

        res = eval_loc_pred_prec_rec_simple(
            det_graph_val,      
            gt_graph_val,      
            args={
                'localization_method': 'iou',
                'iou_thresh': 0.5,
                'topk': 100,
            })
        for key, val in res.items():
            self.stats[f'{key}_at100_iou_0.5'] = val

        res = eval_loc_pred_prec_rec_simple(
            det_graph_val,      
            gt_graph_val,      
            args={
                'localization_method': 'iou',
                'iou_thresh': 0.5,
                'topk': 50,
            })
        for key, val in res.items():
            self.stats[f'{key}_at50_iou_0.5'] = val


        return self.stats

    def save_data(self):
        pass
    
    def save_stats(self):
        pass