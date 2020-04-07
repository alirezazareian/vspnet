import numpy as np

HEATMAP_RESIZE_SHAPE = (1024, 1024)

def evaluate(model_output, eval_func_list, args={}):
    output_dict = {}
    for fn in eval_func_list:
        res = fn(model_output, args)
        output_dict.update(res)
        
    return output_dict

def evaluate_multi(model_output, eval_func_list, args_list, label_list):
    output_dict = {}
    for fn, args, label in zip(eval_func_list, args_list, label_list):
        res = fn(model_output, args)
        for key, val in res.items():
            output_dict[f'{label}_{key}'] = val
    return output_dict


def accuracy_for_keys(model_output, args):
    keys = args['keys']
    
    num_correct = {k: 0 for k in keys}
    num_incorrect = {k: 0 for k in keys}
    for batch in model_output:
        for k in keys:
            num_correct[k] += batch[k]['num_correct']
            num_incorrect[k] += batch[k]['num_incorrect']
    
    acc = {k + '_accuracy': num_correct[k] / (num_correct[k] + num_incorrect[k]) 
           for k in keys}
    
    if args.get('verbose'):
        print(acc)
    
    return acc


def iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intsec = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    iou = intsec / (area1 + area2 - intsec)
    return iou

def pw_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    intsec = (np.maximum(0.0, np.minimum(box1[:, np.newaxis, 2], box2[np.newaxis, :, 2]) - 
                              np.maximum(box1[:, np.newaxis, 0], box2[np.newaxis, :, 0])) * 
              np.maximum(0.0, np.minimum(box1[:, np.newaxis, 3], box2[np.newaxis, :, 3]) - 
                              np.maximum(box1[:, np.newaxis, 1], box2[np.newaxis, :, 1])))
    iou = intsec / (area1[:, np.newaxis] + area2[np.newaxis, :] - intsec)
    return iou


def is_inside(dot, box):
    return dot[0] >= box[0] and dot[0] <= box[2] and dot[1] >= box[1] and dot[1] <= box[3]


def attention_correctness(heatmap, box):
    heatmap = heatmap / heatmap.sum()

    box_scaled = [int(box[0] * heatmap.shape[1]), 
                  int(box[1] * heatmap.shape[0]), 
                  int(box[2] * heatmap.shape[1]), 
                  int(box[3] * heatmap.shape[0])]
    
    return np.sum(heatmap[box_scaled[1]:box_scaled[3], box_scaled[0]:box_scaled[2]])
    
    
def heatmap_iou(heatmap, box, max_box_volume=1.0):
    heat_volume = heatmap.sum()

    box_scaled = [int(box[0] * heatmap.shape[1]), 
                  int(box[1] * heatmap.shape[0]), 
                  int(box[2] * heatmap.shape[1]), 
                  int(box[3] * heatmap.shape[0])]
    box_area = (box_scaled[3] - box_scaled[1]) * (box_scaled[2] - box_scaled[0])
    box_height = max_box_volume / (heatmap.shape[0] * heatmap.shape[1])
    box_volume = box_height * box_area
    
    intersec = np.sum(heatmap[box_scaled[1]:box_scaled[3], box_scaled[0]:box_scaled[2]]) * box_height
    iou = intersec / (heat_volume + box_volume - intersec)
    return iou


def heatmap_iou_batch(heatmap, boxes, max_box_volume=1.0):
    heat_volume = heatmap.sum()
    heatmap_integral = np.cumsum(np.cumsum(heatmap, axis=0), axis=1)

    box_scaled = np.copy(boxes)
    box_scaled[:, 0] *= (heatmap.shape[1] - 1)
    box_scaled[:, 1] *= (heatmap.shape[0] - 1)
    box_scaled[:, 2] *= (heatmap.shape[1] - 1)
    box_scaled[:, 3] *= (heatmap.shape[0] - 1)
    box_scaled = np.round(box_scaled).astype('int32')
    
    box_area = (box_scaled[:,3 ] - box_scaled[:, 1]) * (box_scaled[:, 2] - box_scaled[:, 0])
    box_height = max_box_volume / (heatmap.shape[0] * heatmap.shape[1])
    box_volume = box_height * box_area
    
    intersec = (heatmap_integral[box_scaled[:, 3], box_scaled[:, 2]] - 
                heatmap_integral[box_scaled[:, 3], box_scaled[:, 0]] - 
                heatmap_integral[box_scaled[:, 1], box_scaled[:, 2]] + 
                heatmap_integral[box_scaled[:, 1], box_scaled[:, 0]]
               ) * box_height
    
    iou = intersec / (heat_volume + box_volume - intersec)
    return iou




