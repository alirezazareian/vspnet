import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

slim = tf.contrib.slim

act_fn_dict = {
    'relu': tf.nn.relu,
    'leaky_relu': lambda x: tf.maximum(x, 0.1 * x),
    'swish': lambda x: x * tf.nn.sigmoid(x),
}

def mlp(x, num_unit_list, act_fn_hid, act_fn_last, name, reuse=None, dropout=None, dropout_last=True):
    with tf.variable_scope(name, reuse=reuse):
        for i, d in enumerate(num_unit_list):
            x = tf.layers.dense(x, d, activation=act_fn_hid if i < len(num_unit_list) - 1 else act_fn_last, name=f'dense_layer_{i}')
            if dropout != None and (dropout_last or i < len(num_unit_list) - 1):
                x = tf.nn.dropout(x, (1.0 - dropout))
    return x

def MLP(num_unit_list, act_fn_hid, act_fn_last, name, reuse=None, dropout=None, dropout_last=True):
    return lambda x: mlp(x, num_unit_list, act_fn_hid, act_fn_last, name, reuse=reuse, dropout=dropout, dropout_last=dropout_last)

class MLPClass:
    def __init__(self, num_unit_list, act_fn_hid, act_fn_last, name, reuse=None, dropout=None, dropout_last=True):
        self.num_unit_list = num_unit_list
        self.act_fn_hid = act_fn_hid
        self.act_fn_last = act_fn_last
        self.name = name
        self.reuse = reuse
        self.dropout = dropout
        self.dropout_last = dropout_last
        self.scope = tf.get_variable_scope()
    def __call__(self, x):
        with tf.variable_scope(self.scope):
            ret = mlp(x, self.num_unit_list, self.act_fn_hid, self.act_fn_last, self.name, reuse=self.reuse, dropout=self.dropout, dropout_last=self.dropout_last)
        self.reuse = True
        return ret
    
def nd_batch_proc(x, fn, num_batch_dim=-1):
    sh = get_tensor_shape(x)
    x = tf.reshape(x, [-1] + sh[num_batch_dim:])
    x = fn(x)
    sh2 = get_tensor_shape(x)
    x = tf.reshape(x, sh[:num_batch_dim] + sh2[1:])    
    return x

def relaxed_softmax_old(x, null_logit=0.0, heat=1.0):
    cc_shape = get_tensor_shape(x)
    cc_shape[-1] = 1
    x = tf.concat((x, null_logit * tf.ones(cc_shape, dtype=x.dtype)), axis=-1) / heat
    x = tf.nn.softmax(x)
    x = x[..., :-1]            
    return x

def relaxed_softmax(x, null_logit=0.0, heat=1.0):
    cc_shape = get_tensor_shape(x)
    cc_shape[-1] = 1
    x = x - ((tf.reduce_min(x) + tf.reduce_max(x)) / 2.0)
    x = tf.concat((x, null_logit * tf.ones(cc_shape, dtype=x.dtype)), axis=-1) / heat
    x = tf.nn.softmax(x)
    x = x[..., :-1]            
    return x

def mask_keep_topk(x, k, binary=False):
    _, topk = tf.math.top_k(x, k)
    mask = tf.one_hot(topk, get_tensor_shape(x, -1), dtype=x.dtype)
    mask = tf.reduce_sum(mask, axis=-2)
    if binary:
        return mask
    else:
        return x * mask



    
def roi_align(feature_map, box_coords, output_shape=None, pad_border=True, subregion_resolution=2):
    """
    feature_map: (b, h, w, c) float32
    box_coords: (b, n, 4) float32 with values (x1, y1, x2, y2) in range [0, 1]
    output_shape: tuple of int: (height, width). None means (1, 1) followed by squeeze
    pad_border: whether to do symmetric padding or not. this can be quite slow
    """
    def transform_fpcoor_for_tf(boxes, image_shape, output_shape):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(output_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(output_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(output_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(output_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # getting shapes
    b = get_tensor_shape(feature_map, 0)
    h = get_tensor_shape(feature_map, 1)
    w = get_tensor_shape(feature_map, 2)
    c = get_tensor_shape(feature_map, 3)
    n = get_tensor_shape(box_coords, 1)
    
    boxes = tf.reshape(box_coords, [b * n, 4])
    box_ind = tf.reshape(tf.tile(tf.range(b)[:, tf.newaxis], [1, n]), [b * n])
    
    # identifying zero boxes
    zero_boxes = tf.reduce_all(tf.equal(boxes, 0.0), axis=1)
    
    # unnormalizing boxes
    boxes = tf.maximum(0., tf.minimum(1., boxes))
    boxes = tf.stack((
        boxes[:, 0] * w,
        boxes[:, 1] * h,
        boxes[:, 2] * w,
        boxes[:, 3] * h,
    ), axis=-1)

    if pad_border:        
        feature_map = tf.pad(feature_map, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1
    
    # Expand bbox to a minium size of 1
    boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    boxes_wh = boxes_x2y2 - boxes_x1y1
    boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    boxes_newwh = tf.maximum(boxes_wh, 1.)
    boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)
    
    resize_shape = output_shape if output_shape != None else [1, 1]
    resize_shape = [resize_shape[0] * subregion_resolution, resize_shape[1] * subregion_resolution]
    
    boxes = transform_fpcoor_for_tf(boxes, [h, w], resize_shape)
    ret = tf.image.crop_and_resize(
        feature_map, boxes, tf.cast(box_ind, tf.int32),
        crop_size=resize_shape)
    
    ret = tf.where(tf.broadcast_to(zero_boxes[:, tf.newaxis, tf.newaxis, tf.newaxis], tf.shape(ret)), tf.zeros_like(ret), ret)
    
    if subregion_resolution > 1:
        ret = slim.max_pool2d(ret, [subregion_resolution, subregion_resolution], scope='pool5')
    
    ret = tf.reshape(ret, [b, n] + (list(output_shape) if output_shape != None else []) + [c])        
    return ret    
    
    



'''    
def _roi_align_op(feature_map, box_coords, output_shape):
    slc = tf.slice(feature_map, (box_coords[1], box_coords[0], 0), (box_coords[3], box_coords[2], -1))
    if output_shape is None:
        out = tf.reduce_sum(slc, axis=[0, 1])
    else:
        out = tf.image.resize_bilinear(slc[tf.newaxis, ...], output_shape)
    return out


def roi_align(feature_map, box_coords, output_shape=None):
    """
    feature_map: (b, h, w, c) float32
    box_coords: (b, n, 4) float32 with values (x1, y1, x2, y2) in range [0, 1]
    """
    b = get_tensor_shape(feature_map, 0)
    h = get_tensor_shape(feature_map, 1)
    w = get_tensor_shape(feature_map, 2)
    c = get_tensor_shape(feature_map, 3)
    n = get_tensor_shape(box_coords, 1)
    
    def _roi_align_op(feature_map, box_coords):
        slc = tf.slice(feature_map, (box_coords[1], box_coords[0], 0), (box_coords[3], box_coords[2], c))
        if output_shape is None:
            out = tf.reduce_sum(slc, axis=[0, 1])
        else:
            out = tf.image.resize_bilinear(slc[tf.newaxis, ...], output_shape)[0]
        return out
    
    box_coords = tf.stack((
        box_coords[:, :, 0] * w,
        box_coords[:, :, 1] * h,
        box_coords[:, :, 2] * w,
        box_coords[:, :, 3] * h,
    ), axis=-1)
    box_coords = tf.cast(tf.round(box_coords), tf.int32)

    box_coords = tf.stack((
        tf.minimum(w - 1, tf.maximum(0, box_coords[:, :, 0])),
        tf.minimum(h - 1, tf.maximum(0, box_coords[:, :, 1])),
        tf.minimum(w, box_coords[:, :, 2]),
        tf.minimum(h, box_coords[:, :, 3]),
    ), axis=-1)        
    box_coords = tf.stack((
        box_coords[:, :, 0],
        box_coords[:, :, 1],
        tf.maximum(1, box_coords[:, :, 2] - box_coords[:, :, 0]),
        tf.maximum(1, box_coords[:, :, 3] - box_coords[:, :, 1]),
    ), axis=-1)
    feature_map = tf.tile(feature_map[:, tf.newaxis, :, :, :], [1, n, 1, 1, 1])
    feature_map = tf.reshape(feature_map, [b * n, h, w, c])
    box_coords = tf.reshape(box_coords, [b * n, 4])
    out = tf.map_fn(
        lambda x: _roi_align_op(x[0], x[1]),
        (feature_map, box_coords), 
        dtype=tf.float32
    )
    out = tf.reshape(out, [b, n] + get_tensor_shape(out)[1:])
    return out
'''    




def get_tensor_shape(tensor, axis=None):
    if axis is None:
        unknown_shape = tf.shape(tensor)
        shape = []
        for i, item in enumerate(tensor.get_shape()):
            if str(item) == '?':
                shape.append(unknown_shape[i])
            else:
                shape.append(int(item))
        return list(shape)
    else:
        if str(tensor.get_shape()[axis]) == '?':
            return tf.shape(tensor)[axis]
        else:
            return int(tensor.get_shape()[axis])
    
    
def get_variables_starting_with(name, trainables_only=False):
    variables = tf.trainable_variables() if trainables_only else tf.global_variables()
    return [v for v in variables if v.name.startswith(name)]


def list_variables_in_file(filename):
    print_tensors_in_checkpoint_file(filename, '', False, True)    
    

def upload_to_gpu(value, name, sess):
    ph = tf.placeholder (value.dtype)
    ingpu = tf.Variable (initial_value=ph, trainable=False, validate_shape=False, name=name)
    sess.run (ingpu.initializer, feed_dict={ph: value})
    return ingpu

