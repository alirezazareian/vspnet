import cv2
import numpy as np

def load_preproc_image(path, preproc_type):
    with open(path, 'rb') as fin:
        img_bgr = cv2.imdecode(np.frombuffer(fin.read(), dtype='uint8'), cv2.IMREAD_COLOR)
    if preproc_type == 'VGG_force':
        return load_preproc_image_vgg(img_bgr, mode='force')
    elif preproc_type == 'VGG_crop':
        return load_preproc_image_vgg(img_bgr, mode='crop')
    elif preproc_type == 'VGG_crop_aug':
        return load_preproc_image_vgg(img_bgr, mode='crop_aug')
    elif preproc_type == 'VGG_flex':
        return load_preproc_image_vgg(img_bgr, mode='flex')
    elif preproc_type == 'Inception':
        return load_preproc_image_inception(img_bgr)
    else:
        raise NotImplementedError
        
        
def load_preproc_image_vgg(img_bgr, mode):
    if mode == 'flex' or mode == 'crop' or mode == 'crop_aug':
        lowest_dim = 256
        highest_dim = 456 if mode == 'flax' else 10000
        h, w, _ = img_bgr.shape
        scale = lowest_dim / min(h, w)
        if highest_dim != None and scale * max(h, w) > highest_dim:
            scale = highest_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
        
        if mode == 'crop':
            midr = int(img_bgr.shape[0] / 2)
            midc = int(img_bgr.shape[1] / 2)
            img_bgr = img_bgr[midr - 112:midr + 112, midc - 112:midc + 112, :]
        elif mode == 'crop_aug':
            min_midr = 112
            max_midr = img_bgr.shape[0] - 112
            min_midc = 112
            max_midc = img_bgr.shape[1] - 112
            #print(img_bgr.shape)
            midr = np.random.randint(min_midr, max_midr+1)
            midc = np.random.randint(min_midc, max_midc+1)
            img_bgr = img_bgr[midr - 112:midr + 112, midc - 112:midc + 112, :]
            if np.random.rand() < 0.5:
                img_bgr = img_bgr[:, ::-1, :]
            
    elif mode == 'force':
        img_bgr = cv2.resize(img_bgr, (224, 224))
        
    img_rgb = img_bgr[:,:,[2,1,0]]
    img_rgb = img_rgb.astype('float32') - [[[123.68, 116.78, 103.94]]]
    return img_rgb


def load_preproc_image_inception(img_bgr):
    img_bgr = cv2.resize(img_bgr, (331, 331))
    img_rgb = img_bgr[:,:,[2,1,0]]
    img_rgb = (img_rgb.astype('float32') / (255. / 2.)) - 1.0
    return img_rgb


def stack_dynamic_pad(images):
    padded_images = []
    max_w = 0
    max_h = 0
    for img in images:
        if img.shape[0] > max_h:
            max_h = img.shape[0]
        if img.shape[1] > max_w:
            max_w = img.shape[1]
    for img in images:
        extra_h = max_h - img.shape[0]
        if extra_h % 2 == 0:
            x_h_before = int(extra_h / 2)
            x_h_after = x_h_before
        else:
            x_h_before = int((extra_h - 1) / 2)
            x_h_after = x_h_before + 1

        extra_w = max_w - img.shape[1]
        if extra_w % 2 == 0:
            x_w_before = int(extra_w / 2)
            x_w_after = x_w_before
        else:
            x_w_before = int((extra_w - 1) / 2)
            x_w_after = x_w_before + 1
                        
        p_img = np.pad(img, [(x_h_before, x_h_after), (x_w_before, x_w_after), (0, 0)], 'constant')
        padded_images.append(p_img)
        
    batch = np.asarray(padded_images, dtype='float32')    
    return batch