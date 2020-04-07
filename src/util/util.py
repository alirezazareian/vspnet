import os
import time

import numpy
import hickle
import pickle


def split(lst, sizes):
    output = []
    cursor = 0
    for s in sizes:
        output.append (lst[cursor:cursor+s])
        cursor += s
    return tuple (output)
    
    
def tic ():
    return time.time()

def toc (start_time, label=None):
    t = time.time() - start_time
    if label is not None:
        print (label+' Time: %.2f'%t)
    return t

    
def round_floats(value, digits):
    if isinstance(value, float) or isinstance(value, numpy.float32) or isinstance(value, numpy.float64):
        return round(value, digits)
    elif isinstance(value, dict):
        return {k:round_floats(v, digits) for k,v in value.items()}
    else:
        return value
    
    
def pickle_save(obj, filename, hdf5=False):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w' if hdf5 else 'wb') as fout:
        (hickle if hdf5 else pickle).dump(obj, fout)
        
def pickle_load(filename, hdf5=False):
    with open(filename, 'r' if hdf5 else 'rb') as fout:
        data = (hickle if hdf5 else pickle).load(fout)
    return data
        