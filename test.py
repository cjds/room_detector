import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def main():

  output='captions.h5'
  root='mscoco/train2014/'
  data = json.load(open("annotations/captions_val2014.json", 'r'))
  images=[] #IMAGES
  L=[]

  for v in data['images']:
  	images.append(v['file_name'])
 
  for v in data['annotations']:
  	print v
  
  f = h5py.File(output, "w")

  #
  f.create_dataset("labels", dtype='uint32', data=L)

  dset = f.create_dataset("images", (N,3,256,256), dtype='uint8') # space for resized images
  for img in images:
    # load the image
    I = imread(os.path.join(root, img))
    try:
        Ir = imresize(I, (256,256))
    except:
        print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
        raise

    # handle grayscale input images
    if len(Ir.shape) == 2:
      Ir = Ir[:,:,np.newaxis]
      Ir = np.concatenate((Ir,Ir,Ir), axis=2)
    # and swap order of axes from (256,256,3) to (3,256,256)
    Ir = Ir.transpose(2,0,1)
    # write to h5
    dset[i] = Ir
    if i % 1000 == 0:
      print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
  f.close()

  print 'wrote images'

  
if __name__ == "__main__":
  main()
