#from IPython.display import Image, display
#display(Image(filename='test.png', width=420))
#display(Image(filename='test1.png', width=420))

import tensorflow as tf
import numpy as np
import skimage.io as io
import skimage.transform as transform
from os.path import join
import network_vfn as nw
import json
import time

global_dtype = tf.float32
global_dtype_np = np.float32

# In the following example, we are going to feed only one image into the network
batch_size = 1

# TODO: Change this if your model file is located somewhere else
snapshot = '../model-spp-max'

tf.reset_default_graph()
embedding_dim = 1000
ranking_loss = 'svm'
net_data = np.load('alexnet.npy', encoding='bytes').item()
image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size,227,227,3])
var_dict = nw.get_variable_dict(net_data)
SPP = True
pooling = 'max'
with tf.variable_scope("ranker") as scope:
   feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
   score_func = nw.score(feature_vec)
# load pre-trained model
saver = tf.train.Saver(tf.global_variables())
sess = tf.Session(config=tf.ConfigProto())
sess.run(tf.global_variables_initializer())
saver.restore(sess, snapshot)


# This is the definition of helper function
def evaluate_aesthetics_score(images):
    scores = np.zeros(shape=(len(images),))
    for i in range(len(images)):
        img = images[i].astype(np.float32)/255
        img_resize = transform.resize(img, (227, 227))-0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        scores[i] = sess.run([score_func], feed_dict={image_placeholder: img_resize})[0]
    return scores

# add the following codes in the main function
images = [
    io.imread('test.jpg')[:,:,:3],   # remember to replace with the filename of your test image
    io.imread('test1.jpg')[:,:,:3]   # remember to replace with the filename of your test image
]
scores = evaluate_aesthetics_score(images)
print ('Poorly Cropped Image vs Well Cropped Image')
print (scores)



