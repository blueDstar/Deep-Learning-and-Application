import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

feature_map = np.array([[9, 15, 11],
                       [12, 18, 12],
                       [9, 15, 11]])

feature_map_tensor = tf.convert_to_tensor(feature_map, dtype = tf.float32)
pooled_output = tf.nn.max_pool2d(feature_map_tensor[None, ..., None],
                                 ksize = 2, strides = 1, padding = 'VALID')

print("Feature map sau khi áp dụng max pooling:")
print(pooled_output.numpy().squeeze())