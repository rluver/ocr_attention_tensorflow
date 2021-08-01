# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:48:06 2021

@author: MJH

#refer: https://keras.io/examples/vision/captcha_ocr/
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.strings import reduce_join
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


IMAGE_HEIGHT, IMAGE_WIDTH = 150, 650



with open('word.json', 'r', encoding = 'utf-8') as f:
    characters = eval(f.read())

char_to_num = StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)




def split_data(images, labels, train_size = 0.9, shuffle = True):    
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
    
    

def encode_single_sample(img_path, label = False):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels = 1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm = [1, 0, 2])
    try:
        # 6. Map the characters in label to numbers
        label = char_to_num(tf.strings.unicode_split(label, input_encoding = 'UTF-8'))
        # 7. Return a dict as our model is expecting two inputs
        return {'input_image': img, 'input_label': label}   
    except:
        return {'input_image': img}



def decode_batch_predictions(pred, max_length):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = K.ctc_decode(pred, input_length = input_len, greedy = True)[0][0][
        :, : max_length
    ]
    
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
        
    return output_text