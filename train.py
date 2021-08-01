# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:55:52 2021

@author: MJH
"""
from model import OCR_Attention
from auxiliary import *

import json
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.kera.scallbacks import EarlyStopping, TensorBoard
from pathlib import Path
from tqdm import tqdm




def main(korean_image_path, korean_information_path, english_image_path):
    
    ### korean ###
    korean_data_dir = Path(korean_image_path)
    korean_images = sorted(list(map(str, list(korean_data_dir.glob('*.png')))))
    
    with open(korean_information_path, 'r', encoding = 'utf-8') as f:
        korean_data_info = json.loads(f.read())
        
    korean_word_image_data_info = list(filter(lambda x: x['file_name'][:2] == '02', korean_data_info['images']))
    korean_max_width = max(list(map(lambda x: x['width'], korean_word_image_data_info)))
    korean_max_height = max(list(map(lambda x: x['height'], korean_word_image_data_info)))
        
    korean_annotations = list(filter(lambda x: x['image_id'][:3] in ['022', '023', '024'], korean_data_info['annotations']))    
    existing_img_id_list = list(map(lambda x: x.split('\\')[-1].split('.')[0], korean_images))
    
    korean_annotations = list(filter(lambda x: x['image_id'] in existing_img_id_list, tqdm(korean_annotations)))    
    korean_word_image_data_info = list(filter(lambda x: x['id'] in existing_img_id_list, tqdm(korean_word_image_data_info)))
        
    korean_labels = [img['text'] for img in tqdm(korean_annotations)]
    korean_characters = set(char for label in korean_labels for char in label)
    
    print('Number of images found: ', len(korean_word_image_data_info))
    print('Number of labels found: ', len(korean_labels))
    print('Number of unique characters: ', len(korean_characters))
    
    
    ### english ###
    english_data_base_dir = Path(english_image_path)
    with open(os.path.join(english_data_base_dir, 'imlist.txt'), 'r') as f:
        english_images = f.read().split('\n')
    english_images = list(map(lambda x: os.path.join(english_data_base_dir, x[2:]), english_images))
    
    english_labels = [img.split(os.path.sep)[-1].split('.jpg')[0].split('/')[-1].split('_')[1] for img in tqdm(english_images)]
    english_characters = set(char for label in tqdm(english_labels) for char in label)
        
    print('Number of images found: ', len(english_images))
    print('Number of labels found: ', len(english_labels))
    print('Number of unique characters: ', len(english_characters))
    
    
    ### append ###
    temp_labels = [korean_labels, english_labels]
    temp_labels = list(itertools.chain(*temp_labels))
    
    max_length = max([len(label) for label in temp_labels])
    # padding
    temp_labels = list(map(lambda x: ''.join([x, ' ' * (max_length - len(x))]), temp_labels))
    
    characters = [korean_characters, english_characters]
    characters = list(itertools.chain(*characters))
    characters.append(' ')
    with open('word.json', 'w', encoding = 'utf-8') as f:
        f.write(str(characters))
    
    
    temp_images = [korean_images, english_images]
    temp_images = list(itertools.chain(*images))
    
    
    ### filter ###
    images = []
    labels = []
    
    for idx, temp in tqdm(enumerate(zip(temp_images, temp_labels))):
        image, label = temp
        try:
            image_contents = tf.io.read_file(image)
            tf.image.decode_jpeg(image_contents)
            
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(e)
    
    
    print(f'max_height: {korean_max_height}')
    print(f'max_width: {korean_max_width}')
        
    BATCH_SIZE = 192
    IMAGE_HEIGHT = korean_max_height
    IMAGE_WIDTH = 650


    # preprocessing
    char_to_num = StringLookup(
        vocabulary = list(characters), num_oov_indices = 0, mask_token = None
    )
        

    # Splitting data into training and validation sets
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    
    # tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    )
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    )
    
    
    # # visualize data
    # _, ax = plt.subplots(4, 4, figsize = (10, 5))
    # for batch in train_dataset.take(1):
    #     images = batch['input_layer']
    #     labels = batch['label_input']
    #     for i in range(16):
    #         img = (images[i] * 255).numpy().astype('uint8')
    #         label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
    #         ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap = 'gray')
    #         ax[i // 4, i % 4].set_title(label)
    #         ax[i // 4, i % 4].axis('off')
    # plt.show()
    
    
    config = {
        'num_classes': len(characters) + 1,
        'image_height': IMAGE_HEIGHT,
        'image_width': IMAGE_WIDTH
        }
    
    with open('config.json', 'w', encoding = 'utf-8') as f:
        f.write(str(config))
    
    
    model = OCR_Attention(**config).build_model()
    model.summary()
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = EARLY_STOPPING_PATIENCE,
        restore_best_weights = True
        )
    
    tensorboard = TensorBoard(
        log_dir = 'log'
        )    
    
    history = model.fit(
        train_dataset,
        validation_data = validation_dataset,
        epochs = EPOCHS,
        callbacks = [early_stopping, tensorboard]
        )
    



if __name__ == '__main__':
    
    korean_image_path = 'dataset/word'
    korean_information_path = 'dataset/printed_data_ionfo.json'
    english_image_path = 'dataset/word_eng'
        
    main(korean_image_path, korean_information_path, english_image_path)