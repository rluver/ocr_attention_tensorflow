# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 23:13:26 2021

@author: MJH
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import OCR_Attention
from auxiliary import decode_batch_predictions, encode_single_sample
from tensorflow.keras.models import Model




class OCR:
    
    def __init__(self, config_path, model_path, max_length):
        with open(config_path, 'r', encoding = 'utf-8') as f:
            config = eval(f.read())            
        self.model = OCR_Attention(**config).build_model()
        
        # prediction model
        self.model = Model(
            self.model.get_layer(name = 'input_image').input,
            self.model.get_layer(name = 'classification_layer').output
            )
        self.model.load_weights(model_path)
        self.model.summary()
        
        self.max_length = max_length
        
        
    def predict(self, image_path):
        
        image = tf.data.Dataset.from_tensor_slices([image_path])
        image = (
            image.map(
                encode_single_sample, num_parallel_calls = tf.data.experimental.AUTOTUNE
                )
            .batch(1)
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            )
        
        image = list(image.take(1))
        self.image = image
        
        preds = self.model.predict(image)
        pred_texts = decode_batch_predictions(preds, self.max_length)
        self.pred_texts = pred_texts
        
        return pred_texts
    
    
    def visualize(self):
        
        _, ax = plt.subplots(1, 1, figsize = (2, 2))
        try:
            image = (self.image[0]['input_image'][0, :, :, 0] * 255).numpy().astype(np.uint8)
            image = image.T
            title = f'Prediction: {self.pred_texts}'
            ax.imshow(image, cmap = 'gray')
            ax.set_title(title)
            ax.axis('off')
            plt.show()
            
        except Exception as e:
            print(e)




if __name__ == '__main__':

    image_path = ''    
    
    ocr = OCR('config.json', 'model/model', 25)
    ocr.predict(image_path)
    ocr.visualize()