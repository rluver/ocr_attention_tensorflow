# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:37:42 2021

@author: MJH
"""
from layers.multi_head_attention import MultiHeadAttention
from layers.ctc_layer import CTCLayer

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    MaxPool2D, 
    Reshape, 
    BatchNormalization, 
    Dropout,
    LayerNormalization
    )




class OCR_Attention:
    
    def __init__(self, num_classes, image_height, image_width):
        
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width
        
    
    def ctc_lambda_function(self, args):
        
        labels, y_pred, input_length, label_length = args
        
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
  
    def build_model(self):

        inputs = Input(shape = (self.image_width, self.image_height, 1), name = 'input_image')
        labels = Input(shape = (None, ), name = 'input_label', dtype = 'float32')        
        
        convolution_layer_1 = Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_1_1'
            )(inputs)
            
        pooling_layer_1 = MaxPool2D(
            pool_size = (2, 2),
            strides = (2, 2),
            name = 'pooling_layer_1'
          )(convolution_layer_1)
                
            
            
        convolution_layer_2 = Conv2D(
            filters = 128, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_2_1'
            )(pooling_layer_1)
          
        pooling_layer_2 = MaxPool2D(
            pool_size = (2, 2),
            strides = (2, 2),
            name = 'pooling_layer_2_2'
          )(convolution_layer_2)
          
          
          
        convolution_layer_3_1 = Conv2D(
            filters = 256, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_3_1'          
            )(pooling_layer_2)
          
        convolution_layer_3_2 = Conv2D(
            filters = 256, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            name = 'convolution_layer_3_2',
            padding = 'same'
            )(convolution_layer_3_1)
          
        pooling_layer_3 = MaxPool2D(
            pool_size = (2, 2),
            strides = (1, 2),
            name = 'pooling_layer_3'
          )(convolution_layer_3_2)
          
          
          
        convolution_layer_4 = Conv2D(
            filters = 512, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_4_1'
            )(pooling_layer_3)    
          
        batch_normalization_layer_1 = BatchNormalization(name = 'batch_normalization_layer_1')(convolution_layer_4)
          
          
          
        convolution_layer_5 = Conv2D(
            filters = 512, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_5_1'
            )(batch_normalization_layer_1)
          
        batch_normalization_layer_2 = BatchNormalization(name = 'batch_normalization_layer_2')(convolution_layer_5)
          
        pooling_layer_4 = MaxPool2D(
              pool_size = (2, 2),
              strides = (2, 1),
              name = 'pooling_layer_4'
              )(batch_normalization_layer_2)
          
          
          
        convolution_layer_6 = Conv2D(
            filters = 512, 
            kernel_size = (2, 2),
            strides = (1, 1),
            activation = 'relu',
            padding = 'valid',
            name = 'convolution_layer_6_1'
            )(pooling_layer_4)
                  
        
        reshape_layer = Reshape((-1, 512), name = 'reshape_layer')(convolution_layer_6)
        
        
        multi_head_attention_layer = MultiHeadAttention(d_model = 512, num_heads = 8, name = 'multi_head_attention_layer')(
            {
                'query': reshape_layer,
                'key': reshape_layer,
                'value': reshape_layer
              }
            )        
        
        dropout_layer_1 = Dropout(rate = 0.5, name = 'dropout_layer_1')(multi_head_attention_layer)
        residual_layer_1 = LayerNormalization(epsilon = 1e-6, name = 'layer_normalization_layer_1')(multi_head_attention_layer + dropout_layer_1)
        
        ffnn_1 = Dense(units = 2048, activation = 'relu', name = 'ffnn_layer_1')(residual_layer_1)
        ffnn_2 = Dense(units = 512, activation = 'relu', name = 'ffnn_layer_2')(ffnn_1)
        
        dropout_layer_2 = Dropout(rate = 0.5, name = 'dropout_layer_2')(ffnn_2)
        residual_layer_2 = LayerNormalization(epsilon = 1e-6, name = 'layer_normalization_layer_2')(residual_layer_1 + dropout_layer_2)
 
        outputs = Dense(units = self.num_classes, activation = 'softmax', name = 'classification_layer')(residual_layer_2)        
        ctc_loss = CTCLayer(name = 'ctc_loss')(labels, outputs)
        
        model = Model(inputs = [inputs, labels], outputs = ctc_loss)
        model.compile(optimizer = tf.keras.optimizers.Adam())
        
        return model