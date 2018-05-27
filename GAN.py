# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:56:59 2018

@author: Saurabh
"""

from keras import optimizers
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.datasets import mnist
import sys

import matplotlib.pyplot as plt 

    

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.noise_shape = (100,)
        
        optimizer = optimizers.Adam(0.0002, 0.5)
        
        '''
        Building a Discriminator
        '''
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics = ['accuracy'])
        
        '''
        Building a Generator
        '''
        
        self.generator = self.build_generator()
        self.generator.compile(loss = 'binary_crossentropy',optimizer = optimizer)
        
        '''
        WE ARE NOT TRAINING DISCRIMINATOR WHILE USING COMBINED MODEL
        '''
        self.discriminator.trainable = False
        
        '''
        generating Image
        '''
        noise_zspace = Input(shape = self.noise_shape)
        
        img = self.generator(noise_zspace)
        
        valid = self.discriminator(img)
        
        self.combined_model = Model(noise_zspace,valid)
        self.combined_model.compile(loss = 'binary_crossentropy',optimizer = optimizer)
        
    
                
    
    def build_discriminator(self):
        
        '''
        Neural Network Architecture 
        '''
        model = Sequential()
        model.add(Flatten(input_shape = self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dense(1,activation = 'sigmoid'))
        
        print ('Discriminator Summary')
        model.summary()
        ##Doubt:: why are not we directly returning model
        ##Doubt:: what does this Input function return
        img = Input(shape = self.img_shape)
        validity = model(img)
        
        return Model(img,validity)
    
    def build_generator(self):
        
        '''
        Neural Network Architecture
        '''
        noise_shape = (100,)
        
        model = Sequential()
        model.add(LeakyReLU(alpha = 0.2,input_shape = noise_shape))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(BatchNormalization(momentum = 0.8))
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(BatchNormalization(momentum = 0.8))
    
        model.add(Dense(np.prod(self.img_shape),activation = 'tanh'))
        
        model.add(Reshape(self.img_shape))
        
        print ('Generator Shape')
        
        model.summary()
        
        noise = Input(shape = noise_shape)
        
        img = model(noise)
        
        return Model(noise,img)
    
    def train(self,epochs,batch_size,save_interval,check):
        
        '''Loading the MNIST dataset'''
        
        if check == 0:
            (X_train, _),(_,_) = mnist.load_data()
            np.save('xtrain.npy',X_train)
        else:
            X_train = np.load('xtrain.npy')
            
        
        '''
        Rescale image
        '''
        
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        #np.expand_dims(X_train, axis = 3)
        
        half_batch = int(batch_size/2)
        
        for epoch in range(epochs):
            '''
            Train Dsicriminator
            
            '''
            idx = np.random.randint(0,X_train.shape[0],half_batch)
            imgs = X_train[idx]
            
            noise = np.random.normal(0,1,(half_batch,100))
            
            '''generate the half batch of images'''
            
            gen_imgs = self.generator.predict(noise)
            
            print (len(gen_imgs))
            
            '''
            Training the Discriminator
            '''
            loss_real = self.discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
            loss_fake = self.discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
            
            print (loss_real)
            
            d_loss = 0.5*np.add(loss_real,loss_fake)
            
            '''
            Train Generator
            '''
            
            g_loss = self.combined_model.train_on_batch(noise,np.ones((half_batch,1)))
            
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            if epoch%save_interval==0:
                self.sample_images(epoch)
                
    def sample_images(self,epoch):
        r,c = 5,5
        noise = np.random.randint(0,60000,(r*c,100))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        
            
                
        
        
        
        
    
    
if __name__=='__main__':
    check = sys.argv[1]
    gan = GAN()
    check = int(check)
    gan.train(32000,32,500,check)
    
    
    
        
        
        
        