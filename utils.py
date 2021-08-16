# coding: utf-8
import os
import numpy as np
from scipy.misc import imresize,imread,imsave
from PIL import Image
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, resnet_v1, vgg, nets_factory

slim = tf.contrib.slim


def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]

def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2

def inv_vgg_normalization(image):
  return np.clip(image + [123.68, 116.78, 103.94],0,255)

def inv_inception_normalization(image):
  return np.clip((image + 1.0) * 0.5 * 255,0,255)

normalization_fn_map = {
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': inception_normalization,
    'resnet_v2_101': inception_normalization,
    'resnet_v2_152': inception_normalization,
    'resnet_v2_200': inception_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
}

inv_normalization_fn_map = {
    'inception_v1': inv_inception_normalization,
    'inception_v2': inv_inception_normalization,
    'inception_v3': inv_inception_normalization,
    'inception_v4': inv_inception_normalization,
    'inception_resnet_v2': inv_inception_normalization,
    'resnet_v1_50': inv_vgg_normalization,
    'resnet_v1_101': inv_vgg_normalization,
    'resnet_v1_152': inv_vgg_normalization,
    'resnet_v1_200': inv_vgg_normalization,
    'resnet_v2_50': inv_inception_normalization,
    'resnet_v2_101': inv_inception_normalization,
    'resnet_v2_152': inv_inception_normalization,
    'resnet_v2_200': inv_inception_normalization,
    'vgg_16': inv_vgg_normalization,
    'vgg_19': inv_vgg_normalization,
}

offset = {
    'inception_v1': 1,
    'inception_v2': 1,
    'inception_v3': 1,
    'inception_v4': 1,
    'inception_resnet_v2': 1,
    'resnet_v1_50': 0,
    'resnet_v1_101': 0,
    'resnet_v1_152': 0,
    'resnet_v1_200': 0,
    'resnet_v2_50': 1,
    'resnet_v2_101': 1,
    'resnet_v2_152': 1,
    'resnet_v2_200': 1,
    'vgg_16': 0,
    'vgg_19': 0,
  }

image_size={
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'inception_v4': 299,
    'inception_resnet_v2': 299,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg_16': 224,
    'vgg_19': 224,
  }
base_path='./models_tf'

checkpoint_paths = {
    'inception_v1': None,
    'inception_v2': None,
    'inception_v3': base_path+'/inception_v3.ckpt',
    'inception_v4': base_path+'/inception_v4.ckpt',
    'inception_resnet_v2': base_path+'/inception_resnet_v2_2016_08_30.ckpt',
    'resnet_v1_50': base_path+'/resnet_v1_50.ckpt',
    'resnet_v1_101': None,
    'resnet_v1_152': base_path+'/resnet_v1_152.ckpt',
    'resnet_v1_200': None,
    'resnet_v2_50': base_path+'/resnet_v2_50/resnet_v2_50.ckpt',
    'resnet_v2_101': None,
    'resnet_v2_152': base_path+'/resnet_v2_152/resnet_v2_152.ckpt',
    'resnet_v2_200': None,
    'vgg_16': base_path+'/vgg_16.ckpt',
    'vgg_19': base_path+'/vgg_19.ckpt',
    'adv_inception_v3':base_path+'/adv_inception_v3/adv_inception_v3.ckpt',
    'adv_inception_resnet_v2':base_path+'/adv_inception_resnet_v2/adv_inception_resnet_v2.ckpt',
    'ens3_adv_inception_v3':base_path+'/ens3_adv_inception_v3/ens3_adv_inception_v3.ckpt',
    'ens4_adv_inception_v3':base_path+'/ens4_adv_inception_v3/ens4_adv_inception_v3.ckpt',
    'ens_adv_inception_resnet_v2':base_path+'/ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt'
  }

ground_truth=None
with open('./labels.txt') as f:
    ground_truth=f.read().split('\n')[:-1]

def load_image(image_path, image_size, batch_size):
    images = []
    filenames=[]
    labels=[]
    idx=0

    files=os.listdir(image_path)
    files.sort(key=lambda x: int(x[:-4]))
    for i,filename in enumerate(files):
        # image = imread(image_path + filename)
        # image = imresize(image, (image_size, image_size)).astype(np.float)
        image=Image.open(image_path + filename)
        image=image.resize((image_size,image_size))
        image=np.array(image)
        images.append(image)
        filenames.append(filename)

        labels.append(int(ground_truth[i]))
        idx+=1
        if idx==batch_size:
            yield np.array(images),np.array(filenames),np.array(labels)
            idx=0
            images=[]
            filenames=[]
            labels=[]
    if idx>0:
        yield np.array(images), np.array(filenames),np.array(labels)

def save_image(images,names,output_dir):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        # imsave(output_dir+name,images[i].astype('uint8'))
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


if __name__=='__main__':
    pass


