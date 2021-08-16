import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names=['inception_v3','inception_v4','inception_resnet_v2','resnet_v1_50','resnet_v1_152',
             'resnet_v2_50','resnet_v2_152','vgg_16','vgg_19','adv_inception_v3','adv_inception_resnet_v2',
             'ens3_adv_inception_v3','ens4_adv_inception_v3','ens_adv_inception_resnet_v2']

def verify(model_name,ori_image_path,adv_image_path):

    checkpoint_path=utils.checkpoint_paths[model_name]

    if model_name=='adv_inception_v3' or model_name=='ens3_adv_inception_v3' or model_name=='ens4_adv_inception_v3':
        model_name='inception_v3'
    elif model_name=='adv_inception_resnet_v2' or model_name=='ens_adv_inception_resnet_v2':
        model_name='inception_resnet_v2'

    num_classes=1000+utils.offset[model_name]

    network_fn = utils.nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes),
        is_training=False)

    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    image_size = utils.image_size[model_name]

    batch_size=200
    image_ph=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])

    logits, _ = network_fn(image_ph)
    predictions = tf.argmax(logits, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)

        ori_pre=[] # prediction for original images
        adv_pre=[] # prediction label for adversarial images
        ground_truth=[] # grund truth for original images

        for images,names,labels in utils.load_image(ori_image_path, image_size, batch_size):
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            ground_truth.extend(labels)
            ori_pre.extend(pres)

        for images,names,labels in utils.load_image(adv_image_path, image_size, batch_size):
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            adv_pre.extend(pres)
    tf.reset_default_graph()

    ori_pre=np.array(ori_pre)
    adv_pre=np.array(adv_pre)
    ground_truth=np.array(ground_truth)

    if num_classes==1000:
        ground_truth=ground_truth-1

    return ori_pre,adv_pre,ground_truth


def main(ori_path='./dataset/images/',adv_path='./adv/',output_file='./log.csv'):
    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]
    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(model_names)
        for model_name in model_names:
            print(model_name)
            ori_pre,adv_pre,ground_truth=verify(model_name,ori_path,adv_path)
            ori_accuracy = np.sum(ori_pre == ground_truth)/1000
            adv_accuracy = np.sum(adv_pre == ground_truth)/1000
            adv_successrate = np.sum(ori_pre != adv_pre)/1000
            adv_successrate2 = np.sum(ground_truth != adv_pre) / 1000
            print('ori_acc:{:.1%}/adv_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(ori_accuracy,adv_accuracy,adv_successrate,adv_successrate2))
            ori_accuracys.append('{:.1%}'.format(ori_accuracy))
            adv_accuracys.append('{:.1%}'.format(adv_accuracy))
            adv_successrates.append('{:.1%}'.format(adv_successrate))
        # print(adv_successrates)
        # writer.writerow(ori_accuracys)
        writer.writerow(adv_successrates)
        # writer.writerow(adv_accuracys)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./adv/FIA/')
    parser.add_argument('--output_file', default='./log.csv')
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file)
