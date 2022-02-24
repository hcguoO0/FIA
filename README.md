# Feature Importance-aware Attack(FIA)

This repository contains the code for the paper: 

**[Feature Importance-aware Transferable Adversarial Attacks](https://arxiv.org/pdf/2107.14185.pdf)  (ICCV 2021)**

## Requirements

- Python 3.6.8
- Keras 2.2.4
- Tensorflow 1.14.0
- Numpy 1.16.2
- Pillow 6.0.0
- Scipy 1.2.1

## Experiments

#### Introduction

- `attack.py` : the implementation for different attacks.

- `verify.py` : the code for evaluating generated adversarial examples on different models.

  You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim,  https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./models_tf`.

#### Example Usage

##### Generate adversarial examples:

- FIA

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv3/conv3_3/Relu --ens 30 --probb 0.7 --output_dir ./adv/FIA/
```

- PIM:

```
python attack.py --model_name vgg_16 --attack_method PIM --amplification_factor 10 --gamma 1 --Pkern_size 3 --output_dir ./adv/PIM/
```

- FIA+PIDIM

```
python attack.py --model_name vgg_16 --attack_method FIAPIM --layer_name vgg_16/conv3/conv3_3/Relu --ens 30 --probb 0.7 --amplification_factor 2.5 --gamma 0.5 --Pkern_size 3 --image_size 224 --image_resize 250 --prob 0.7 --output_dir ./adv/FIAPIDIM/
```

Different attack methods have different parameter setting, and the detailed setting can be found in our paper.

##### Evaluate the attack success rate

```
python verify.py --ori_path ./dataset/images/ --adv_path ./adv/FIA/ --output_file ./log.csv
```

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{wang2021feature,
  title={Feature importance-aware transferable adversarial attacks},
  author={Wang, Zhibo and Guo, Hengchang and Zhang, Zhifei and Liu, Wenxin and Qin, Zhan and Ren, Kui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7639--7648},
  year={2021}
  }
```
