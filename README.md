U-SCIFNet: Infrared Small Target Detection through Improved Skip Connections in U-Net
====

Algorithm Introduction
----

本文提出了一个空间通道交互融合的U形网络(U-SCIFNet)来实现准确的单帧红外小目标检测。在两种公共数据集(NUAA-SIRST，NUST-SIRST)上的证明了我们方法的有效性。本文的贡献如下：

  1. Propose a feature fusion strategy for the skip connections in U-Net.
 
  2. Establish a method for multi-scale fusion of global and local spatial features.
  
  3. Build high-level semantic associations through cross-channel information interaction.
 
  4. The detection performance for infrared small targets outperforms other advanced methods.

![image](https://github.com/privary/U-SCIFNet/blob/main/overall%20structure.png)

Installation
----

```angular2html
# Python == 3.8
# Cuda == 11.3
# RTX 3090(24GB)

conda create -n py38 python=3.8
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install einops
```
You may also need to install other packages, if you encounter a package missing error, you just need to install it using the pip command.

数据集结构
---

如果你想要在自己的数据集上训练，你需要按照下列的结构准备数据:
```
  ├──./datasets/
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── SIRST3
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST3.txt
  │    │    │    ├── test_SIRST3.txt
```
Training
---

第一步是在train.py的parser对象中更改设置，可选择数据集，指定训练轮次，其他所有配置包括图像切块大小，批量大小等都在其中。
运行指令:
```angular2html
python train.py
```
The results including log files, otherlogs files, model weights, etc., 位于parser中指定的save和log_dir的存放路径。

Testing
---

第一步是在test.py的parser对象中更改设置，需要指定权重文件的加载路径以及测验结果的保存路径。
运行指令：
```angular2html
python test.py
```

Performance
----

| Dataset         | mIoU (%) | nIoU (%) | Pd (%)|  Fa (10^-6)|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| NUDT-SIRST    |  94.05  |  94.31   | 98.35  | 2.34  |
| SIRST3    | 82.21  |  82.44 | 98.23 | 9.78 | 

*This code is highly borrowed from [ABC](https://github.com/PANPEIWEN/ABC). Thanks to Peiwen Pan.


















