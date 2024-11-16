U-SCIFNet: Infrared Small Target Detection through Improved Skip Connections in U-Net
====

Algorithm Introduction
----

We propose a U-shaped network with Spatial-channel Interaction Fusion (U-SCIFNet) for accurate single-frame infrared small target detection. The experimental results on two public datasets (NUAA-SIRST,NUST-SIRST) demonstrate the effectiveness of our method. The contribution of this paper are as follows:

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
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install einops
```
You may also need to install other packages, if you encounter a package missing error, you just need to install it using the pip command.

Dataset Structure
---

If you want to train on your own dataset, you need to prepare the data in the following structure:
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

The first step is to modify the settings in the parser object in train.py, where you can select the dataset, specify the number of training epochs, and configure other settings such as image patch size and batch size.

Run command:
```angular2html
python train.py
```
The results, including log files, other logs, model weights, etc., are stored in the paths specified by the save and log_dir in the parser.

Testing
---

The initial step involves changing the settings in the parser object in test.py, where you need to specify the paths for loading the weight file and saving the test results.
Run command:
```angular2html
python test.py
```

Performance
----

| Dataset         | mIoU (%) | nIoU (%) | Pd (%)|  Fa (10^-6)|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| NUDT-SIRST    |  94.05  |  94.31   | 98.35  | 2.34  |
| SIRST3    | 82.21  |  82.44 | 98.23 | 9.78 | 

Please note：The trained weight files are located in the __weights folder__ and are available for download as a reference.

*This code is highly borrowed from [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

*This code is highly borrowed from [ABC](https://github.com/PANPEIWEN/ABC). Thanks to Peiwen Pan.

*The overall repository style is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to Shuai Yuan.

 Referrence
 ---
 1. Dai, Y., Wu, Y., Zhou, F., & Barnard, K. (2021). Asymmetric contextual modulation for infrared small target detection. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 950-959).[[code]](https://github.com/YimianDai/open-acm)

 2. Li, B., Xiao, C., Wang, L., Wang, Y., Lin, Z., Li, M., ... & Guo, Y. (2022). Dense nested attention network for infrared small target detection. IEEE Transactions on Image Processing, 32, 1745-1758.[[code]](https://github.com/YeRen123455/Infrared-Small-Target-Detection)

 3. Wu, X., Hong, D., & Chanussot, J. (2022). UIU-Net: U-Net in U-Net for infrared small object detection. IEEE Transactions on Image Processing, 32, 364-376.[[code]](https://github.com/danfenghong/IEEE_TIP_UIU-Net)

 4. Wang, H., Cao, P., Yang, J., & Zaiane, O. (2024). Narrowing the semantic gaps in u-net with learnable skip connections: The case of medical image segmentation. Neural Networks, 178, 106546.[[code]](https://github.com/McGregorWwww/UDTransNet)

 5. Yuan, S., Qin, H., Yan, X., Akhtar, N., & Mian, A. (2024). Sctransnet: Spatial-channel cross transformer network for infrared small target detection. IEEE Transactions on Geoscience and Remote Sensing.[[code]](https://github.com/xdFai/SCTransNet)

 6. Xu, S., Zheng, S., Xu, W., Xu, R., Wang, C., Zhang, J., ... & Guo, L. (2024). HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection. arXiv preprint arXiv:2403.10778.[[code]](https://github.com/zhengshuchen/HCFNet)

 7. Pan, P., Wang, H., Wang, C., & Nie, C. (2023, July). ABC: Attention with bilinear correlation for infrared small target detection. In 2023 IEEE International Conference on Multimedia and Expo (ICME) (pp. 2381-2386). IEEE.[[code]](https://github.com/PANPEIWEN/ABC)

 8. Zang, D., Su, W., Zhang, B., & Liu, H. (2025). DCANet: Dense Convolutional Attention Network for infrared small target detection. Measurement, 240, 115595.[[code]](https://github.com/Tianzishu/DCANet)


















