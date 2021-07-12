# Benchmark of deep architectures for segmentation of medical images

The repository contains the source code used for evaluation of UNet-based [1] deep neural network models used for medical image segmentation with use of nnUnet framework [2]. Following architectures were reimplemented and compared with basic version of Unet: UNet++ [3], UNet3+ [4], ResUnet [5], CPFNet [6] and CS2-Net [7].

## Data
All datasets used for comparison are avilable in public domain:
* [task 1](https://data.mendeley.com/datasets/zm6bxzhmfz)
* [task 2](https://data.mendeley.com/datasets/6x684vg2bg)
* [task 3](https://www.kaggle.com/krzysztofrzecki/bone-marrow-oedema-data)
* [task 4, 5, 6](http://medicaldecathlon.com/)

## How to run
0. Make sure to have [nnUnet](https://github.com/MIC-DKFZ/nnUNet) installed and configured properly.
1. Code for each model is run separately and training/prediction scripts are provided in adequate directories.
2. Change workdir and other paths in run_training.py script.
3. Run training with:
```bash
python3 run_training.py TASK_NAME_OR_ID FOLD_NO
```
4. Run predictions with:
```bash
python3 predict_simple.py -i path/to/test/images -o path/to/results/directory -tr trainerName -m 2d -p nnUNetPlansv2.1 -t taskName -chk model_best
```

## References
1. Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28
2. Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z'
3. Zhou Z., Rahman Siddiquee M.M., Tajbakhsh N., Liang J. (2018) UNet++: A Nested U-Net Architecture for Medical Image Segmentation. In: Stoyanov D. et al. (eds) Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. DLMIA 2018, ML-CDS 2018. Lecture Notes in Computer Science, vol 11045. Springer, Cham. https://doi.org/10.1007/978-3-030-00889-5_1
4. Huang, Huimin & Lin, Lanfen & Tong, Ruofeng & Hu, Hongjie & Qiaowei, Zhang & Iwamoto, Yutaro & Han, Xian-Hua & Chen, Yen-Wei & Wu, Jian. (2020). UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 1055-1059. 10.1109/ICASSP40776.2020.9053405
5. Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu, ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data,
ISPRS Journal of Photogrammetry and Remote Sensing, Volume 162, 2020, Pages 94-114, https://doi.org/10.1016/j.isprsjprs.2020.01.013
6. S. Feng et al., "CPFNet: Context Pyramid Fusion Network for Medical Image Segmentation," in IEEE Transactions on Medical Imaging, vol. 39, no. 10, pp. 3008-3018, Oct. 2020, doi: 10.1109/TMI.2020.2983721.
7. Lei Mou, Yitian Zhao, Huazhu Fu, Yonghuai Liu, Jun Cheng, Yalin Zheng, Pan Su, Jianlong Yang, Li Chen, Alejandro F. Frangi, Masahiro Akiba, Jiang Liu, CS2-Net: Deep learning segmentation of curvilinear structures in medical imaging, Medical Image Analysis, Volume 67, 2021, 101874,https://doi.org/10.1016/j.media.2020.101874
