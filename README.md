# VSPNet

Code for the CVPR 2020 oral paper: [Weakly Supervised Visual Semantic Parsing](http://openaccess.thecvf.com/content_CVPR_2020/html/Zareian_Weakly_Supervised_Visual_Semantic_Parsing_CVPR_2020_paper.html)

Watch the talk:

[<img src="https://img.youtube.com/vi/IDOlnZvY5vY/maxresdefault.jpg" width="75%">](https://www.youtube.com/watch?v=IDOlnZvY5vY)

```
@InProceedings{Zareian_2020_CVPR,
author = {Zareian, Alireza and Karaman, Svebor and Chang, Shih-Fu},
title = {Weakly Supervised Visual Semantic Parsing},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

This package can reproduce numbers reported in Table 1 and Table 2 of the paper.

In order to use the code, configure python and cuda, and install the packages in requirements.txt. 

There are also some preprocessed files such as scene graphs and object proposals. You can download them from the following three links, and place the content in the folders ./data, ./metadata, and ./checkpoints respectively. Checkpoints are only required if you want to evaluate pretrained models without training from scratch.
* [data](https://www.dropbox.com/sh/eb60553z4md36x2/AACOM9jvJFyHRDcuuGEzHY98a?dl=0)
* [metadata](https://www.dropbox.com/sh/oa8u7qolfpf1op0/AACivQp5RmtmykbqmWeupOZEa?dl=0)
* [checkpoints](https://www.dropbox.com/sh/1qud8usl0xyoybe/AAAm6_CcdL46I2TfywWTk_jma?dl=0)

Note that one of the files which contains proposal features is very large (185GB). If you cannot download that file, you can still run all experiments except 0052 and 0054.

Each group of numbers in the table are created using a jupyter notebook. Each jupyter notebook has 3 steps, each containing a set of code blocks. The first step loads the data and builds the model. The second step trains the model, and the third step evaluates the model. You can skip the second step to only evaluate the downloaded checkpoints. The notebooks already contain the numbers we have got. 

Here is a list of notebooks, with a pointer to the numbers each produce:
```
ipynb/experiments/0061.ipynb --- table 1 row 3 (ablation 1)
ipynb/experiments/0059.ipynb --- table 1 row 4 (ablation 2)
ipynb/experiments/0063.ipynb --- table 1 row 5 (ablation 3)
ipynb/experiments/0065.ipynb --- table 1 row 6 (ablation 4)
ipynb/experiments/0058.ipynb --- table 1 row 7 (ablation 5)
ipynb/experiments/0055.ipynb --- table 1 row 8 (WS VSPNet)
ipynb/experiments/0054.ipynb --- table 1 row 11 (FS VSPNet)

ipynb/experiments/0052.ipynb --- table 2 row 6 (FS VSPNet) columns 4-5 (SGGen)
ipynb/experiments/0064.ipynb --- table 2 row 6 (FS VSPNet) columns 6-7 (SGCls)
ipynb/experiments/0056.ipynb --- table 2 row 6 (FS VSPNet) columns 6-7 (PredCls)
ipynb/experiments/0053.ipynb --- table 2 row 7 (WS VSPNet) columns 4-5 (SGGen)
ipynb/experiments/0057.ipynb --- table 2 row 7 (WS VSPNet) columns 6-9 (SGCls and PredCls)
```

The evaluation code will print several outputs. The ones we report on the paper are:
```
PerImageLocPredMR_at100_iou_0.5 (R@100 PredCls)
PerImageLocPredMR_at50_iou_0.5 (R@50 PredCls)
PerImageOneHopMP_phrase_at100_iou_0.5 (R@100 PhrDet)
PerImageOneHopMP_phrase_at50_iou_0.5 (R@50 PhrDet)
PerImageOneHopMR_at100_iou_0.5 (R@100 SGGen or SGCls depending if the experiment loads xfeat_proposals or xfeat_gtbox)
PerImageOneHopMR_at50_iou_0.5 (R@50 SGGen or SGCls depending if the experiment loads xfeat_proposals or xfeat_gtbox)
```

Created and maintained by [Alireza Zareian](https://www.linkedin.com/in/az2407/) at [DVMM](http://www.ee.columbia.edu/ln/dvmm/) - Columbia University.
