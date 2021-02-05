## Weakly Supervised Visual Semantic Parsing

[Alireza Zareian](https://www.linkedin.com/in/az2407/), &nbsp; &nbsp; &nbsp; &nbsp; 
[Svebor Karaman](http://www.sveborkaraman.com/), &nbsp; &nbsp; &nbsp; &nbsp; 
[Shih-Fu Chang](https://www.ee.columbia.edu/~sfchang/)

[Digital Video and Multimedia](https://www.ee.columbia.edu/ln/dvmm/)<br/>
Columbia University

A machine learning framework for faster and more powerful image understanding with less supervision cost.

Published and presented as an oral paper at CVPR 2020 (Conference on Computer Vision and Pattern Recognition).

<a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zareian_Weakly_Supervised_Visual_Semantic_Parsing_CVPR_2020_paper.pdf"><button>Read Our Paper</button></a>
&nbsp; &nbsp; &nbsp; &nbsp; 
<a href="https://www.youtube.com/watch?v=IDOlnZvY5vY"><button>Watch The Talk</button></a>

### Abstract

Scene Graph Generation (SGG) aims to extract entities, predicates and their semantic structure from images, enabling deep understanding of visual content, with many applications such as visual reasoning and image retrieval. Nevertheless, existing SGG methods require millions of manually annotated bounding boxes for training, and are computationally inefficient, as they exhaustively process all pairs of object proposals to detect predicates. In this paper, we address those two limitations by first proposing a generalized formulation of SGG, namely Visual Semantic Parsing, which disentangles entity and predicate recognition, and enables sub-quadratic performance. Then we propose the Visual Semantic Parsing Network, VSPNet, based on a dynamic, attention-based, bipartite message passing framework that jointly infers graph nodes and edges through an iterative process. Additionally, we propose the first graph-based weakly supervised learning framework, based on a novel graph alignment algorithm, which enables training without bounding box annotations. Through extensive experiments, we show that VSPNet outperforms weakly supervised baselines significantly and approaches fully supervised performance, while being several times faster. We publicly release the source code of our method.

### Citation:
```
@InProceedings{Zareian_2020_CVPR,
author = {Zareian, Alireza and Karaman, Svebor and Chang, Shih-Fu},
title = {Weakly Supervised Visual Semantic Parsing},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Quick Overview:

![Method](vspnet-method.png)

Given an input image and object proposals, a scene graph is produced by an iterative process involving a multi-headed attention module that infers edges between entities and predicates, and a novel message passing module to propagate information between nodes and update their states. To define a classification loss for each node and edge, the ground truth graph is aligned to our output graph through a novel weakly supervised algorithm. Red represents mistake.

### Oral Talk:

[<img src="https://img.youtube.com/vi/IDOlnZvY5vY/maxresdefault.jpg" width="75%">](https://www.youtube.com/watch?v=IDOlnZvY5vY)


