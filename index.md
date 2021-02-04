## [Weakly Supervised Visual Semantic Parsing](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zareian_Weakly_Supervised_Visual_Semantic_Parsing_CVPR_2020_paper.pdf)

[Alireza Zareian](https://www.linkedin.com/in/az2407/), [Svebor Karaman](http://www.sveborkaraman.com/), [Shih-Fu Chang](https://www.ee.columbia.edu/~sfchang/)

[Digital Video and Multimedia](https://www.ee.columbia.edu/ln/dvmm/)

Columbia University

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
