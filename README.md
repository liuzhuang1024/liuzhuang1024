### <div align="center">工作<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/npsFeel5.png"/>整理</div>

----
## 公式识别代码
  + [im2latex](https://github.com/guillaumegenthial/im2latex)
  + [math-formula-recognition](https://github.com/jungomi/math-formula-recognition)
  + **[Pytorch-Handwritten-Mathematical-Expression-Recognition](https://github.com/whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition)**
  + [BTTR](https://github.com/Green-Wood/BTTR)
  + [Master-Ocr]()
  + [Master-Table]()
---
## 注意力机制
  + [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
  - [Image-Local-Attention](https://github.com/zzd1992/Image-Local-Attention)

---
## 常用网站
  - CHROME
    + [CHROME](https://www.isical.ac.in/~crohme/index.html)
    + [CHRAOME竞赛地址](http://www.iapr-tc11.org/mediawiki/index.php/CROHME:_Competition_on_Recognition_of_Online_Handwritten_Mathematical_Expressions)
  - [Datasets per Topic](http://tc11.cvc.uab.es/datasets/type/)
  - [OCR 大全](https://github.com/WenmuZhou/OCR_DataSet)
----
## DataSet
  - [graviti](https://gas.graviti.cn/open-datasets)
  - [超神经](https://hyper.ai/datasets)
----
## 论文
  + [Multi-Scale Attention with Dense Encoder for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/1801.03530.pdf)
  + [Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition](http://staff.ustc.edu.cn/~jundu/Publications/publications/PR17-1.pdf)
  + [MASTER: Multi-Aspect Non-local Network forScene Text Recognition](https://arxiv.org/pdf/1910.02562.pdf "master-ocr")
  + [Center-Loss](https://ydwen.github.io/papers/WenECCV16.pdf)
  + [Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer](https://arxiv.org/abs/2105.02412)
----

## 文章
  + [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)
  + [基于Seq2Seq的公式识别引擎](https://zhuanlan.zhihu.com/p/183182208)

----
## 目标检测
  + [ATSS 目标检测采样策略](https://github.com/sfzhang15/ATSS)
  + [WeightedBoxesFusion 目标检测集成方法](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

---
## MarkDown
  + [MarkDown-常用操作](https://x-pp.github.io/2019/04/18/markdown%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C/)
  + [拓展](https://blog.csdn.net/m0_37925202/article/details/80461714)

---

## 字符串处理相关
  + unicodedata
  + string
  + string.translate
  + re

---

## 格式处理相关
  + pprint
  + rich

---

## 模型训练trick
  + 数据增广
  + 梯度正则
  + 学习率调整方法
  + 权重正则
  + 多卡训练
  + 注意力机制
  + 激活函数
  + Center Loss
  + [SWA](https://github.com/timgaripov/swa)
    - [PyTorch 1.6 now includes Stochastic Weight Averaging](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)
    - [SWALR](https://arxiv.org/abs/1803.05407)
  + 自集成
  

---
## 冠军解决方案
  + [ICDAR2021-公式检测](https://github.com/Yuxiang1995/ICDAR2021_MFD)

---
## Reading
  + [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)

---
## TODO
  - [ ] 阅读master源码
  - [ ] 阅读codeWAP源码
  - [ ] 阅读Attention源码
  - [x] ~~实现Center Loss~~
  - [ ] 阅读Transformer源码并理解
  - [x] ~~[度量学习]()~~
  - [x] ~~阅读[Focal CTC Loss](https://downloads.hindawi.com/journals/complexity/2019/9345861.pdf)~~
  - [x] ~~Center Loss~~ 
    + Code
      - [pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)
      - [crnn-ctc-centerloss](https://github.com/tommyMessi/crnn_ctc-centerloss)
    + 文章
      - [辅助理解 1](https://blog.csdn.net/fxwfxw7037681/article/details/114440117)
      - [辅助理解 2](https://blog.csdn.net/jacke121/article/details/90480434?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control)
  - [ ] [图像处理](https://legacy.imagemagick.org/Usage/distorts/#shepards?tdsourcetag=s_pcqq_aiomsg)
    

----
## [Other]()
  + [行人识重 deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid "using pytorch-center-loss")
  + [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) 

----

## Pytorch
  + [自定义操作torch.autograd.Function](https://zhuanlan.zhihu.com/p/344802526)
  + [pytorch-loss](https://github.com/CoinCheung/pytorch-loss)

---

## 高效工具
  + Ray
  + wandb
  + Time
  + MarkDown
  + Git
    - [commit之后，想撤销commit](https://blog.csdn.net/w958796636/article/details/53611133)
      - git checkout
      - git reset [--mixed [--soft [--hard ]]]
      - git stash
      - git stage

---

## 数学
  - [Animation engine for explanatory math videos](https://github.com/3b1b/manim)
    - [3b1b](https://space.bilibili.com/88461692/)

---

## 蒸馏
  - [ICLR 2021 | SEED：自监督蒸馏学习，显著提升小模型性能！](https://www.aminer.cn/research_report/607965a1e409f29eb73e2e97)
    - [SEED: Self-supervised Distillation For Visual Representation](https://arxiv.org/abs/2101.04731)
  - [Three mysteries in deep learning: Ensemble, knowledge distillation, and self-distillation](https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/)

---
## 图像检索
  - [autofaiss](https://github.com/criteo/autofaiss)
  - [faiss]()

## OCR
  - [sightseq](https://github.com/zhiqwang/sightseq)
    - [lightseq](https://github.com/bytedance/lightseq)
  - [mmocr](https://github.com/open-mmlab/mmocr)
  - [DAVAR-Lab-OCR](https://github.com/hikopensource/DAVAR-Lab-OCR)

## 加速
  - [contiguous_pytorch_params](https://github.com/PhilJd/contiguous_pytorch_params)