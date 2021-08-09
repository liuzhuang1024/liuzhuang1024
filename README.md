**<div align="center">工作<img src="https://csdnimg.cn/release/blogv2/dist/pc/img/npsFeel5.png"/>整理</div>**

----
## 公式识别
  + [im2latex](https://github.com/guillaumegenthial/im2latex)
  + [math-formula-recognition](https://github.com/jungomi/math-formula-recognition)
  + **[Pytorch-Handwritten-Mathematical-Expression-Recognition](https://github.com/whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition)**
  + [BTTR](https://github.com/Green-Wood/BTTR)
  + [Master-Ocr]()
  + [Master-Table]()
  + [文章]()
    + [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)
    + [基于Seq2Seq的公式识别引擎](https://zhuanlan.zhihu.com/p/183182208)
  + 论文
    + [Multi-Scale Attention with Dense Encoder for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/1801.03530.pdf)
    + [Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition](http://staff.ustc.edu.cn/~jundu/Publications/publications/PR17-1.pdf)
    + [MASTER: Multi-Aspect Non-local Network forScene Text Recognition](https://arxiv.org/pdf/1910.02562.pdf "master-ocr")
    + [Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer](https://arxiv.org/abs/2105.02412)
---
## 注意力机制
  - [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
  - [Image-Local-Attention](https://github.com/zzd1992/Image-Local-Attention)

---
## 常用数据集网站
  - CHROME
    + [CHROME](https://www.isical.ac.in/~crohme/index.html)
    + [CHRAOME竞赛地址](http://www.iapr-tc11.org/mediawiki/index.php/CROHME:_Competition_on_Recognition_of_Online_Handwritten_Mathematical_Expressions)
  - [Datasets per Topic](http://tc11.cvc.uab.es/datasets/type/)
  - [OCR Dataset 大全](https://github.com/WenmuZhou/OCR_DataSet)
----
## DataSet
  - [graviti](https://gas.graviti.cn/open-datasets)
  - [超神经](https://hyper.ai/datasets)
----
## 目标检测
  + [ATSS 目标检测采样策略](https://github.com/sfzhang15/ATSS)
  + [WeightedBoxesFusion 目标检测集成方法](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
  + [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  + [YOLOV5](https://github.com/ultralytics/yolov5)

---
## MarkDown
  + [MarkDown-常用操作](https://x-pp.github.io/2019/04/18/markdown%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C/)
  + [拓展](https://blog.csdn.net/m0_37925202/article/details/80461714)
  + [Markdown快速入门](https://sspai.com/post/45816)
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
  + [python-tabulate](https://github.com/astanin/python-tabulate)

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
  - [ ] tldtr
  - [ ] 阅读master源码
  - [ ] 阅读codeWAP源码
  - [ ] 阅读Attention源码
  - [x] 实现Center Loss
  - [ ] 实现ACE
  - [x] 实现SWA
  - [x] 实现Pytorch AMP
  - [ ] 从头实现CRNN，实现Encoder-Decoder
  - [ ] 自己实现一套训练框架
  - [ ] 阅读Transformer源码并理解
  - [x] [度量学习]()
  - [x] 阅读[Focal CTC Loss](https://downloads.hindawi.com/journals/complexity/2019/9345861.pdf)
  - [x] Center Loss
    + Code
      - [pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)
      - [crnn-ctc-centerloss](https://github.com/tommyMessi/crnn_ctc-centerloss)
    + 论文
      + [Center-Loss](https://ydwen.github.io/papers/WenECCV16.pdf)
    + 文章
      - [辅助理解 1](https://blog.csdn.net/fxwfxw7037681/article/details/114440117)
      - [辅助理解 2](https://blog.csdn.net/jacke121/article/details/90480434?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control)
  - [ ] [图像处理](https://legacy.imagemagick.org/Usage/distorts/#shepards?tdsourcetag=s_pcqq_aiomsg)
    

----
## [Other]()
  + [行人识重 deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid "using pytorch-center-loss")
  + [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) 
  + [DexiNed 边缘检测](https://github.com/xavysp/DexiNed)

----

## Pytorch
  + [自定义操作torch.autograd.Function](https://zhuanlan.zhihu.com/p/344802526)
  + [pytorch-loss](https://github.com/CoinCheung/pytorch-loss)
  + BatchNorm
    + 很重要，也很有效
    + [Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs](https://arxiv.org/abs/2003.00152)
  + 模型加速
    + [contiguous_pytorch_params](https://github.com/PhilJd/contiguous_pytorch_params)
    + [DDP]()
    + [Pytorch amp/apex](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
      + [多精度训练教程 非官网](https://mp.weixin.qq.com/s/HKWsM6iDNCmDsu7aDeDwbw)
  + 无监督
    + [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch '半监督')
    + [Dino]('无监督')
  + Pytorch 入门
    + [code-of-learn-deep-learning-with-pytorch](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch)

---

## 可视化
  - [Tools-to-Design-or-Visualize-Architecture-of-Neural-Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)

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

## Ubuntu 
  - [zsh](https://ohmyz.sh/#install)
  - [fzf](https://github.com/junegunn/fzf)
  - [pip](https://www.runoob.com/w3cnote/pip-cn-mirror.html)
  - usermod / passwd 修改默认目录
  - chmod / chown 修改文件权限
  - gpustat
---
## 数学
  - [Animation engine for explanatory math videos](https://github.com/3b1b/manim)
    - [3b1b](https://space.bilibili.com/88461692/)

---

## LeetCode
  - [LeetCode 题库](https://leetcode-cn.com/problemset/all/)
---
## 蒸馏、弱监督、半监督、自学习、无监督
  - [ICLR 2021 | SEED：自监督蒸馏学习，显著提升小模型性能！](https://www.aminer.cn/research_report/607965a1e409f29eb73e2e97)
    - [SEED: Self-supervised Distillation For Visual Representation](https://arxiv.org/abs/2101.04731)
  - [Three mysteries in deep learning: Ensemble, knowledge distillation, and self-distillation](https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/)
  - [pseudo-labeling-to-deal-with-small-datasets-what-why-how](https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af)
  - **[R-Drop](https://github.com/dropreg/R-Drop)**
  - [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels)
  - [Sequence-to-Sequence Contrastive Learning for Text Recognition
](https://arxiv.org/abs/2012.10873)

---
## 数据可视化
  - [an-introduction-to-t-sne-with-python-example](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

---
## 图像检索
  - [autofaiss](https://github.com/criteo/autofaiss)
  - [faiss]()

---
## **OCR**
  - [sightseq](https://github.com/zhiqwang/sightseq)
    - [lightseq](https://github.com/bytedance/lightseq)
  - [mmocr](https://github.com/open-mmlab/mmocr)
  - [DAVAR-Lab-OCR](https://github.com/hikopensource/DAVAR-Lab-OCR)
    - ACE LOSS
      - [ACE loss](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_recognition/ace)
      - [Aggregation-Cross-Entropy](https://github.com/summerlvsong/Aggregation-Cross-Entropy)
    - [rflearning](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_recognition/rflearning 'Reciprocal Feature Learning via Explicit andImplicit Tasks in Scene Text Recognition')
  - [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
  - [vedastr](https://github.com/Media-Smart/vedastr)
  - [KISS: Keeping it Simple for Scene Text Recognition.](https://github.com/Bartzi/kiss)
  - **[mtl-text-recognition](https://github.com/bityigoss/mtl-text-recognition)**
  - [textocr facebook](https://textvqa.org/?p=/textocr)
  - 弱监督
    - [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels)
    - [Sequence-to-Sequence Contrastive Learning for Text Recognition
](https://arxiv.org/abs/2012.10873)
  - TODO
    - [ ] [SAR-Strong-Baseline-for-Text-Recognition](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition)
    - [ ] [SEED](https://github.com/Pay20Y/SEED 'SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition From Qiao Zhi')
  - 数据集
    - [OCR Dataset 大全](https://github.com/WenmuZhou/OCR_DataSet)
  - [实战]()
    - [WenmuZhou PytorchOCR (包含部署模块)](https://github.com/WenmuZhou/PytorchOCR)
    - [BADBADBADBOY pytorchOCR (包含center-loss以及剪枝)](https://github.com/BADBADBADBOY/pytorchOCR)
    - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
      - [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)
      - [RapidOCR](https://github.com/RapidAI/RapidOCR)
  - [文章]()
    - [一文读懂CRNN+CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)
---
## C++
  - [folly](https://github.com/facebook/folly)
  - [入门](https://segmentfault.com/a/1190000037727214)

---

## Python
  - help
    - [code example](https://www.programcreek.com/python/)
---
## OpenCV
  - [编译OpenCV](https://www.jianshu.com/p/8fd19e45e01b) 

---
## 编译/运行库相关
  - ldd
  - cmake
  - make

---
## 论文-Search
  - arxiv-vanity
  - arxiv
  - [sci-hub](https://www.sci-hub.shop/)
  - [connectedpapers](https://www.connectedpapers.com/)

---

## Github
  - [fastgit](hub.fastgit.org)
  - Github520

---
## 部署
- [tensorrtx 众多应用仓库](https://github.com/wang-xinyu/tensorrtx)
- [Savior](https://github.com/novioleo/Savior)
- [triton]()
  - [server](https://github.com/triton-inference-server/server)


---
## 自建云盘
  - [自建云盘教程](https://zhuanlan.zhihu.com/p/44103820)
  - [win 局域网共享](https://zhuanlan.zhihu.com/p/83983289)
  - [smb://]()
---
## 工作
  - [张大妈](https://hizdm.cn/city/beijing/)