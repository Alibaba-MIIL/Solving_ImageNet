# Solving ImageNet: a Unified Scheme for Training any Backbone to Top Results

Official PyTorch Implementation
<br> [Paper](tbd) |
> Tal Ridnik, Hussam Lawen, Emanuel Ben-Baruch, Asaf Noy<br/> DAMO Academy, Alibaba
> Group

**Abstract**

ImageNet serves as the primary dataset for evaluating the quality of computer-vision models. The common practice today is training each architecture with a tailor-made scheme, designed and tuned by an expert.
In this paper, we present a unified scheme for training any backbone on ImageNet. The scheme, named USI (Unified Scheme for ImageNet), is based on knowledge distillation and modern tricks.  It requires no adjustments or hyper-parameters tuning between different models, and is efficient in terms of training times.
We test USI on a wide variety of architectures, including CNNs, Transformers, Mobile-oriented and MLP-only. On all models tested, USI outperforms previous state-of-the-art results. Hence, we are able to transform training on ImageNet from an expert-oriented task to an automatic seamless routine.
Since USI accepts any backbone and trains it to top results, it also enables to perform methodical comparisons, and identify the most efficient backbones along the speed-accuracy Pareto curve.

<p align="center">
 <table class="tg">
   <tr>
    <td class="tg-c3ow"><img src="./pics/pic1.png" align="center" width="600""></td>
  </tr>
</table>
</p>

## How to Training on ImageNet with USI scheme
USI scheme does not require hyper-parameter tuning. The base training configuration works well for any backbone.
All the results presented in the paper are fully reproducible.

An example code - training ResNet50 model with USI:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py  \
/mnt/Imagenet/
--model=resnet50
--kd_model_name=tresnet_l_v2
```

Some additional degrees of freedom that might be usefull:

- Adjusting the batch size (defualt - 128): ```--batch-size=...```
- Choosing a different teacher (default - tresnet_l_v2): ```--kd_model_name=...```
- Choosing pretrain weights for the teacher (if none are given, we are using the default pretrain weights): ```--kd_model_path=...```

## Acknowledgements

The training code is based on the excellent [timm repository](https://github.com/rwightman/pytorch-image-models).

## Citation
TBD
