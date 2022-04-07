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
    <td class="tg-c3ow"><img src="./pics/pic1.png" align="center" width="300""></td>
    <td class="tg-c3ow"><img src="./pics/pic3.png" align="center" width="300""></td>


  </tr>
</table>
</p>

## Acknowledgements

The training code is based on the excellent [timm repository](https://github.com/rwightman/pytorch-image-models).
