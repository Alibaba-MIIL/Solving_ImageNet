#  Model Zoo

With USI, we can reliably identify models that provide [good speed-accuracy trade-off](Speed_Accuracy_Comparisons.md).

For those top models, we provide here weights from large-scale pretraining on ImageNet-21K. We recommended using the large-scale weights for transfer learning - they almost always provide superior results on transfer, compared to 1K weights.

| Backbone  |  21K Single-label Pretraining weights | 21K Multi-label Pretraining weights |  ImageNet-1K Accurcy [\%] |
| :------------: | :--------------: | :--------------: | :--------------: |
**TResNet-L** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/tresnet_l_v2/single_label_ls.pth) | [Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/tresnet_l_v2/multi_label_ls.pth) | [83.9](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/USI/tresnet_l_v2_83_9.pth) |
**TResNet-M** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/tresnet_m/single_label_ls.pth) | [Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/tresnet_m/multi_label_ls.pth) | 82.5 |
**ResNet50** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/resnet50/single_label_ls.pth) | [Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/resnet50/multi_label_ls.pth) | 81.0 |
**MobileNetV3_Large_100** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/mobilenetv3_large_100/single_label_ls.pth) | N/A | 77.3 |
**LeViT-384** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/levit_384/single_label_ls.pth) | [Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/levit_384/multi_label_ls.pth) | 82.7 |
**LeViT-768** |[Link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/levit_768/single_label_ls.pth) | N/A  | [84.2](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/unified/levit_768/levit_768_84_2.pth) |
**[EdgeNeXt-S](https://arxiv.org/abs/2206.10589)** |N/A | N/A | [81.1](https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.1/edgenext_small_usi.pth) |

