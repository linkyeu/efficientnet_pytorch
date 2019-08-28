# EfficientNet (PyTorch)

My implementation of [__Efficient-Net__](https://arxiv.org/abs/1905.11946) in `nn.Sequence` manner (i.e. without `nn.Module`) for feature extraction layers. This approach is more convinient for applying  [Class Activation Mapping)](http://gradcam.cloudcv.org/) or working with [fast.ai](https://docs.fast.ai/) library.

__To load pre-trained model simply run:__
```
import efficientnet
model = efficientnet.efficientnet(net="B4", pretrained=True)
```

__For features extraction simply run:__
```
import efficientnet
image = torch.randn(1, 3, 300, 300)
model = efficientnet.efficientnet(net="B4", pretrained=True)
features = model.features(image)
```
In same way you can get output from any layer.



