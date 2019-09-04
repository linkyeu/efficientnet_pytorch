# EfficientNet (PyTorch)

My implementation of [__Efficient-Net__](https://arxiv.org/abs/1905.11946) in `nn.Sequence` manner (i.e. without `nn.Module`) for feature extraction layers. This approach is more convinient for applying  [Class Activation Mapping](http://gradcam.cloudcv.org/) or working with [fast.ai](https://docs.fast.ai/) library.

## About EfficientNet

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

### To load pre-trained model simply run
```
import efficientnet
model = efficientnet.efficientnet(net="B4", pretrained=True)
```
where `B4` could be replaced with any model scale from B0 to B7. Weights will be downloaded automatically.

### For features extraction simply run
```
import efficientnet
image = torch.randn(1, 3, 300, 300)
model = efficientnet.efficientnet(net="B4", pretrained=True)
features = model.features(image)
```
In same way you can get output from any layer.</br>

Weights were copied from [here](https://github.com/lukemelas/EfficientNet-PyTorch) and adopted for my implementation.



