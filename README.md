# torch_fx_matcher
An easy pattern match tool for the torch.fx.

# Demonstration
```python
from matcher import Matcher
from torchvision.models import resnet

# step1. Create a resnet18 model by the torchvision.
model = resnet.resnet18(True).eval()

# step2. Trace the model using the torch.fx
matcher = Matcher().trace(model)

# step3. Match the traced model using the custom pattern.
matcher.match(
    """
        Conv2d(?, a)
        BatchNorm2d(a, ?)
    """
)
matcher.print_matchs()

# step4. Fuse the Conv2d and the BatchNormalization layer based on traced model.
def fuse_bn_replacement(matcher, isubgraph, subgraph):
    conv = subgraph[0]
    bn   = subgraph[1]
    fused_conv = Matcher._fuse_conv_bn(matcher.get_module(conv), matcher.get_module(bn))
    Matcher._replace_node_module(conv, matcher.modules, fused_conv)
    bn.replace_all_uses_with(conv)
    matcher.traced.graph.erase_node(bn)

matcher.replace(fuse_bn_replacement)
```

- Run demo:
```bash
$> python main.py
```

# Subsraph Rules
```python
layername1/layername2([input_argument1, input_argument2], [output_argument1, output_argument2])
layername(input_argument, output_argument)

where:
   ? will match any layer or argument.

For example1:
    """
    Conv(?, c0)
    Sigmoid(c0, s0)
    Mul([s0, c0], ?)
    """

For example2:
    """
    Conv/Avgpool(?, c0)
    ?(c0, s0)
    Mul([s0, c0], ?)
    """
```

# Reference
- No reference