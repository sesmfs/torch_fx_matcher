from matcher import Matcher
from torchvision.models import resnet
import torch

# step1. Create a resnet18 model by the torchvision.
model = resnet.resnet18(True).eval()
input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    original_output = model(input)

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
print(matcher.lexer.graph)

# step4. Fuse the Conv2d and the BatchNormalization layer based on traced model.
def fuse_bn_replacement(matcher, isubgraph, subgraph):
    conv = subgraph[0]
    bn   = subgraph[1]
    fused_conv = Matcher._fuse_conv_bn(matcher.get_module(conv), matcher.get_module(bn))
    Matcher._replace_node_module(conv, matcher.modules, fused_conv)
    bn.replace_all_uses_with(conv)
    matcher.traced.graph.erase_node(bn)

matcher.replace(fuse_bn_replacement)

# pip install tabular  # if there is an error about the tabular
# matcher.traced.graph.print_tabular()

with torch.no_grad():
    fused_output = matcher.traced(input)

diff = (original_output - fused_output).abs()
print(f"Absolute difference: max={diff.max():.5f}, sum={diff.sum():.5f}")
