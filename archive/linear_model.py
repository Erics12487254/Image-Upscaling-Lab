import torch

# Four inputs produce one score.
layer = torch.nn.Linear(4, 1)

# Set known weights so the output matches the hand calculation.
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1.0, 0.0, 0.5, -0.5]]))
    layer.bias.copy_(torch.tensor([0.1]))

# One image represented by its four flattened pixel values.
image = torch.tensor([[0.0, 1.0, 0.5, 0.5]])
score = layer(image)

print(score)  # tensor([[0.1000]])
