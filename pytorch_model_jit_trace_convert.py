import torch
import torchvision.models as models

# Step 1: Load the pretrained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Set the model to evaluation mode
resnet50.eval()

# Step 2: Prepare an example input
# Create an example input tensor with the shape that ResNet-50 expects: [batch_size, 3, 224, 224]
example_input = torch.randn(1, 3, 224, 224)

# Step 3: Trace the model
traced_model = torch.jit.trace(resnet50, example_input)

# Step 4: Save the traced model
traced_model.save("resnet50_traced_model.pt")

print("Model traced and saved successfully as 'resnet50_traced_model.pt'")