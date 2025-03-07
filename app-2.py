import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class names
class_names = ['COVID', 'Lung_Opacity', 'No_Tumor', 'Normal', 'Tumor', 'Viral_Pneumonia']

# Load the model
def load_model(model_path, num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model_path = 'C:/Users/Pmhat/OneDrive/Desktop/Prajot-ka-project-2--main/transfer_balanced_learning_model.pth'
num_classes = len(class_names)
model = load_model(model_path, num_classes)

# Function to make predictions
def predict(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    input_batch = image.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]

    return {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}, predicted_label

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=len(class_names)),
        gr.Label(label="Predicted Class")
    ],
    title="Medical Image Classification",
    description="Upload a medical image to classify it into one of the following categories: COVID, Lung Opacity, No Tumor, Normal, Tumor, or Viral Pneumonia."
)

# Launch the interface
iface.launch(share=True)