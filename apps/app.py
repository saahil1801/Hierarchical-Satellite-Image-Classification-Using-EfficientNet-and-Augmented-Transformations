import streamlit as st
import torch
from PIL import Image
import requests
from torchvision import transforms
import matplotlib.pyplot as plt
from super_gradients.training import models
import pathlib
import textwrap
from pathlib import Path, PurePath

# Define the configuration
class config:
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_DIR = 'data/train'
    CHECKPOINT_PATH = 'checkpoints/0_Baseline_Experiment/RUN_20240627_191223_220175/ckpt_best.pth'  # Update this to the absolute path

# Load the best model
best_full_model = models.get('efficientnet_b0',
                             num_classes=5,  # Adjust the number of classes
                             checkpoint_path=config.CHECKPOINT_PATH)
best_full_model.eval()
best_full_model.to(config.DEVICE)

# Transformation function
transform = transforms.Compose([
    transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])

# Class names (replace with your actual class names)
class_names = sorted([d.name for d in Path(config.TRAIN_DIR).iterdir() if d.is_dir()])

def predict_image(image, model, transform, class_names):
    transformed_image = transform(image).unsqueeze(0).to(config.DEVICE)
    with torch.inference_mode():
        output = model(transformed_image)
    probs = torch.softmax(output, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    return class_names[pred_label], probs[0][pred_label].item()

# Streamlit UI
st.title("Yoga Pose Prediction")
st.write("Upload an image of a yoga pose to get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    label, probability = predict_image(image, best_full_model, transform, class_names)
    
    st.write(f"Predicted Pose: {label}")
    st.write(f"Probability: {probability:.3f}")

    # Plot the image with prediction
    fig, ax = plt.subplots()
    ax.imshow(image)
    title = f"Pred: {label} | Prob: {probability:.3f}"
    plt.title("\n".join(textwrap.wrap(title, width=20)))
    plt.axis("off")
    st.pyplot(fig)

# To run this app, save it as `app.py` and execute `streamlit run app.py` in your terminal
