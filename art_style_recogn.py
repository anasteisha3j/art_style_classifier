import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# YOLO model
yolo_model = YOLO("yolov8n.pt")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet classifier
num_classes = 13
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

resnet_model.load_state_dict(torch.load("art_style_resnet18.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()


classes=['Academic_Art','Art_Nouveau','Baroque',
'Expressionism','Japanese_Art','Neoclassicism','Primitivism',
'Realism','Renaissance','Rococo','Romanticism','Symbolism',
'Western_Medieval', 'Cubism', 'Fauvism', 'Impressionism','Surrealism']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_style(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = resnet_model(img_tensor)
        prob = torch.softmax(out, dim=1)[0]
    idx = prob.argmax().item()
    return classes[idx], float(prob[idx])

# Streamlit UI
st.title("Art Style Detector & Classifier")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    results = yolo_model(img)
    
    if len(results[0].boxes) == 0:
        style, conf = predict_style(img)
        st.write(f"**Detected style:** {style} ({conf:.2f})")
        st.image(img, caption=f"{style} ({conf:.2f})")
    else:

        for det in results[0].boxes:
            x1, y1, x2, y2 = map(float, det.xyxy[0])
            crop = img.crop((x1, y1, x2, y2))
            style, conf = predict_style(crop)

            st.write(f"**Detected style:** {style} ({conf:.2f})")
            st.image(crop, caption=f"{style} ({conf:.2f})")
