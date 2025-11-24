# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import torch.nn as nn

# # YOLO model
# yolo_model = YOLO("yolov8n.pt")

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ResNet classifier
# num_classes = 13
# resnet_model = models.resnet18(weights=None)
# resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

# resnet_model.load_state_dict(torch.load("art_style_resnet18.pth", map_location=device))
# resnet_model.to(device)
# resnet_model.eval()


# classes=['Academic_Art','Art_Nouveau','Baroque',
# 'Expressionism','Japanese_Art','Neoclassicism','Primitivism',
# 'Realism','Renaissance','Rococo','Romanticism','Symbolism',
# 'Western_Medieval', 'Cubism', 'Fauvism', 'Impressionism','Surrealism']
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# def predict_style(img):
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = resnet_model(img_tensor)
#         prob = torch.softmax(out, dim=1)[0]
#     idx = prob.argmax().item()
#     return classes[idx], float(prob[idx])

# # Streamlit UI
# st.title("Art Style Detector & Classifier")

# uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded:
#     img = Image.open(uploaded).convert("RGB")
#     st.image(img, caption="Uploaded image", use_container_width=True)

#     results = yolo_model(img)
    
#     if len(results[0].boxes) == 0:
#         style, conf = predict_style(img)
#         st.write(f"**Detected style:** {style} ({conf:.2f})")
#         st.image(img, caption=f"{style} ({conf:.2f})")
#     else:

#         for det in results[0].boxes:
#             x1, y1, x2, y2 = map(float, det.xyxy[0])
#             crop = img.crop((x1, y1, x2, y2))
#             style, conf = predict_style(crop)

#             st.write(f"**Detected style:** {style} ({conf:.2f})")
#             st.image(crop, caption=f"{style} ({conf:.2f})")




import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Academic_Art','Art_Nouveau','Baroque','Expressionism','Japanese_Art',
           'Neoclassicism','Primitivism','Realism','Renaissance','Rococo',
           'Romanticism','Symbolism','Western_Medieval']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()        
])

@st.cache_resource
def load_style_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load("art_style_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

style_model = load_style_model()

@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8x.pt") 
    except:
        return None

yolo = load_yolo()

def predict_style(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = style_model(x)
        probs = F.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()

st.title("Art Style Detector")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])


if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Завантажене зображення", use_container_width=True)

    target_crop = None

    if yolo:
        results = yolo(img, conf=0.25, iou=0.6, verbose=False)[0]
        for box in results.boxes:
            label = results.names[int(box.cls.item())]
            conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if label in ["painting", "picture", "tv", "mirror"]:
                crop = img.crop((x1, y1, x2, y2))
                # картина має займати хоча б 5% площі
                if crop.width * crop.height > img.width * img.height * 0.05:
                    target_crop = crop
                    break  

    if target_crop:
        st.image(target_crop, caption="Знайдено картину", use_container_width=True)
        final_img = target_crop
    else:
        final_img = img

    with st.spinner("Визначаю стиль..."):
        style, confidence = predict_style(final_img)

    st.success(f"**Стиль: {style}**")
    st.metric("Достовірність", f"{confidence:.1%}")

    with st.expander("Показати топ-3 можливі стилі"):
        x = transform(final_img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(style_model(x), dim=1)[0]
        top3 = torch.topk(probs, 3)
        for i in range(3):
            s = classes[top3.indices[i]]
            c = top3.values[i].item()
            st.write(f"{i+1}. **{s}** – {c:.1%}")