import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define class names (sesuaikan dengan dataset Anda)
CLASS_NAMES = ['Parang', 'Kawung', 'Mega Mendung', 'Truntum', 'Sido Mukti']  # TODO: Ganti dengan nama class asli
NUM_CLASSES = len(CLASS_NAMES)

# Load VGG16 model architecture
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)  # Adjust final layer
model.load_state_dict(torch.load('vgg16_batik_best.pth', map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Model loaded successfully on {device}")

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    """Predict batik motif from uploaded image."""
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Return formatted result
    result = f"**Motif Batik:** {predicted_class}\n\n**Confidence:** {confidence_score:.2f}%"
    return result

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Gambar Batik"),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="🎨 Klasifikasi Motif Batik Indonesia",
    description="Upload gambar batik untuk mengidentifikasi motifnya menggunakan VGG16 Deep Learning",
    examples=[],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)