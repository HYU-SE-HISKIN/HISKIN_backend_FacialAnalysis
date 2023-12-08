from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from vit_pytorch import ViT
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ViT model architecture
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=3,  # Replace with the actual number of classes
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Load the pre-trained weights
state_dict = torch.load("best_model.pth")

# Load model architecture first
model.load_state_dict(state_dict['model'], strict=False)  # Set strict to False to ignore missing keys

# Load the filtered state_dict into the model
model.load_state_dict(state_dict['model'], strict=False)

model.eval()

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define label map (replace this with your actual label map)
label_map = {0: 'acne', 1: 'bags', 2: 'redness'}  # Replace with your actual labels and indices

def preprocess_image(image):
    # Open the image and apply transformations
    img = Image.open(image).convert("RGB")
    img = image_transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)

@app.route('/', methods=['GET'])
def index():
    return "Hello, this is the index page!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Preprocess the uploaded image
        input_image = preprocess_image(file)

        # Perform inference
        with torch.no_grad():
            output = model(input_image)

        # Get predicted class
        predicted_class = torch.argmax(torch.softmax(output, dim=1)).item()
        
        # Map class index to class label
        predicted_label = label_map.get(predicted_class, 'Unknown')

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

