from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Load TensorFlow/Keras models
inception_model = load_model('best_inceptionv3_model.keras')
vgg19_model = load_model('vgg19_best_model.keras')

# PyTorch Model architecture
class FoodClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
pytorch_model = FoodClassifierCNN(num_classes).to(device)
pytorch_model.load_state_dict(torch.load("best_model (1).pth", map_location=device))
pytorch_model.eval()

# Class names
class_names = [
    'Dhokla', 'Poha', 'Rice Bhakri', 'Upma', 'Bhaji',
    'Dosa', 'Idli', 'Medu Vada', 'Roti', 'Samosa'
]

# PyTorch image preprocessing
pytorch_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# HTML template with dual model results
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Food Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 1000px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 15px;
            padding: 40px 20px;
            margin: 30px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #f8f9fa;
        }
        .upload-area:hover {
            border-color: #2980b9;
            background: #e3f2fd;
            transform: translateY(-2px);
        }
        .upload-icon {
            font-size: 48px;
            color: #3498db;
            margin-bottom: 20px;
        }
        .upload-text {
            color: #7f8c8d;
            font-size: 20px;
            margin-bottom: 15px;
        }
        .file-input {
            display: none;
        }
        .image-preview {
    max-width: 100%;
    max-height: 300px;
    border-radius: 10px;
    margin: 20px auto; /* Center horizontally */
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    display: none;
}

        .results-container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        .model-results {
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border: 2px solid #e9ecef;
            display: none;
        }
        .model-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }
        .inception-results {
            border-color: #e74c3c;
        }
        .inception-results .model-title {
            border-bottom-color: #e74c3c;
            color: #e74c3c;
        }
        .vgg19-results {
            border-color: #27ae60;
        }
        .vgg19-results .model-title {
            border-bottom-color: #27ae60;
            color: #27ae60;
        }
        .pytorch-results {
            border-color: #f39c12;
        }
        .pytorch-results .model-title {
            border-bottom-color: #f39c12;
            color: #f39c12;
        }
        .prediction {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .confidence {
            font-size: 18px;
            color: #7f8c8d;
        }
        .loading {
            display: none;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            color: #e74c3c;
            background: #fdf2f2;
            border: 1px solid #fecaca;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        .status {
            background: #d5f4e6;
            color: #27ae60;
            border: 1px solid #a3d9a5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        @media (max-width: 1024px) {
            .results-container {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Food Classifier</h1>
        <div class="status">
             Models loaded and ready!
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click to select a food image</div>
            <input type="file" id="imageInput" class="file-input" accept="image/*">
        </div>

        <img id="imagePreview" class="image-preview" alt="Preview">
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Classifying image with all three models...</p>
        </div>

        <div class="error" id="error"></div>
        
        <div class="results-container">
            <div class="model-results inception-results" id="inceptionResults">
                <div class="model-title">InceptionV3 Model</div>
                <div class="prediction" id="inceptionPrediction"></div>
                <div class="confidence" id="inceptionConfidence"></div>
            </div>
            
            <div class="model-results vgg19-results" id="vgg19Results">
                <div class="model-title">VGG19 Model</div>
                <div class="prediction" id="vgg19Prediction"></div>
                <div class="confidence" id="vgg19Confidence"></div>
            </div>
            
            <div class="model-results pytorch-results" id="pytorchResults">
                <div class="model-title">Our Model</div>
                <div class="prediction" id="pytorchPrediction"></div>
                <div class="confidence" id="pytorchConfidence"></div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        const inceptionResults = document.getElementById('inceptionResults');
        const vgg19Results = document.getElementById('vgg19Results');
        const pytorchResults = document.getElementById('pytorchResults');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');

        uploadArea.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageSelect);

        function handleImageSelect(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    hideError();
                    hideResults();
                    classifyImage(e.target.result);
                };
                reader.readAsDataURL(file);
            }
        }

        async function classifyImage(imageSrc) {
            try {
                showLoading();
                hideError();
                hideResults();

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageSrc
                    })
                });

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                }

            } catch (err) {
                showError('Error classifying image: ' + err.message);
            } finally {
                hideLoading();
            }
        }

        function showResults(data) {
            // InceptionV3 results
            document.getElementById('inceptionPrediction').textContent = `Predicted: ${data.inception.prediction}`;
            document.getElementById('inceptionConfidence').textContent = `Confidence: ${data.inception.confidence}%`;
            
            // VGG19 results
            document.getElementById('vgg19Prediction').textContent = `Predicted: ${data.vgg19.prediction}`;
            document.getElementById('vgg19Confidence').textContent = `Confidence: ${data.vgg19.confidence}%`;
            
            // PyTorch results
            document.getElementById('pytorchPrediction').textContent = `Predicted: ${data.pytorch.prediction}`;
            document.getElementById('pytorchConfidence').textContent = `Confidence: ${data.pytorch.confidence}%`;
            
            inceptionResults.style.display = 'block';
            vgg19Results.style.display = 'block';
            pytorchResults.style.display = 'block';
        }

        function hideResults() {
            inceptionResults.style.display = 'none';
            vgg19Results.style.display = 'none';
            pytorchResults.style.display = 'none';
        }

        function showLoading() {
            loading.style.display = 'block';
        }

        function hideLoading() {
            loading.style.display = 'none';
        }

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }

        function hideError() {
            error.style.display = 'none';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        image_data = data['image']
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # === KERAS MODELS PREDICTION ===
        # Resize to 224x224 for both Keras models (InceptionV3 and VGG19)
        keras_img = img.resize((224, 224))
        keras_img_array = np.array(keras_img)
        keras_img_array = keras_img_array.astype('float32') / 255.0
        keras_img_array = np.expand_dims(keras_img_array, axis=0)
        
        # Predict with InceptionV3 model
        inception_predictions = inception_model.predict(keras_img_array, verbose=0)
        inception_predicted_class_idx = np.argmax(inception_predictions[0])
        inception_confidence = float(inception_predictions[0][inception_predicted_class_idx]) * 100
        
        # Predict with VGG19 model
        vgg19_predictions = vgg19_model.predict(keras_img_array, verbose=0)
        vgg19_predicted_class_idx = np.argmax(vgg19_predictions[0])
        vgg19_confidence = float(vgg19_predictions[0][vgg19_predicted_class_idx]) * 100
        
        # === PYTORCH MODEL PREDICTION ===
        # Preprocess for PyTorch model
        pytorch_img_tensor = pytorch_transform(img).unsqueeze(0).to(device)
        
        # Predict with PyTorch model
        with torch.no_grad():
            pytorch_outputs = pytorch_model(pytorch_img_tensor)
            pytorch_probabilities = torch.nn.functional.softmax(pytorch_outputs[0], dim=0)
            pytorch_confidence, pytorch_predicted = torch.max(pytorch_probabilities, 0)
        
        return jsonify({
            'inception': {
                'prediction': class_names[inception_predicted_class_idx],
                'confidence': f"{inception_confidence:.2f}"
            },
            'vgg19': {
                'prediction': class_names[vgg19_predicted_class_idx],
                'confidence': f"{vgg19_confidence:.2f}"
            },
            'pytorch': {
                'prediction': class_names[pytorch_predicted.item()],
                'confidence': f"{pytorch_confidence.item() * 100:.2f}"
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Loading models...")
    print("✅ InceptionV3 model loaded successfully!")
    print("✅ VGG19 model loaded successfully!")
    print("✅ PyTorch CNN model loaded successfully!")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)