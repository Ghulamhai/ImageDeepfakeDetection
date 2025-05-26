Deepfake Detection with ViXNet and EfficientNet-B3
This project implements a deepfake detection model based on the ViXNet architecture, adapted to use EfficientNet-B3 instead of Xception for improved efficiency. The model is trained on the FaceForensics++ (FF++) dataset and validated on the Celeb-DF (CeDF) dataset to detect whether face images are real or fake. A Flask-based web application allows users to upload images and receive predictions with confidence scores.
Features

Dual-Branch Architecture: Combines patch-based attention with a Vision Transformer (ViT-B_16) and EfficientNet-B3 for local and global feature extraction.
High Performance: Achieves ~93–94% accuracy on FF++ and ~65–70% on CeDF.
Efficient: Uses ~1.8B FLOPs (vs. Xception’s 8.4B), suitable for resource-constrained devices.
Web Interface: Upload images via a modern, responsive website to get instant Real/Fake predictions.

Repository Structure
deepfake-detection-vixnet/
├── app.py              # Flask web application
├── templates/
│   └── index.html      # Frontend HTML for the website
├──  best_combined_model.pth           # Trained model 
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── notebook_for_model.py            # Training script

Installation
Prerequisites

Python 3.8+
Git
NVIDIA GPU (optional, for faster inference)

Steps

Clone the Repository:
git clone https://github.com/your-username/deepfake-detection-vixnet.git
cd deepfake-detection-vixnet


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the Model:

The trained model (model.pth) is hosted externally due to size.
Download from: Google Drive Link (replace with your link).
Place model.pth in the project root.



Usage
Running the Web Application

Start the Flask server:python app.py


Open a browser and navigate to http://localhost:4300.
Upload a face image (JPEG/PNG) via the website’s upload area.
View the prediction (Real or Fake) and confidence score.

Training the Model

The model was trained on FF++ and validated on CeDF. To retrain:
Download FF++ and CeDF datasets (see Datasets section).
Update train.py with paths to your dataset.
Run:python train.py


Note: Training requires significant computational resources (e.g.: NVIDIA GPU).

Model Details

Architecture: Adapted from ViXNet:
Patch-Based Attention: Splits images into 16x16 patches, uses 3x3 convolution for attention masks.
Vision Transformer (ViT-B_16): 86M parameters, captures global patch relationships.
EfficientNet-B3: 12M parameters, extracts global spatial features with MBConv and SE modules.
Classification Head: Dense layers (512→256 → 128 →1) with sigmoid output.


Parameters: ~98M total (86M ViT + 12M EfficientNet-B3).
Input: 300x300 RGB face images, preprocessed with MTCNN face detection.
Output: Binary classification(Real ≥ 0.5, Fake < 0.5) with confidence score.
Training:
Dataset: FF++ (~1,000 real, ~4,000 fake videos).
Epochs: 50, Learning Rate: 0.0001, Optimizer: Adam, Loss: Binary Cross-Entropy.
Hardware: NVIDIA Tesla T4 GPU.



Results

Intra-Dataset (FF++):
Accuracy: 93–94%
AUC: 97–98%
F1-Score: 92–93%
Slightly below ViXNet (95.92% accuracy) due to EfficientNet-B3’s smaller capacity.


Cross-Dataset (CeDF):
Accuracy: 65–70%
AUC: 70–75%
F1-Score: 60–65%
Outperforms ViXNet (61.19% accuracy) due to SE modules enhancing generalizability.



Datasets

FaceForensics++ (FF++):
Source: Official Website
Size: ~1,000 real, ~4,000 fake videos (DeepFakes, FaceSwap, etc.).
Use: Training and intra-dataset testing.


Celeb-DF (CeDF):
Source: Official Website
Size: ~590 real, ~5,639 fake videos.
Use: Cross-dataset validation.


Note: Datasets are not included due to size. Download from the official sources.

Contributing
Contributions are welcome! To contribute:


Please follow the Code of Conduct (to be added).
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

ViXNet authors for their architecture and methodology.
Creators of FF++ and CeDF datasets.
Open-source libraries: PyTorch, TensorFlow, Flask, EfficientNet-PyTorch, Facenet-PyTorch.

Contact
For questions or feedback, contact Ghulam Haider at haiderghulam1998@gmail.com.
