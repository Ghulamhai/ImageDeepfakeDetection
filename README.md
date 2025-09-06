# Deepfake Detection with ViXNet and EfficientNet-B3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Framework](https://img.shields.io/badge/Framework-PyTorch_&_Flask-orange.svg)

This project provides an efficient and powerful deepfake detection system based on the **ViXNet** architecture. This implementation replaces the original Xception backbone with a more lightweight **EfficientNet-B3**, creating a model optimized for performance on resource-constrained devices.

The model is trained on the **FaceForensics++ (FF++)** dataset and validated for generalization on the **Celeb-DF (CeDF)** dataset. A clean, modern web application built with Flask allows for easy, real-time inference.

![Deepfake Detection Web App](https://raw.githubusercontent.com/Ghulam-Haider/Deepfake-Detection-with-ViXNet-and-EfficientNet-B3/main/assets/webapp_1.png)
![Deepfake Detection Web App](https://raw.githubusercontent.com/Ghulam-Haider/Deepfake-Detection-with-ViXNet-and-EfficientNet-B3/main/assets/webapp_2.png)

---

## Key Features

* **Advanced Dual-Branch Architecture**: Combines a patch-based Vision Transformer (ViT-B_16) for global context and an EfficientNet-B3 for local spatial feature extraction.
* **High-Performance Detection**: Achieves **~94% accuracy** on the FF++ dataset and demonstrates strong generalization with **~70% accuracy** on the challenging Celeb-DF dataset.
* **Computationally Efficient**: Requires only **~1.8B FLOPs** for inference, a significant reduction from the 8.4B FLOPs of the original Xception-based model.
* **Interactive Web Interface**: A simple and responsive web app to upload a face image and receive an instant **Real** or **Fake** prediction with a confidence score.

---

## 🛠 Installation and Setup

### 1. Prerequisites

* Python 3.8 or higher
* Git
* NVIDIA GPU (Recommended for faster inference)

### 2. Clone the Repository


# PASTE YOUR GITHUB REPO URL HERE
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]

3. Set Up a Virtual Environment
On macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
On Windows:

Bash

python -m venv venv
.\venv\Scripts\activate
4. Install Dependencies
Bash

pip install -r requirements.txt
5. Download the Pre-trained Model
The trained model file (best_combined_model.pth) is hosted externally due to its size.

Download from: PASTE YOUR MODEL'S GOOGLE DRIVE OR OTHER DOWNLOAD LINK HERE

Place the downloaded best_combined_model.pth file in the root directory of the project.

 Usage
Running the Web Application
The web application provides a simple interface for testing the model. The workflow is as follows:
 
![Workflow](https://raw.githubusercontent.com/Ghulam-Haider/Deepfake-Detection-with-ViXNet-and-EfficientNet-B3/main/assets/workflow.png)
Start the Flask server from the project's root directory:

Bash

python app.py
Open your web browser and navigate to http://127.0.0.1:4300.

Upload a JPEG or PNG image containing a face.

The model will return a prediction (Real or Fake) along with a confidence score.

Training the Model
The training script is provided in the notebook_for_model.py file. To retrain or fine-tune the model:

Download Datasets: Obtain the FaceForensics++ and Celeb-DF datasets from their official sources (see Datasets section).

Update Paths: Modify the notebook to point to the correct paths for your downloaded datasets.

Run Training: Execute the cells in the notebook. Note: Training is a computationally intensive process and requires a powerful NVIDIA GPU.

 Model Architecture
This model adapts the ViXNet architecture by integrating EfficientNet-B3 for enhanced efficiency and generalization.



![Model Architecture](https://raw.githubusercontent.com/Ghulam-Haider/Deepfake-Detection-with-ViXNet-and-EfficientNet-B3/main/assets/architecture)
Component	Details
Core Architecture	Adapted from ViXNet, replacing Xception with EfficientNet-B3.
Global Branch (ViT)	A Vision Transformer (ViT-B_16) with 86M parameters to capture global relationships between image patches.
Spatial Branch (CNN)	An EfficientNet-B3 with 12M parameters to extract global spatial features using MBConv and Squeeze-and-Excitation (SE) modules.
Total Parameters	~98 Million
Input	300x300 RGB images of faces, preprocessed with MTCNN.
Output	A single sigmoid output. A score ≥ 0.5 is classified as Real, and < 0.5 is classified as Fake.
Training Details	Dataset: FaceForensics++, Epochs: 50, Optimizer: Adam, Learning Rate: 1e-4, Loss: Binary Cross-Entropy.

Export to Sheets
 Performance
Intra-Dataset Evaluation (on FaceForensics++)
Metric	Score
Accuracy	93–94%
AUC	97–98%
F1-Score	92–93%

Export to Sheets
Cross-Dataset Evaluation (on Celeb-DF)
Metric	Score
Accuracy	65–70%
AUC	70–75%
F1-Score	60–65%

Export to Sheets
📂 Repository Structure
deepfake-detection-vixnet/
│
├── app.py
├── best_combined_model.pth
├── requirements.txt
├── notebook_for_model.py
├── README.md
│
└── templates/
    └── index.html
🗂️ Datasets
The datasets are not included in this repository due to their large size. Please download them from the official sources:

FaceForensics++ (FF++): Used for training. Official Website

Celeb-DF (CeDF): Used for validation. Official Website



📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
The original ViXNet authors for their innovative architecture.

The creators of the FaceForensics++ and Celeb-DF datasets.

The open-source community for libraries like PyTorch, Flask, and Facenet-PyTorch.

 Contact
For any questions or feedback, please reach out to Ghulam Haider at haiderghulam1998@gmail.com
