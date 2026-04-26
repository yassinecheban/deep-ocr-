# OCR Deep Learning Project

A complete Optical Character Recognition (OCR) system for printed text using deep learning. Built with TensorFlow/Keras and deployed as a web application.

## 🎯 Features

- **CRNN Architecture**: Combines CNN feature extraction with BiLSTM sequence modeling
- **CTC Loss**: End-to-end training without character-level alignment
- **66 Character Classes**: Digits (0-9), uppercase (A-Z), lowercase (a-z), and punctuation
- **Web Interface**: Real-time character recognition with drawing canvas and image upload
- **Docker Support**: Containerized backend and frontend for easy deployment
- **Cloud Ready**: Deployable on Google Cloud Run

## 📁 Project Structure

```
OCR Project/
├── app.py                      # Flask backend API
├── ocr_model.py                # Model architecture and preprocessing
├── train.py                    # Training script
├── eval_accuracy.py            # Model evaluation
├── requirements.txt            # Python dependencies
├── Dockerfile.backend          # Backend container
├── Dockerfile.frontend         # Frontend container
├── nginx.conf                  # Nginx configuration for frontend
├── templates/
│   └── index.html              # Web UI
├── checkpoints/
│   └── ocr_predict.keras       # Trained model weights
└── report.tex                  # LaTeX project report
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "OCR Project"
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open browser**
```
http://localhost:5000
```

## 🐳 Docker Deployment

### Backend

```bash
docker build -f Dockerfile.backend -t ocr-backend .
docker run -p 8080:8080 ocr-backend
```

### Frontend

```bash
docker build -f Dockerfile.frontend -t ocr-frontend .
docker run -p 80:80 ocr-frontend
```

## ☁️ Google Cloud Run Deployment

### Prerequisites
- Google Cloud account
- gcloud CLI installed
- Docker Desktop running

### Deploy Backend

```bash
# Set your project ID
PROJECT_ID="your-project-id"

# Build and push
docker build -f Dockerfile.backend -t us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-backend:v1 .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-backend:v1

# Deploy
gcloud run deploy ocr-backend \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-backend:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120
```

### Deploy Frontend

1. Update `templates/index.html` with your backend URL
2. Build and deploy:

```bash
docker build -f Dockerfile.frontend -t us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-frontend:v1 .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-frontend:v1

gcloud run deploy ocr-frontend \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ocr-repo/ocr-frontend:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 256Mi
```

## 🏗️ Architecture

### Model Architecture

```
Input (128×32×1)
    ↓
CNN Feature Extractor (3 Conv + MaxPool layers)
    ↓
Reshape to Sequence (16 time steps)
    ↓
BiLSTM Layers (2 stacked)
    ↓
Dense Output (76 classes: 75 chars + CTC blank)
    ↓
CTC Loss / Greedy Decoding
```

**Model Size**: ~497K parameters  
**Input**: 128×32 grayscale images  
**Output**: Variable-length text sequences

### Preprocessing Pipeline

1. Grayscale conversion
2. Otsu binarization
3. Aspect-ratio-preserving resize
4. Normalization to [0, 1]
5. Transpose for CTC (W, H, C)

## 📊 Dataset

- **Total Images**: 422,400
- **Classes**: 66 (digits, letters, punctuation)
- **Images per Class**: 6,400
- **Train/Val Split**: 90% / 10%

## 🔧 Training

```bash
python train.py
```

**Hyperparameters**:
- Batch size: 32
- Epochs: 100 (with early stopping)
- Learning rate: 0.001 (with ReduceLROnPlateau)
- Optimizer: Adam

## 📈 Evaluation

```bash
python eval_accuracy.py
```

Measures exact-match accuracy on validation set.

## 🛠️ Tech Stack

- **Backend**: Flask, TensorFlow 2.21, Keras 3.14, OpenCV
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Deployment**: Docker, Google Cloud Run, Nginx
- **Training**: Python 3.11, NumPy, scikit-learn

## 📝 API Endpoints

### `GET /`
Health check and API info

### `POST /predict`
Upload image for OCR prediction

**Request**: `multipart/form-data` with `image` field  
**Response**:
```json
{
  "status": "success",
  "prediction": "recognized text"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of an academic assignment at ITBS Nabeul.

## 👥 Authors

- **Mohamed Yassine Chebâane**
- **Issam Idin Mbarek**

**Academic Supervisor**: M. Ben Taleb Ahmed

## 🙏 Acknowledgments

- TensorFlow and Keras teams
- OpenCV community
- ITBS Nabeul for academic support

## 📚 References

- Graves et al. (2006) - Connectionist Temporal Classification
- Shi et al. (2016) - CRNN for Scene Text Recognition
- Otsu (1979) - Automatic Threshold Selection

---

**Year**: 2024-2025  
**Institution**: École Supérieure Privée des Technologies de l'Information et de Management de Nabeul
