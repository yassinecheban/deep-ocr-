# Git Commit Guide

## ✅ Files to COMMIT (push to Git)

### Core Application
- `app.py` - Flask backend API
- `ocr_model.py` - Model architecture and preprocessing
- `train.py` - Training script
- `eval_accuracy.py` - Evaluation script
- `data_loader.py` - Data loading utilities
- `test_full_page.py` - Full page OCR testing

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `README.md` - Project documentation

### Docker & Deployment
- `Dockerfile.backend` - Backend container definition
- `Dockerfile.frontend` - Frontend container definition
- `nginx.conf` - Nginx configuration

### Frontend
- `templates/index.html` - Web UI

### Model Checkpoint
- `checkpoints/ocr_predict.keras` (~2MB) - **KEEP THIS** - needed for deployment

### Documentation (LaTeX Report)
- `report.tex` - Main report file
- `introduction.tex`
- `chapitre1.tex`
- `chapitre2.tex`
- `chapitre3.tex`
- `chapitre4.tex`
- `abstract.tex`

### Optional
- `ocr_project.ipynb` - Jupyter notebook (if you want to share experiments)

---

## ❌ Files to IGNORE (already in .gitignore)

### Large Dataset Files
- `DATASET/` folder (~27GB) - **TOO LARGE**
- `train_val_images/` - Training images
- `archive.zip` (6.9GB)
- `annot.csv` (155MB)
- `annot.parquet` (42MB)
- `TextOCR_0.1_train.json` (267MB)

### Virtual Environments
- `venv/`
- `venv_old/`

### Training Artifacts
- `logs/` - TensorBoard logs
- `scratch/` - Temporary files
- `__pycache__/` - Python cache

### Redundant Model Files
- `checkpoints/ocr_best.keras` - Duplicate checkpoint
- `best_ocr_model.h5` - Old format
- `ocr_inference_model.h5` - Old format

### Test Images
- All `.jpg`, `.png`, `.webp`, `.svg` test files

---

## 🚀 Git Commands

### Initialize Repository

```bash
cd "D:\Deep learning\OCR Project"
git init
git add .
git commit -m "Initial commit: OCR Deep Learning Project"
```

### Create GitHub Repository

1. Go to https://github.com/new
2. Create repository (e.g., `ocr-deep-learning`)
3. **DO NOT** initialize with README (you already have one)

### Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/ocr-deep-learning.git
git branch -M main
git push -u origin main
```

### Check What Will Be Committed

```bash
git status
```

### Check Repository Size

```bash
git count-objects -vH
```

**Expected size**: ~10-20MB (mostly the model checkpoint)

---

## ⚠️ Important Notes

1. **Model Checkpoint**: The `checkpoints/ocr_predict.keras` file (~2MB) is small enough for Git and is REQUIRED for deployment. Keep it.

2. **Dataset**: The DATASET folder is ~27GB and should NEVER be pushed to Git. It's already in `.gitignore`.

3. **Git LFS**: If you want to version large files later, use Git Large File Storage:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git lfs track "*.keras"
   ```

4. **Private vs Public**: If your repository contains sensitive data or you don't want to share the trained model publicly, make the repository **private**.

---

## 📦 What Users Need to Run Your Project

Users cloning your repo will have:
- ✅ All source code
- ✅ Trained model (`ocr_predict.keras`)
- ✅ Docker files for deployment
- ✅ Documentation

They will NOT have:
- ❌ Training dataset (they can train their own or use your pre-trained model)
- ❌ Your virtual environment (they create their own)

This is the standard practice for ML projects on GitHub.
