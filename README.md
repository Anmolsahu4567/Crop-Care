# Crop-Care
Crop Care is an AI-powered plant disease recognition system built using Deep Learning and Streamlit that detects crop diseases from leaf images and provides quick diagnostic guidance.

# 🌿 Crop Care – Plant Disease Recognition System

Crop Care is an AI-powered web application that detects plant diseases from leaf images using a trained Deep Learning model. It helps farmers, agriculture students, and researchers identify crop diseases quickly and take early preventive action.

This project is an academic and practical implementation of **Artificial Intelligence in Agriculture**.

---

## 🚀 Features
- Upload a leaf image and get instant disease prediction
- Supports **38 different plant disease and healthy classes**
- Simple and user-friendly web interface using **Streamlit**
- Fast and accurate predictions using a trained **CNN model**
- Works on common image formats (jpg, jpeg, png)
- Useful for **farmers, students, and academic projects**

---

## 🧠 Technology Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Web Framework:** Streamlit  
- **Libraries Used:** NumPy, PIL, TensorFlow  
- **Model Type:** Convolutional Neural Network (CNN)

---

## 📂 Dataset Information
- Dataset Name: **New Plant Diseases Dataset**
- Source: Kaggle  
- Dataset Link:  
  🔗 https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

- Total Images: **~87,000 RGB images**
- Classes: **38 crop disease & healthy categories**
- Dataset split:
  - Train: 70,295 images
  - Validation: 17,572 images
  - Test: 33 images
- Dataset prepared using **offline data augmentation** while preserving directory structure.

---

## 🧪 How It Works
1. User uploads a plant leaf image.
2. The image is preprocessed and resized to **128×128**.
3. The trained CNN model analyzes the image.
4. The model predicts the disease class.
5. The predicted disease name is displayed on the screen.

---

## 🖥️ Installation & Run Locally

```bash
git clone https://github.com/<your-username>/crop-care.git
cd crop-care
pip install -r requirements.txt
streamlit run main.py
```

> ⚠️ **Note:** The trained model file (`trained_model.keras`) is not included in this repository
> because of its size. Download it from `<add your model download link here>` and place it in
> the project's root folder before running the app.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Developer
Developed by **Anmol Sahu**
