# 🌿 Plant Leaf Disease Detection: Dataset, Model, and Deployment

This project demonstrates a **Plant Leaf Disease Detection** pipeline using deep learning and computer vision. It leverages the **Plant Leaf Disease Dataset** to identify plant diseases with an impressive **98% accuracy** using a **CNN model with InceptionV3 as the backbone**. The model is deployed for public use on [**Hugging Face Spaces**](https://huggingface.co/spaces/bhanusAI/plantifysol) with an interactive UI powered by **Gradio**.

---

## 📑 Table of Contents

- [Dataset Overview](#dataset-overview)
- [Model Details](#model-details)
- [Project Features](#project-features)
- [Technologies Used](#technologies-used)
- [Deployment](#deployment)

---

## 📊 Dataset Overview

The dataset consists of **61,486 images** across **39 plant classes**, including both healthy and diseased leaves. The images are augmented using several techniques to improve model generalization.

### 🌱 Plant Types
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato  

### 🔑 Key Details
- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)  
- **Number of Classes**: 39  
- **Augmentation Techniques**:
  - Flipping
  - Gamma correction
  - Noise injection
  - PCA color augmentation
  - Rotation
  - Scaling  

---

## 🧠 Model Details

- **Model Architecture**: Convolutional Neural Network (CNN)  
- **Backbone**: InceptionV3  
- **Accuracy**: 98% on the validation set  
- **Loss Function**: Categorical Cross-Entropy  
- **Optimizer**: Adam  

### Results
- The model achieves **98% accuracy** on the validation dataset.  
- **Sample Prediction**:
  - **Input**: Leaf image  
  - **Output**: Class label (e.g., `Tomato_yellow_leaf_curl_virus`) and disease probability.  

### 🔗 [Kaggle Notebook](https://www.kaggle.com/code/vbhanu5ai/leaf-disease-detection)

---

## 🚀 Project Features

1. **High-Performance Model**:
   - Utilizes the InceptionV3 backbone for robust image classification.
   - Achieves **98% accuracy** on the Plant Leaf Disease Dataset.

2. **Interactive Deployment**:
   - Test the model on **Hugging Face Spaces**.
   - Upload an image and specify the plant type for predictions.

3. **Workflow**:
   - **Data Preprocessing**: Image resizing, augmentation.
   - **Model Training**: TensorFlow framework with GPU acceleration on **Kaggle P100 GPU**.
   - **Model Deployment**: Hugging Face Spaces and Gradio for UI.

---

## 🛠️ Technologies Used

| **Technology**        | **Purpose**                        |
|------------------------|------------------------------------|
| OpenCV                 | Image processing and augmentation |
| TensorFlow             | Model training and deep learning  |
| NumPy                  | Numerical computations            |
| Pandas                 | Data handling                     |
| Matplotlib             | Visualizing images and metrics    |
| Pickle                 | Model serialization               |
| Gradio                 | Interactive user interface        |
| Hugging Face Spaces    | Model deployment platform         |

---

## 🌐 Deployment

The trained model is deployed on **Hugging Face Spaces** with an interactive UI powered by **Gradio**. Test it live here:  

👉 **[Plant Leaf Disease Detection App](https://huggingface.co/spaces/bhanusAI/plantifysol)**  

![Hugging Face Output](https://github.com/Vbhanusai/leaf-disease-detection/blob/main/images/hf_output.png?raw=true)
