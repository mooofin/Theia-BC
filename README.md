# KNN-DiagnosisModel

## Overview
This project implements a **K-Nearest Neighbors (KNN) classifier** to predict whether a tumor is **malignant** or **benign** based on various medical features. The model is trained on the **Wisconsin Breast Cancer Dataset (WBCD)**, a well-known dataset in medical machine learning.

## Features & Methodology
- **Dataset**: The model uses the **WBCD**, which includes measurements such as tumor **radius**, **texture**, **perimeter**, **symmetry**, and other relevant features extracted from digitized images of fine-needle aspirate biopsies.
- **Preprocessing**: Data is **cleaned**, **normalized**, and **split** into training and testing sets to ensure effective learning.
- **Model Selection**: The **KNN algorithm** is chosen for its simplicity and effectiveness in classification tasks.
- **Optimization**: The value of **k (number of neighbors)** is optimized separately to achieve the best classification accuracy.
- **Evaluation**: The trained model is tested on new data to evaluate its **accuracy**, **precision**, and **recall**.

## Implementation Details
- The main classification logic is implemented in **`KNN.py`**.
- The optimal **k-value** is determined in a separate script.
- The trained model is **saved** as a `.pkl` file for future reuse.
- The saved model can be loaded later to classify new medical data without retraining.

## Future Improvements(soon :3)
- Implementing **cross-validation** for more robust k-value selection.
- Exploring other classification techniques such as **SVM** or **Random Forest** for comparison.
- Deploying the model as a **web application** for user-friendly diagnosis predictions.
