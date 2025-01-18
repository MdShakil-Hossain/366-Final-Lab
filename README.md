# Final Lab: Image Classification with Caltech-101 Dataset and Grad-CAM

## Objective
The goal of this project was to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using the **Caltech-101 dataset**. Additionally, **Grad-CAM** (Gradient-weighted Class Activation Mapping) was used to provide explainability for model predictions.

---

## Dataset Overview
- **Caltech-101** contains images from 101 object categories and 1 background category.
- Approximately **9,146 images** in total, with 40–800 images per category.
- Images were resized and augmented to maintain consistency and improve generalization.

---

## Methodology

### **1. Dataset Preprocessing**
- Images were preprocessed using PyTorch:
  - Resized to `(128, 128)`.
  - Data augmentation techniques included:
    - Random horizontal/vertical flips.
    - Random rotations.
    - Color jittering.
  - Normalized with ImageNet mean and standard deviation values.
- The dataset was split into:
  - **Training Set**: 80%.
  - **Validation Set**: 10%.
  - **Test Set**: 10%.

### **2. Model Architecture**
- The **ResNet50** pre-trained model was used.
- The final fully connected layer was adjusted for **102 classes**.
- Optimizer: **Adam** with a learning rate of `0.001`.
- Loss Function: **CrossEntropyLoss**.

### **3. Training and Evaluation**
- The model was trained for **10 epochs**.
- Training and validation accuracy/loss were tracked.
- The test accuracy achieved was **0.55%**, which indicates potential overfitting or dataset complexity. Recommendations were provided for improvement.

### **4. Explainable AI (Grad-CAM)**
- **Grad-CAM** was implemented to provide explainability by highlighting the regions of the image that contributed most to the model's predictions.
- Heatmaps were generated for several test images, showcasing the areas of focus for the CNN.

---

## Results

### **Training Metrics**
- Training Loss: Decreased steadily over epochs.
- Validation Accuracy: Showed minor improvements, but the model struggled to generalize well to unseen data.

### **Test Performance**
- Test Accuracy: **0.55%**.
- Classification Report: Highlighted misclassifications across multiple categories, suggesting the need for better balancing and augmentation.

### **Grad-CAM Insights**
- Grad-CAM visualizations revealed the model's focus on certain regions of test images.
- These insights helped identify areas where the model struggled to differentiate between classes.

---

## Challenges and Solutions

### **Challenges**
1. **Dataset Imbalance**: Some classes had significantly fewer images.
2. **Model Complexity**: ResNet50 might have been too complex for the dataset size.
3. **Low Test Accuracy**: Despite adequate training, the model's performance on the test set was low.

### **Solutions**
1. Used **data augmentation** to artificially increase variability in the training set.
2. Recommended using a simpler model (e.g., ResNet18) to reduce overfitting.
3. Proposed **class-weighted loss functions** to handle class imbalance.

---

## Conclusion
This project successfully demonstrated the use of CNNs for image classification and explainability using Grad-CAM. While the model's performance can be further improved, the Grad-CAM visualizations provided valuable insights into its decision-making process.

---

## Future Work
- Explore simpler architectures (e.g., MobileNetV2, ResNet18).
- Use advanced data augmentation techniques (e.g., MixUp or CutMix).
- Apply hyperparameter tuning to find optimal training configurations.

---
