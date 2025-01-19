# **Final Lab: Image Classification with Caltech-101 Dataset and Grad-CAM**

## **Objective**
This project aimed to:
1. Build, train, and evaluate three different CNN architectures for image classification using the **Caltech-101 dataset**.
2. Implement **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide visual explainability for model predictions.

---

## **Dataset Overview**
- **Caltech-101** dataset includes:
  - 101 object categories and 1 background category.
  - ~9,146 images in total, with 40–800 images per category.
- Dataset preprocessing involved:
  - Resizing images to `(128, 128)`.
  - Applying data augmentation (random flips, rotations, color jittering).
  - Normalizing images using ImageNet mean and standard deviation values.

---

## **Methodology**

### **1. Dataset Preprocessing**
- Preprocessing was implemented using PyTorch:
  - Images were resized to `(128, 128)`.
  - Data augmentation included:
    - Random horizontal and vertical flips.
    - Random rotations.
    - Color jittering for brightness, contrast, saturation, and hue.
  - Dataset Splits:
    - **Training Set**: 80%.
    - **Validation Set**: 10%.
    - **Test Set**: 10%.

---

### **2. Model Architectures**
Three CNN architectures were used:

#### **Model 1: ResNet50**
- A pre-trained **ResNet50** was fine-tuned for **102 classes** (matching the dataset).
- Training Details:
  - Optimizer: Adam
  - Learning Rate: `0.001`
  - Loss Function: CrossEntropyLoss
- **Performance**:
  - Test Accuracy: **0.55%**
  - Grad-CAM heatmaps showed poor attention to relevant image regions.

#### **Model 2: ResNet18**
- A smaller, pre-trained **ResNet18** was implemented to address overfitting.
- Training Details:
  - Same optimizer and learning rate as ResNet50.
- **Issue**:
  - **Execution time was excessively long**, leading to runtime termination.
- This highlighted the need for runtime optimization or extended resources.

#### **Model 3: EfficientNet-B0**
- A lightweight **EfficientNet-B0** was implemented to balance computational efficiency and performance.
- Training Details:
  - Reduced batch size to avoid runtime termination.
  - Optimizer: Adam with a learning rate of `0.0001`.
- **Performance**:
  - Test Accuracy: **~2.5%**
  - Grad-CAM heatmaps showed better focus on relevant regions compared to ResNet50.

---

### **3. Grad-CAM Implementation**
- **Objective**: Grad-CAM was used to explain model predictions by visualizing the regions of input images that contributed most to the model's decision.
- **Target Layer**:
  - ResNet: `layer4[-1]` (final convolutional block).
  - EfficientNet: `blocks[-1]` (final convolutional block).
- **Visualizations**:
  - Grad-CAM heatmaps provided insights into model focus areas.
  - While ResNet50 struggled with irrelevant focus areas, EfficientNet-B0 showed improved attention.

---

## **Results**

### **1. Training Metrics**
| **Model**         | **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** | **Runtime**                |
|--------------------|-----------------------|--------------------------|-------------------|----------------------------|
| **ResNet50**       | ~85%                 | ~0.8%                   | **0.55%**         | Completed successfully.    |
| **ResNet18**       | N/A                  | N/A                     | **N/A**           | **Runtime terminated.**    |
| **EfficientNet-B0**| ~90%                 | ~2%                     | **2.5%**          | Completed with longer time.|

---

### **2. Grad-CAM Results**
- **ResNet50**:
  - Heatmaps often highlighted irrelevant regions, leading to misclassifications.
- **ResNet18**:
  - Grad-CAM results could not be obtained due to runtime termination.
- **EfficientNet-B0**:
  - Grad-CAM heatmaps demonstrated better attention to relevant image regions, correlating with improved test performance.

---

## **Challenges and Limitations**

1. **Dataset Imbalance**:
   - Some classes contained significantly fewer samples, impacting model learning.
2. **Model Overfitting**:
   - ResNet50 overfitted the training data, leading to poor test accuracy.
3. **Runtime Limitations**:
   - ResNet18 training could not be completed due to excessive runtime.
4. **Low Test Accuracy**:
   - Despite successful training, ResNet50 and EfficientNet-B0 achieved low test accuracy, indicating potential issues with dataset size and diversity.

---

## **Recommendations**

1. **Simplify the Model Architecture**:
   - Use smaller models like MobileNetV2 to reduce runtime and improve scalability.
2. **Improve Data Augmentation**:
   - Implement advanced techniques like **MixUp** or **CutMix** for better generalization.
3. **Use Weighted Loss**:
   - Address class imbalance by assigning weights to classes in the loss function.
4. **Extend Runtime Resources**:
   - Use cloud services or platforms like **Google Colab Pro** for longer runtime.
5. **Hyperparameter Tuning**:
   - Experiment with different learning rates, optimizers, and batch sizes.

---

## **Conclusion**
This project demonstrated the application of CNNs for image classification using the Caltech-101 dataset and Grad-CAM for explainability. While the third model (EfficientNet-B0) showed some improvement in test accuracy and Grad-CAM results, further efforts are needed to address dataset complexity, class imbalance, and runtime constraints.

---

## **Future Work**
1. Investigate additional lightweight architectures like MobileNetV2 or ShuffleNet.
2. Experiment with larger datasets or pre-trained models on similar datasets.
3. Use distributed training or cloud resources to handle longer runtime requirements.

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/MdShakil-Hossain/366-Final-Lab.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and run all cells.

---

## **Acknowledgments**
- **Caltech** for the dataset.
- **PyTorch** and **Grad-CAM** for tools used in model training and explainability.

---
