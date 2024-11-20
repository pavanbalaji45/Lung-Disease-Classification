# Lung-Disease-Classification

This project focuses on the classification of lung diseases using chest X-ray images. The goal is to identify and categorize six distinct types of lung diseases using deep learning models. We have employed several advanced architectures such as CNN+LSTM, GRU, Peephole LSTM, and ensemble models like DenseNet+MobileNet, VGG16+MobileNet for robust classification performance.

## Classes

The dataset comprises six classes of lung diseases:
- **CARDIOMEGALY**
- **COVID**
- **NORMAL**
- **PNEUMONIA**
- **PNEUMOTHORAX**
- **TUBERCULOSIS**

## Dataset

- **Source**: [COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset](https://www.kaggle.com/datasets/andrewmvd/covid19-pneumonia-normal-chest-xray)
- The dataset consists of 16,050 training images and 4,104 testing images, divided across six disease categories.
- Each category contains **2313** chest X-ray images.
  
### Data Augmentation

To improve the robustness of the model and avoid overfitting, data augmentation techniques were applied:
- **Rescaling**: Scaling images for normalization.
- **Rotation**: Random rotation of images to simulate different angles.
- **Shift**: Random shifting of images for spatial variance.
- **Flips**: Horizontal and vertical flips to simulate different orientations.
- **Zoom**: Random zooming to simulate distance variations in images.

These techniques increase the diversity of the training set and help the model generalize better on unseen data, ensuring the model adapts to a variety of conditions, patient anatomies, and X-ray orientations.

### Data Balancing

- **SMOTE** (Synthetic Minority Over-sampling Technique) was used for data balancing to mitigate class imbalances. It generates synthetic samples for underrepresented classes to create a balanced dataset.

## Models Used

The project implements multiple deep learning architectures to achieve the best performance for classifying lung diseases. The models include:

1. **Peephole LSTM**: A variant of LSTM that uses additional connections between the layers for better memory retention.
2. **CNN + LSTM**: A combination of Convolutional Neural Networks (CNNs) for feature extraction and LSTMs for sequential data processing.
3. **GRU**: Gated Recurrent Units, an alternative to LSTM that simplifies the computation while still capturing dependencies in sequential data.
4. **GRU + Peephole LSTM**: A hybrid model combining the strengths of GRUs and Peephole LSTMs.
5. **Bidirectional LSTM (BiLSTM)**: LSTMs that process data in both forward and backward directions, enhancing feature extraction from the input sequences.
6. **DenseNet + MobileNet**: An ensemble model combining DenseNet's feature reuse and MobileNet's lightweight architecture for efficient computation.
7. **VGG16 + MobileNet**: Another ensemble combining VGG16's deep layers with MobileNet for lightweight, efficient processing.

### Performance Evaluation

- **Accuracy**: Model performance was primarily evaluated based on accuracy, where the model's ability to correctly predict lung disease classes was measured.
- **Cross-validation**: A robust validation technique to ensure the generalization of the model.

## Installation

### Prerequisites

Ensure the following libraries are installed:
```bash
pip install tensorflow keras opencv-python scikit-learn imbalanced-learn matplotlib
```
Dataset Preparation
Download the dataset from Kaggle.
dataset/
    train/
        CARDIOMEGALY/
        COVID/
        NORMAL/
        PNEUMONIA/
        PNEUMOTHORAX/
        TUBERCULOSIS/
    test/
        CARDIOMEGALY/
        COVID/
        NORMAL/
        PNEUMONIA/
        PNEUMOTHORAX/
        TUBERCULOSIS/
Running the Code
To train the models:

Place the dataset and model files in the appropriate directories.
Results
The ensemble models, particularly the DenseNet + MobileNet and VGG16 + MobileNet, achieved the highest classification accuracy, making them the best-performing models for this task.
Data augmentation and the use of SMOTE significantly improved the model's ability to generalize and handle imbalanced classes.
Conclusion
This project demonstrates the application of various advanced deep learning models for the classification of lung diseases from chest X-ray images. The results highlight the importance of combining CNNs with recurrent networks like LSTMs and GRUs, as well as leveraging ensemble techniques for better performance.

References
Kaggle Dataset: COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset
SMOTE: SMOTE: Synthetic Minority Over-sampling Technique




![before smote](https://github.com/user-attachments/assets/ebdbdd86-8b5e-49fa-8cc5-b97751a605e5)


![after](https://github.com/user-attachments/assets/e71c2646-e74c-44f1-93ad-296c2d24f63e)
