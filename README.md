# MiniProject



# Signature Forgery Detection Using Machine Learning

## Project Overview

This project implements a machine learning model to detect forged signatures using features extracted from images of genuine and forged signatures. The system processes the signature images, extracts several key features, and trains a neural network model to classify signatures as either genuine or forged.

## Features Extracted:
- **Ratio of White Pixels**: Proportion of the signature area.
- **Centroid**: The geometric center of the signature.
- **Eccentricity**: The elongation of the signature.
- **Solidity**: Measures the density of the signature area.
- **Skewness and Kurtosis**: Measures of asymmetry and tailedness of the signature distribution.

## Technologies Used
- **Python**: Core programming language used.
- **NumPy**: For numerical operations.
- **Pandas**: To handle data in CSV format.
- **Matplotlib**: For visualization of images.
- **Scikit-Image**: Image processing and feature extraction.
- **TensorFlow**: To build and train the neural network model.
- **Keras**: For categorical data processing.
- **SciPy**: For advanced image filtering.

## Project Structure

```bash
Signature-Forgery-Detection/
│
├── real/                           # Directory for genuine signature images
├── forged/                         # Directory for forged signature images
├── features/                       # Directory where extracted features are saved
│   ├── Training/                   # Training data CSV files
│   ├── Testing/                    # Testing data CSV files
├── TestFeatures/                   # Directory for test signature features
├── main.py                         # Main script to run the project
├── README.md                       # Project documentation
└── requirements.txt                # List of required Python packages
```

## Prerequisites

To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib scikit-image tensorflow keras scipy
```

## Running the Project

1. **Prepare Dataset:**
   - Add genuine signatures to the `real/` directory.
   - Add forged signatures to the `forged/` directory.

2. **Extract Features:**
   Run the script to preprocess the images and extract features into CSV files:
   ```bash
   python main.py
   ```

3. **Train and Test the Model:**
   You can train the model and evaluate the performance using:
   ```bash
   python main.py --train --test
   ```

4. **Check Accuracy:**
   The system will output the accuracy of the model on both training and testing data for different persons.

## Model Description

The model is a **3-layer neural network** built using TensorFlow. The architecture consists of:
- **Input Layer**: 9 input features extracted from the images.
- **Hidden Layers**: 3 hidden layers with customizable neuron counts.
- **Output Layer**: A softmax layer for classification (genuine or forged).

The neural network is trained using the Adam optimizer, with the loss function being the squared difference between the model's prediction and the actual output.

## Performance Metrics

The model's performance is measured using **accuracy** for both training and testing sets. You can also visualize the results by enabling the display feature during preprocessing.

## Future Improvements

- Enhance the image preprocessing with more advanced noise removal techniques.
- Experiment with different machine learning models (e.g., SVM, Random Forest).
- Apply deep learning techniques, such as Convolutional Neural Networks (CNNs), for better accuracy.

