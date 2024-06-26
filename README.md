# Neural Network Project

This repository contains a machine learning project focused on neural networks and clustering techniques, developed as part of a Machine Learning lecture.

## Table of Contents
- [Introduction](#introduction)
- [Project Purpose](#project-purpose)
- [Part 1: Neural Network](#part-1-neural-network)
- [Part 2: K-Means Clustering](#part-2-k-means-clustering)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
This repository contains two main parts: a neural network implementation for regression tasks and a K-means clustering algorithm. The goal is to provide hands-on experience with different machine learning techniques and demonstrate their applications using Python.

## Project Purpose
The primary purpose of this project is to apply machine learning techniques to solve practical problems. It includes building and training a neural network for regression tasks and implementing a K-means clustering algorithm for data segmentation. The project demonstrates the use of various libraries and frameworks for data manipulation, model training, and visualization.

## Part 1: Neural Network
The first part of the project involves building a neural network to predict a specific target variable from the dataset.

### Code Details:
- **Libraries Used:**
  - `pandas`: Data manipulation and organization.
  - `scikit-learn`: Data splitting, scaling, and metrics.
  - `numpy`: Numerical operations and array handling.
  - `torch`: Neural network creation.
  - `torch.nn`: Neural network layers.
  - `torch.optim`: Optimizers for training.
  - `torch.utils.data`: Data handling for PyTorch.

### Model Details:
- Number of hidden layers: 2
- Number of nodes in each hidden layer: 4, 3
- Activation function in each layer: ReLU
- Learning rate: 0.01
- Momentum value: 0.9
- Gradient method: SGD
- Batch size: 64
- Number of epochs: 100

### Performance Metrics:
- **Train Results:**
  - MAE: 0.1500
  - MSE: 0.0358
  - RMSE: 0.1893
  - R2: -0.2319
- **Test Results:**
  - MAE: 0.1214
  - MSE: 0.0241
  - RMSE: 0.1551
  - R2: 0.2034

## Part 2: K-Means Clustering
The second part of the project involves implementing a K-means clustering algorithm to segment the data into clusters.

### Code Details:
- **Libraries Used:**
  - `pandas`: Data manipulation and analysis.
  - `math`: Mathematical functions.
  - `tkinter`: GUI framework.
  - `tkinter.ttk`: Themed GUI components.
  - `matplotlib.pyplot`: Plotting and visualization.

### Clustering Details:
- Number of clusters (k): 3
- Maximum iterations: 100
- **Metrics:**
  - WCSS: 29.1837
  - BCSS: 28.2069
  - Dunn Index: 0.2957

### Results:
- **Cluster Counts:**
  - Cluster 1: 45 records
  - Cluster 2: 49 records
  - Cluster 3: 120 records

## Contributing
We welcome contributions to improve this project. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project was developed as part of a Machine Learning lecture. Special thanks to the course instructors and peers for their support and feedback.

## Contact
For any inquiries, please contact:
- Name: Tahsin Berk ÖZTEKİN
- Email: tahsinberkoztekin@gmail.com

