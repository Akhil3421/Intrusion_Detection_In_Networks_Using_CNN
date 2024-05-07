# Intrusion Detection System using CNN-MLP Architecture

This project implements an Intrusion Detection System (IDS) using a hybrid architecture combining Convolutional Neural Network (CNN) and Multilayer Perceptron (MLP) models. The IDS is trained on the UNSW-NB15 dataset to detect and classify network intrusions.

## Project Overview

Network intrusion detection plays a critical role in cybersecurity by identifying unauthorized or malicious activities within a network. This project leverages machine learning techniques to automatically detect and classify various types of network attacks, enhancing network security.

## Project Notebook

The implementation and training of the CNN-MLP model for this project are provided in a Jupyter Notebook hosted on Google Colab. You can access the notebook using the following link:

[Open Jupyter Notebook on Google Colab](https://colab.research.google.com/drive/1Ue2Nu54S1GtQ0QNUnYsYZrcSaNljbs56?usp=sharing)


## Features

- Preprocessing of the UNSW-NB15 dataset, including feature selection, standardization, and encoding.
- Implementation of a CNN-MLP model architecture for intrusion detection.
- Integration with Flask to provide a user interface for model deployment and interaction.
- Utilization of SMOTE ENN (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors) for handling class imbalance.
- Real-time intrusion detection and feedback through the web-based interface.

## Project Structure

- `CNN_MLP_Intrusion_Detection.ipynb`: Jupyter Notebook containing the implementation and training of the CNN-MLP model.
- `app.py`: Flask application for serving the trained model and providing the user interface.
- `templates/`: HTML templates for rendering the user interface.
- `static/`: Static assets (e.g., CSS, JavaScript) for the web interface.

## Getting Started

### Prerequisites

- Python 3.10.8
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Akhil3421/Intrusion_Detection_In_Networks_Using_CNN
   cd intrusion-detection-cnn-mlp
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
### Usage
1. Start the Flask application:
   ```bash
   python app.py
2. Open your web browser and navigate to http://localhost:5000 to access the user interface.
3. Upload CSV files containing network traffic data or manually input features via the web interface.
4. Click "Detect Intrusion" to initiate the intrusion detection process.
5. View predicted intrusion types or probabilities displayed on the interface.

### Future Scope
- Explore advanced deep learning architectures for improved detection accuracy.
- Enhance real-time monitoring capabilities and integration with security operations.
- Deploy the system on cloud platforms for scalability and accessibility.
