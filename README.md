# Blockchain Anomaly Detection with GNNs

This project provides a comprehensive framework for detecting illicit transactions in blockchain networks using Graph Neural Networks (GNNs). All the code for data loading, preprocessing, model training, and evaluation is contained within a single Jupyter Notebook.

---

## About The Project

Blockchain's pseudonymous nature makes it a target for illicit activities such as money laundering and fraud. Traditional detection methods often fail to capture the complex patterns within blockchain transaction networks. This project addresses these challenges by modeling the blockchain ledger as a graph and applying deep learning models to identify suspicious activities.

The framework uses GNNs—specifically **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, and **Graph Isomorphism Networks (GIN)**—to analyze the relational and structural information of transaction graphs, leading to more accurate anomaly detection.

### Key Features:

* **All-in-One Notebook**: All code for data processing, model definition, training, and evaluation is in a single, easy-to-use Jupyter Notebook.
* **Advanced GNN Models**: Implements and evaluates three powerful GNN architectures for comprehensive analysis.
* **Real-World Dataset**: Utilizes the **Elliptic Dataset**, a large, public dataset of Bitcoin transactions labeled as "licit" or "illicit".
* **High Performance**: The GIN model, in particular, demonstrates superior accuracy in identifying fraudulent transactions compared to other models.

---

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

Make sure you have Python 3.x and the following libraries installed:

* `torch` & `torchvision`
* `torch_geometric`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `jupyter notebook` or `jupyter lab`

### Installation

1.  **Clone the repo:**
    ```sh
    git clone https://github.com/adarsh-naik-2004/Blockchain-Fraud-Detection-GNN
    ```
2.  **Install required packages:**
    It is recommended to create a virtual environment first.
    ```sh
    pip install torch torchvision torch-geometric pandas numpy scikit-learn matplotlib jupyterlab
    ```

---

## Usage

1.  Place the Elliptic dataset CSV files into the `/dataset/` directory.
2.  Launch Jupyter and open the `Blockchain_Anomaly_Detection.ipynb` notebook.
    ```sh
    jupyter lab
    ```
3.  Run the cells in the notebook sequentially.

The notebook is self-contained and will guide you through the following steps:
* Loading and preprocessing the data.
* Defining the GCN, GAT, and GIN model architectures.
* Training each model on the transaction graph.
* Evaluating and visualizing the performance with accuracy, precision, recall, F1-scores, and confusion matrices.

---

## Proposed Framework

The project follows a structured framework for anomaly detection, all implemented within the notebook:

1.  **Load Dataset**: The Elliptic dataset (features, edges, and labels) is loaded using pandas.
2.  **Preprocess Data**: Transaction IDs are mapped to numerical indices, and the data is prepared for graph construction.
3.  **Graph Construction**: The transaction data is transformed into a graph structure compatible with PyTorch Geometric (PyG).
4.  **Define GNN Architectures**: GCN, GAT, and GIN model classes are defined directly within the notebook.
5.  **Model Training**: The models are trained to classify transactions, with their performance monitored on a validation set.
6.  **Performance Evaluation**: The trained models are assessed on a test set using metrics like accuracy and confusion matrix analysis to determine their effectiveness.

---

## Results

All models were trained for 100 epochs, and the GIN architecture demonstrated the best performance.

### Model Accuracy:

* **GCN**: Achieved a final training accuracy of **92.29%**.
* **GAT**: Reached a training accuracy of **90.58%**.
* **GIN**: Outperformed both with a training accuracy of **93.41%**.

---
