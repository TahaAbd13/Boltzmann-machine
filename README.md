# Boltzmann Machine - Training and Implementation

This repository contains an implementation of **Restricted Boltzmann Machines (RBMs)** trained on the **MNIST dataset** using the **Contrastive Divergence (CD)** algorithm. The primary goal of this project is to explore the training of an RBM for applications in **feature learning, generative modeling, and dimensionality reduction**. The project is divided into two phases, each represented by a Jupyter Notebook.

## Overview of Notebooks

### Project Phase 1: MNIST Preprocessing and RBM Training
This notebook focuses on loading, preprocessing, and visualizing the MNIST dataset. It introduces the concept of a Restricted Boltzmann Machine and details the training process using the Contrastive Divergence algorithm. The key steps in this phase include:

- Loading the MNIST dataset and binarizing the images based on a pixel intensity threshold.
- Implementing an RBM using PyTorch.
- Training the RBM using the Contrastive Divergence algorithm.
- Generating new samples from the trained RBM.
- Visualizing the learned feature representations.

### Project Phase 2: Advanced RBM Analysis
This notebook builds upon the first phase by diving deeper into the probabilistic modeling aspects of RBMs. It introduces the concept of free energy, explores different variations of Contrastive Divergence, and provides visualization techniques to analyze the training process. The key steps in this phase include:

- A deeper theoretical understanding of RBMs and energy-based models.
- Implementing and analyzing the **Free Energy function**, which helps evaluate how well an RBM represents the input data.
- Experimenting with different values of `k` in Contrastive Divergence (CD-k) training and analyzing their effects on performance.
- Computing and visualizing the **reconstruction error**, which measures how well the RBM can reconstruct input data after passing through the hidden layer.
- Exploring different ways of visualizing training progress, including monitoring changes in free energy and reconstruction error over time.

## Getting Started

### Prerequisites
To run the notebooks in this repository, you will need Python and several libraries installed. The following dependencies are required:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

These dependencies can be installed using the following command:
```bash
pip install torch tensorflow matplotlib numpy scikit-learn seaborn
```

### Running the Notebooks
To execute the notebooks, follow these steps:
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Boltzmann-machine.git
   cd Boltzmann-machine
   ```
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Navigate to and open `Project Phase 1.ipynb` or `Project Phase 2.ipynb`.
4. Run each cell sequentially to execute the code.

## Understanding Restricted Boltzmann Machines

A **Restricted Boltzmann Machine (RBM)** is an energy-based probabilistic model that consists of two layers:
- A **visible layer**, representing observed input data.
- A **hidden layer**, used to learn internal representations of the data.

The RBM learns by minimizing an energy function that defines the probability distribution over visible and hidden units. The probability of a given state (combination of visible and hidden units) is given by:

$$
P(v, h) = \frac{1}{Z} \exp (-E(v, h)) 
$$

where \( E(v, h) \) is the energy function defined as:

$$
E(v, h) = -v^T W h - v^T b - h^T c 
$$

Here, $ W $ represents the weights connecting the visible and hidden layers, and $ b $ and $ c $ are the biases of the respective layers.

### Training an RBM Using Contrastive Divergence

The training of an RBM is performed using the **Contrastive Divergence (CD)** algorithm, which approximates the gradient of the log-likelihood function. The algorithm involves the following steps:
1. Initialize the visible units with training data.
2. Compute the probabilities of the hidden units and sample their states.
3. Reconstruct the visible units from the hidden states.
4. Compute the contrastive divergence loss and update the weights using gradient descent.

This process is repeated for multiple epochs until the RBM learns meaningful feature representations.

### Free Energy and Model Evaluation

A useful metric for evaluating RBM performance is **Free Energy**, which measures how well the RBM represents a given visible state. The free energy function is defined as:

$$ 
F(v) = -v^T b - \sum_{j} \log (1 + e^{(W^T v + c)_j}) 
$$

Lower free energy values indicate that the RBM assigns higher probabilities to the given input, meaning it has successfully learned to represent the data distribution.

## Applications of RBMs
Restricted Boltzmann Machines have various applications, including:
- **Feature Learning:** Extracting meaningful representations from raw data.
- **Dimensionality Reduction:** Reducing the complexity of high-dimensional datasets.
- **Collaborative Filtering:** Recommender systems, such as those used in Netflix’s movie recommendation algorithm.
- **Generative Modeling:** Creating new samples that resemble the training data.
- **Deep Belief Networks (DBNs):** Stacking multiple RBMs to build deeper models for classification and recognition tasks.

## References

- Hinton, G. E. (2002). "Training products of experts by minimizing contrastive divergence."
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). "A fast learning algorithm for deep belief nets."
- Fischer, A., & Igel, C. (2012). "An introduction to restricted Boltzmann machines."

## Conclusion
This repository provides an introduction to Restricted Boltzmann Machines, demonstrating how they can be trained using Contrastive Divergence on the MNIST dataset. The notebooks include theoretical explanations, practical implementations, and visualization techniques to analyze RBM performance. By following this project, users will gain hands-on experience with energy-based models and understand their applications in machine learning.

For further exploration, users can extend the work by:
- Experimenting with different datasets.
- Modifying hyperparameters such as learning rate, number of hidden units, and training epochs.
- Implementing RBMs in a Deep Belief Network (DBN) architecture.

This project serves as a foundational step toward understanding and implementing energy-based models for unsupervised learning tasks.
