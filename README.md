# PyTorch Machine Learning Notebooks

This repository contains Jupyter notebooks demonstrating core PyTorch concepts and deep learning models for classification tasks. Created by muhammad-madridista as part of learning PyTorch for computer vision and NLP. 

## Overview
Notebooks cover tensor operations, neural networks, CNNs, transfer learning, and BERT for tasks like rice type classification, animal image recognition, and sarcasm detection. 

## Notebooks

| Notebook | Description | Dataset | Key Techniques | Performance |
|----------|-------------|---------|----------------|-------------|
| RiceTypeClassification.ipynb  | Tabular classification of rice types using morphological features. | Kaggle Rice Type Classification (18k samples)  | PyTorch NN, Adam optimizer, CSV data. | N/A (tabular model) |
| AnimalsImageClassificationKaggle.ipynb  | CNN for classifying cat/dog/wild animal faces. | Kaggle Animal Faces (~16k images)  | Custom CNN (3 conv layers), DataLoader, transforms, test acc ~95%  | Test Accuracy: 94.96%  |
| ImageClassificationPretrained.ipynb | Transfer learning on bean leaf lesions. | Kaggle Bean Leaf Lesions (~1k train images)  | Pretrained model (e.g., ResNet), ImageFolder. | N/A (incomplete training shown) |
| IntroductionToPytorch.ipynb  | PyTorch fundamentals: tensors, operations, reshaping. | None (tutorials) | Tensors, slicing, math ops (add/sub/mul/etc.). | N/A |
| SimpleNeuralNetworks.ipynb  | Basic feedforward networks. | N/A (assumed MNIST/simple) | Simple NNs, activation functions. | N/A |
| SarcasmDetectionUsingBertModel.ipynb | BERT for text sarcasm detection. | Sarcasm dataset (assumed tabular text). | Hugging Face BERT, fine-tuning. | N/A |
| CNN.ipynb  | CNN on MNIST digits. | MNIST (60k train, 10k test) | Conv2D (1→6→16), maxpool, FC layers, CrossEntropyLoss. | Test Accuracy: ~9.82% (early training)  |

## Setup
1. Install dependencies: `pip install torch torchvision opendatasets scikit-learn matplotlib pandas pillow transformers datasets` 
2. Download Kaggle datasets using provided credentials (opendatasets library). 
3. Use GPU if available: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
4. Run notebooks in Colab (GPU recommended) or Jupyter with PyTorch. 

Most notebooks use Google Colab with T4 GPU and Kaggle API for data. 
## Usage
- Open in Colab via "Open in Colab" badges.
- Adjust hyperparameters (e.g., epochs=10, batch=16, lr=1e-4). 
- Monitor loss/accuracy plots for convergence. 

## Contributing
Fork, add notebooks, update README. Focus on PyTorch ML projects. 
