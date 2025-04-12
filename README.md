# ECGR-5106 Homework 6

## Student Information
**Name:** Yang Xu  
**Student ID:** 801443244  
**Homework Number:** 6  

## GitHub Repository
[https://github.com/xuy50/ecgr5106-hw6](https://github.com/xuy50/ecgr5106-hw6)

---

## Problem 1: Design and Analysis of Vision Transformers for CIFAR-100

### 1.1 Introduction
In this problem, we design a Vision Transformer (ViT) from scratch that is tailored for the CIFAR-100 dataset. CIFAR-100 contains 100 classes of 32x32 RGB images. Our goal is to experiment with different ViT configurations and analyze how changes in hyperparameters influence the computational complexity and performance compared to a ResNet-18 baseline.

### 1.2 Implementation Details
- **Data Loading:**  
  The CIFAR-100 dataset is loaded using `torchvision.datasets.CIFAR100` with standard data augmentation and normalization. The dataset is split into training, validation, and test sets.

- **ResNet-18 Baseline:**  
  A modified ResNet-18 architecture (with the final fully-connected layer adjusted to 100 outputs) is used as the baseline model.

- **Vision Transformer (ViT):**  
  The custom ViT model is composed of:
  - A **Patch Embedding** layer that divides the input image into fixed-size patches.
  - A series of **Transformer Encoder** blocks.
  - A **Classification Head** that uses a class token.
  
  We experimented with different configurations, varying:
  - **Patch Sizes:** 4×4 and 8×8
  - **Embedding Dimensions:** 256 and 512
  - **Transformer Layers:** 4 and 8
  - **Attention Heads:** 2 and 4  
  The MLP hidden dimension is set to four times the embedding dimension (e.g., 1024 when embed_dim=256).

- **Training Setup:**  
  Both the ViT and ResNet-18 models were trained on CIFAR-100 with the same hyperparameters:
  - **Batch Size:** 64 (or 128, as specified in the code)
  - **Epochs:** 50
  - **Optimizer:** Adam with a learning rate of 0.001  
  Model parameters, FLOPs (estimated via tools such as torchinfo or manual computation), training time per epoch, and test accuracy are recorded.

### 1.3 Results
The following table (placeholder) will summarize the results of at least four ViT configurations alongside the ResNet-18 baseline:

| Model            | Patch Size | Embed Dim | Layers | Heads | # of Params | FLOPs   | Training Time/epoch (s) | Test Accuracy |
|------------------|------------|-----------|--------|-------|-------------|---------|-------------------------|---------------|
| ResNet-18        | N/A        | N/A       | N/A    | N/A   | ...         | ...     | ...                     | ...           |
| ViT Config 1     | 4×4        | 256       | 4      | 4     | ...         | ...     | ...                     | ...           |
| ViT Config 2     | 4×4        | 256       | 8      | 4     | ...         | ...     | ...                     | ...           |
| ViT Config 3     | 8×8        | 512       | 4      | 2     | ...         | ...     | ...                     | ...           |
| ViT Config 4     | 8×8        | 512       | 8      | 4     | ...         | ...     | ...                     | ...           |

### 1.4 Analysis
*(Placeholder for your analysis and discussion of the results. Discuss the trade-offs between accuracy, model size, FLOPs, and training time. Explain possible reasons why certain configurations perform better or worse than the ResNet-18 baseline.)*

### 1.5 Figures and Plots
- **Training and Validation Loss & Accuracy Curves:**  
  ![ViT and ResNet-18 Training Curves](./images/hw6_comparison.png)

- **Additional Visualizations:**  
  *(Place any additional figures here.)*

### 1.6 Conclusions
*(Placeholder for conclusions. Add your conclusions and final remarks after analyzing all data.)*

---

## Problem 2: Fine-tuning Pretrained Swin Transformer Models on CIFAR-100

### 2.1 Introduction
This problem explores fine-tuning pretrained Swin Transformer models from the Hugging Face Transformers library on the CIFAR-100 dataset. We compare two pretrained models (Swin-Tiny and Swin-Small) with a Swin Transformer trained from scratch.

### 2.2 Implementation Details
- **Pretrained Models:**  
  The Swin-Tiny (`microsoft/swin-tiny-patch4-window7-224`) and Swin-Small (`microsoft/swin-small-patch4-window7-224`) models are loaded using `SwinForImageClassification.from_pretrained()`. Their classification heads are adjusted to output 100 classes, and the backbone is frozen so that only the classification head is trained.
  
- **Scratch Model:**  
  A Swin Transformer from scratch is implemented (adapted from a simpler ViT or a tiny version) and trained on CIFAR-100 from random initialization.
  
- **Training Setup:**  
  - **Epochs:** 5  
  - **Batch Size:** 32  
  - **Optimizers:**  
    - Pretrained models: Adam with a learning rate of 2e-5  
    - Scratch model: Adam with a learning rate of 0.001  
  - Standard CIFAR-100 preprocessing is applied.

- **Metrics Recorded:**  
  Training time per epoch, final test accuracy, and the number of parameters for each model.

### 2.3 Results
The results are summarized in the table below (placeholder):

| Model                         | Training Method       | # of Params | Training Time/epoch (s) | Final Test Accuracy |
|-------------------------------|-----------------------|-------------|-------------------------|---------------------|
| Swin-Tiny (Pretrained)        | Fine-tuning           | ...         | ...                     | ...                 |
| Swin-Small (Pretrained)       | Fine-tuning           | ...         | ...                     | ...                 |
| Swin Transformer (Scratch)    | Training from scratch | ...         | ...                     | ...                 |

### 2.4 Analysis
*(Placeholder for analysis and discussion of the differences between fine-tuning and training from scratch, as well as the impact of model size and training time on the final test accuracy.)*

### 2.5 Figures and Plots
- **Training Curves and Comparison Figures:**  
  *(Placeholder for images or plots, e.g., training loss curves, accuracy curves, etc.)*

### 2.6 Conclusions
*(Placeholder for conclusions. Summarize your findings and discuss the benefits and drawbacks of fine-tuning versus training from scratch.)*

---

## General Conclusions
*(Placeholder for overall conclusions and discussion of trade-offs among model complexity, training time, and accuracy. Add final remarks after reviewing all experimental results.)*
