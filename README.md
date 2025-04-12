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
In this problem, we design a Vision Transformer (ViT) from scratch for the CIFAR-100 dataset, which consists of 100 classes of 32×32 RGB images. We experiment with different ViT configurations by varying the patch size, embedding dimension, the number of transformer layers, and the number of attention heads. Our goal is to evaluate the trade-offs between model complexity (i.e., parameter count and estimated FLOPs), training time, and performance (validation and test accuracy) compared to a ResNet-18 baseline.

### 1.2 Implementation Details
- **Data Loading:**  
  The CIFAR-100 dataset is loaded using `torchvision.datasets.CIFAR100` with standard data augmentation (random crop, horizontal flip) and normalization. The dataset is split into training, validation, and test sets.

- **ResNet-18 Baseline:**  
  We use a modified ResNet-18 (with the final fully-connected layer set to output 100 classes) as our baseline model.

- **Vision Transformer (ViT) Architecture:**  
  The custom ViT model consists of the following components:
  - **Patch Embedding:** Divides the input 32×32 image into non-overlapping patches (configurations include 4×4 and 8×8).
  - **Transformer Encoder Blocks:** Stacked to form the main network. We experiment with 4 and 8 layers.
  - **Classification Head:** A linear layer that maps the class token features to 100 classes.
  
  We vary the following hyperparameters:
  - **Patch Size:** 4×4 and 8×8
  - **Embedding Dimension:** 256 and 512
  - **Transformer Layers:** 4 and 8
  - **Attention Heads:** 2 and 4  
  The MLP hidden dimension in each transformer block is set to four times the embedding dimension (e.g. 1024 when the embed_dim is 256).

- **Training Setup:**  
  Both the ResNet-18 baseline and ViT models are trained on CIFAR-100 with the same hyperparameters:
  - **Batch Size:** 64 (or 128 as specified in parts of the code)
  - **Epochs:** 50
  - **Optimizer:** Adam with a learning rate of 0.001  
  We record model parameter counts, estimated FLOPs (using tools such as `torchinfo` or manual computation), training time per epoch, and test accuracy.

### 1.3 Experimental Results

#### ResNet-18 Baseline:
- **Parameter Count:** 11,220,132  
- **Test Accuracy:** 65.86%

#### Default ViT:
- **Parameter Count:** 3,214,692  
- **Test Accuracy:** 45.89%

#### ViT Configurations Summary:
| Configuration | Patch Size | Embed Dim | Layers | Heads | # of Params | Final Val Acc | Avg Epoch Time (s) | Test Accuracy |
|---------------|------------|-----------|--------|-------|-------------|---------------|--------------------|---------------|
| **Config1**   | 4×4        | 256       | 4      | 4     | 3,214,692   | 37.64%        | 8.86               | 38.72%        |
| **Config2**   | 4×4        | 256       | 8      | 4     | 6,373,732   | 40.40%        | 16.90              | 42.72%        |
| **Config3**   | 8×8        | 512       | 4      | 2     | 12,769,892  | 6.34%         | 7.58               | 6.52%         |
| **Config4**   | 8×8        | 512       | 8      | 4     | 25,379,428  | 5.34%         | 14.48              | 4.73%         |

#### Figures:
- **Training and Validation Loss Curves:**  
  ![ViT Loss Comparison](./images/p1_loss_comparison_all.png)
- **Validation Accuracy Curves:**  
  ![ViT Accuracy Comparison](./images/p1_accuracy_comparison_all.png)

### 1.4 Analysis and Discussion
- **ResNet-18 Performance:** The ResNet-18 baseline achieves a relatively high test accuracy (65.86%), demonstrating the robustness of traditional CNN architectures on CIFAR-100.
- **ViT Performance:**  
  - The Default ViT, with its chosen configuration, underperforms compared to the ResNet-18 baseline.
  - Among the ViT configurations, those with smaller patch sizes and fewer layers (Config1 and Config2) yield better validation and test accuracies than larger models (Config3 and Config4).  
  - The Config3 and Config4 models, which utilize a larger patch size and embedding dimension (8×8 and 512, respectively), incur a very high parameter count and longer training times. However, their validation and test accuracies are significantly lower, likely due to over-parameterization relative to the dataset size.
- **Trade-offs:**  
  - **Accuracy vs. Complexity:** Increasing model capacity (more layers and larger dimensions) does not always lead to improved accuracy on CIFAR-100.  
  - **Training Time:** More complex configurations require significantly more training time.
  - **Computational Cost:** A larger model size with increased FLOPs often leads to diminishing returns in performance, especially on relatively small datasets.

### 1.5 Conclusions
*(Placeholder for conclusions. In your final report, discuss your observations regarding the trade-offs between model size, computational complexity, training time, and accuracy. Explain potential reasons why certain ViT configurations underperform compared to the ResNet-18 baseline and suggest possible avenues for future improvement.)*

---

## Problem 2: Fine-tuning Pretrained Swin Transformer Models on CIFAR-100

### 2.1 Introduction
In Problem 2, we explore fine-tuning pretrained Swin Transformer models from the Hugging Face Transformers library on the CIFAR-100 dataset. Specifically, we fine-tune the Tiny (`microsoft/swin-tiny-patch4-window7-224`) and Small (`microsoft/swin-small-patch4-window7-224`) variants while comparing their performance with that of a Swin Transformer trained from scratch.

### 2.2 Implementation Details
- **Pretrained Models:**  
  - We load Swin-Tiny and Swin-Small using `SwinForImageClassification.from_pretrained()`, and then adjust their classification heads to output 100 classes.  
  - The backbone is frozen so that only the classifier head is trained.
- **Swin from Scratch:**  
  - A Swin Transformer is instantiated using a configuration derived from Swin-Tiny, modified for CIFAR-100 (with `num_labels=100`), and randomly initialized.
- **Training Setup:**  
  - **Epochs:** 5  
  - **Batch Size:** 32  
  - **Optimizers:**  
    - Pretrained models are trained using Adam with a learning rate of 2e-5.  
    - The scratch model is trained using Adam with a learning rate of 0.001.  
  - **Preprocessing:** Images are resized to 224×224 (to match the expected input of the Swin models) and normalized using the pretrained processor values.
- **Metrics Recorded:**  
  Training time per epoch, final validation accuracy, and test accuracy.

### 2.3 Experimental Results

#### Summary Table:
| Model                         | Training Method       | # of Params | Avg Epoch Time (s) | Final Val Acc | Test Accuracy |
|-------------------------------|-----------------------|-------------|--------------------|---------------|---------------|
| **Pretrained Swin-Tiny**      | Fine-tuning           | 76,900      | 86.87              | 66.14%        | 65.60%        |
| **Pretrained Swin-Small**     | Fine-tuning           | 76,900      | 143.70             | 69.38%        | 69.48%        |
| **Swin Transformer (Scratch)**| Training from scratch | 27,596,254  | 214.68             | 0.90%         | 1.00%         |

#### Figures:
- **Training and Validation Loss Curves:**  
  ![Swin Loss Comparison](./images/p2_swin_loss_comparison.png)
- **Validation Accuracy Curves:**  
  ![Swin Accuracy Comparison](./images/p2_swin_accuracy_comparison.png)

### 2.4 Analysis and Discussion
- **Pretrained Models:**  
  Fine-tuning the pretrained Swin-Tiny and Swin-Small models results in significantly higher validation and test accuracies (around 66–69%) compared to the model trained from scratch.  
  - The Swin-Small model, while taking longer per epoch, slightly outperforms Swin-Tiny on both validation and test sets.
- **Swin from Scratch:**  
  The scratch model, despite having a very large number of parameters, fails to learn effectively (final validation accuracy of only 0.90% and test accuracy of 1.00%). This reinforces the importance of pretraining, especially when working with limited datasets.
- **Trade-offs:**  
  - **Parameter Efficiency:** The pretrained models achieve robust performance with a relatively small number of trainable parameters (as the backbone is frozen).  
  - **Training Time vs. Performance:** Although the Swin-Small model takes longer per epoch than Swin-Tiny, it achieves slightly higher accuracy.  
  - **Learning from Scratch:** A significant disadvantage is observed when training from scratch—the model requires more data and compute to effectively learn meaningful representations.

### 2.5 Conclusions
*(Placeholder for conclusions. In your final submission, summarize the key findings from Problem 2 and discuss why fine-tuning pretrained models is advantageous compared to training from scratch. Highlight the differences between Swin-Tiny and Swin-Small and provide insights into potential future improvements.)*

---

## General Conclusions
- **Model Complexity and Performance:**  
  There exists a trade-off between model size, computational cost, training time, and final accuracy. While more complex models offer higher capacity, they may not necessarily translate to higher accuracy on small datasets without pretraining.
- **Pretrained versus Training from Scratch:**  
  Fine-tuned models benefit from pretrained feature representations and demonstrate significantly better performance than models trained from scratch on CIFAR-100.
- **Task-Specific Considerations:**  
  For CIFAR-100, the ResNet-18 baseline achieved the best performance in Problem 1, suggesting that traditional CNN architectures remain highly effective for this dataset. For Problem 2, fine-tuning pretrained Swin models, particularly Swin-Small, yields competitive results, highlighting the value of pretrained models in transfer learning.
- **Future Work:**  
  Future investigations could explore hybrid models or additional techniques (such as data augmentation strategies, learning rate schedules, or model ensembling) to further improve performance.
