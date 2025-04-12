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
This problem focuses on designing a Vision Transformer (ViT) from scratch for the CIFAR-100 dataset, which comprises 100 classes of 32×32 RGB images. Different ViT configurations are experimented with by varying the patch size, embedding dimension, the number of transformer layers, and the number of attention heads. The objective is to evaluate the trade-offs between model complexity (i.e., parameter count and estimated FLOPs), training time, and performance (validation and test accuracy) compared to a ResNet‑18 baseline.

### 1.2 Implementation Details
- **Data Loading:**  
  The CIFAR‑100 dataset is loaded using `torchvision.datasets.CIFAR100` with standard data augmentation (random crop, horizontal flip) and normalization. The dataset is split into training, validation, and test sets.

- **ResNet‑18 Baseline:**  
  A modified ResNet‑18 architecture is used as the baseline model. The final fully-connected layer is adjusted to output 100 classes.

- **Vision Transformer (ViT) Architecture:**  
  The custom ViT model is composed of the following components:
  - **Patch Embedding:** Divides the input 32×32 image into non-overlapping patches (configurations include 4×4 and 8×8).
  - **Transformer Encoder Blocks:** Stacked to form the main network. Configurations with 4 and 8 layers are examined.
  - **Classification Head:** A linear layer that maps the class token features to 100 classes.
  
  The following hyperparameters are varied:
  - **Patch Size:** 4×4 and 8×8  
  - **Embedding Dimension:** 256 and 512  
  - **Transformer Layers:** 4 and 8  
  - **Attention Heads:** 2 and 4  
  The MLP hidden dimension in each transformer block is set to four times the embedding dimension (e.g., 1024 for an embedding dimension of 256).

- **Training Setup:**  
  Both the ResNet‑18 baseline and ViT models are trained on CIFAR‑100 using similar hyperparameters:
  - **Batch Size:** 64 (or 128, as specified in parts of the code)
  - **Epochs:** 50  
  - **Optimizer:** Adam with a learning rate of 0.001  
  During training, model parameter counts, estimated FLOPs (using tools such as `torchinfo` or manual computation), training time per epoch, and test accuracy are recorded.

### 1.3 Experimental Results

#### ResNet‑18 Baseline:
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
- **ResNet‑18 Performance:**  
  The ResNet‑18 baseline achieves a relatively high test accuracy of 65.86%, demonstrating the robustness of traditional CNN architectures on CIFAR‑100.
- **ViT Performance:**  
  - The Default ViT, with its chosen configuration, underperforms relative to the ResNet‑18 baseline.
  - Among the ViT configurations, those with smaller patch sizes and fewer layers (Config1 and Config2) yield better validation and test accuracies than the larger models (Config3 and Config4).  
  - The Config3 and Config4 models, utilizing a larger patch size and embedding dimension (8×8 and 512 respectively), have a very high parameter count and longer training times; however, their validation and test accuracies are significantly lower, likely due to over-parameterization for the given dataset size.
- **Trade-offs:**  
  - **Accuracy vs. Complexity:** Increasing the model capacity (by adding more layers and larger dimensions) does not always result in improved accuracy on CIFAR‑100.
  - **Training Time:** More complex configurations require significantly longer training times.
  - **Computational Cost:** Increased model size and FLOPs often lead to diminishing returns in performance when operating on relatively small datasets.

### 1.5 Conclusions
*(Placeholder for conclusions. The final report should include observations regarding the trade-offs between model size, computational complexity, training time, and accuracy. Possible reasons for the underperformance of specific ViT configurations relative to ResNet‑18 should be discussed, with suggestions for future improvements.)*

---

## Problem 2: Fine-tuning Pretrained Swin Transformer Models on CIFAR-100

### 2.1 Introduction
This problem investigates the fine-tuning of pretrained Swin Transformer models from the Hugging Face Transformers library on the CIFAR‑100 dataset. The focus is on the Tiny (`microsoft/swin-tiny-patch4-window7-224`) and Small (`microsoft/swin-small-patch4-window7-224`) variants compared against a Swin Transformer trained from scratch.

### 2.2 Implementation Details
- **Pretrained Models:**  
  The Swin-Tiny and Swin-Small models are loaded using `SwinForImageClassification.from_pretrained()` after adjusting the classification head to output 100 classes. The backbone is frozen so that only the classifier head is trained.
- **Swin from Scratch:**  
  A Swin Transformer is instantiated from a configuration derived from Swin-Tiny, modified for CIFAR‑100 (with `num_labels=100`), and randomly initialized.
- **Training Setup:**  
  - **Epochs:** 5  
  - **Batch Size:** 32  
  - **Optimizers:**  
    - Pretrained models are fine-tuned using Adam with a learning rate of 2e-5.  
    - The scratch model is trained using Adam with a learning rate of 0.001.  
  - **Preprocessing:**  
    Images are resized to 224×224 (to match the expected input of the Swin models) and normalized using the pretrained processor values.
- **Metrics Recorded:**  
  Training time per epoch, final validation accuracy, and test accuracy are recorded for each model.

### 2.3 Experimental Results

#### Summary Table:
| Model                          | Training Method       | # of Params | Avg Epoch Time (s) | Final Val Acc | Test Accuracy |
|--------------------------------|-----------------------|-------------|--------------------|---------------|---------------|
| **Pretrained Swin‑Tiny**       | Fine-tuning           | 76,900      | 86.87              | 66.14%        | 65.60%        |
| **Pretrained Swin‑Small**      | Fine-tuning           | 76,900      | 143.70             | 69.38%        | 69.48%        |
| **Swin Transformer (Scratch)** | Training from scratch | 27,596,254  | 214.68             | 0.90%         | 1.00%         |

#### Figures:
- **Training and Validation Loss Curves:**  
  ![Swin Loss Comparison](./images/p2_swin_loss_comparison.png)
- **Validation Accuracy Curves:**  
  ![Swin Accuracy Comparison](./images/p2_swin_accuracy_comparison.png)

### 2.4 Analysis and Discussion
- **Pretrained Models:**  
  Fine-tuning the pretrained Swin‑Tiny and Swin‑Small models produces significantly higher validation and test accuracies (approximately 66–69%) compared to the model trained from scratch. The Swin‑Small model, despite requiring longer training time per epoch, slightly outperforms the Swin‑Tiny model on both validation and test sets.
- **Swin from Scratch:**  
  The scratch model, although possessing a much larger number of parameters, fails to learn effectively, achieving only about 0.90% validation accuracy and 1.00% test accuracy. This result emphasizes the advantage of using pretrained weights, particularly with limited data.
- **Trade-offs:**  
  - **Parameter Efficiency:** The pretrained models yield robust performance with a small number of trainable parameters, as the backbone remains frozen.
  - **Training Time vs. Performance:** The Swin‑Small model takes longer per epoch but achieves slightly better accuracy than Swin‑Tiny.
  - **Training from Scratch:** Training from scratch is at a disadvantage here and indicates that more data or additional regularization would be needed to achieve competitive performance.

### 2.5 Conclusions
*(Placeholder for conclusions. The final report should summarize the key findings from Problem 2 and discuss the benefits and drawbacks of fine-tuning pretrained models compared to training from scratch. Differences between Swin‑Tiny and Swin‑Small should be highlighted, with insights into potential avenues for future improvements.)*

---

## General Conclusions
- **Model Complexity and Performance:**  
  A trade-off exists between model size, computational cost, training time, and final accuracy. Although more complex models offer increased capacity, this does not necessarily translate to higher accuracy on small datasets without pretraining.
- **Pretrained vs. Training from Scratch:**  
  Fine-tuned models benefit from pretrained feature representations and demonstrate significantly better performance on CIFAR‑100 than models trained from scratch.
- **Task-Specific Considerations:**  
  For CIFAR‑100, the ResNet‑18 baseline achieves high performance, indicating that traditional CNN architectures continue to be effective. In Problem 2, fine-tuning pretrained Swin models (especially Swin‑Small) produces competitive results, underscoring the value of transfer learning.
- **Future Work:**  
  Future investigations might explore hybrid models or additional techniques such as improved data augmentation, adaptive learning rate schedules, or ensemble methods to further enhance performance.
