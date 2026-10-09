# Efficient Medical Image Classification Using Vision Transformers and LoRA Fine-Tuning

This project uses three different approaches to use a pretrained Vision Transformer for PneumoniaMNIST: training a new classifier on a frozen encoder, full fine-tuning, and LoRA fine-tuning. The aim was to analyze the real-world compromises between test accuracy, number of parameters that can be trained, how much GPU memory is required, and training time, along with a small experiment that fits into Colab.

The implementation is in [`pneumoniamnist_lora_colab.ipynb`](pneumoniamnist_lora_colab.ipynb). It is designed to run in Google Colab and uses PyTorch, timm, and the official MedMNIST package.

## Dataset

PneumoniaMNIST is a binary-class classification task on pediatric chest X-rays. The MedMNIST consists of 5,856 28x28 grayscale images with preprocessing data divided into 4,708 training, 524 validation and 624 test examples. For this run, there are 234 pneumonia and 390 normal images in the test split.

The notebook contains the official splits and checks their labels and sizes after loading. The ViT-Tiny input is RGB and resized to 224 × 224. Resizing allows the model to be fed the images, but cannot recover any detail lost during the initial 28 × 28 preprocessing. PneumoniaMNIST is for research and educational purposes only.

## Model and experiment design
The backbone is [`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k), a compact 5.7M-parameter pruned and fine-tuned ViT-Tiny pretrained on ImageNet-21k, and fine-tuned on ImageNet-1k. The input normalization used when creating this checkpoint is reported by timm as being mean=0.5 and stddev=1.0 per RGB channel, which is what is used in the notebook. It is licensed under Apache-2.0.

The three configurations start from the same pretrained checkpoint:

1. **Frozen encoder:** the pretrained encoder is fixed and only the new classifier is trained.
2. **Full fine-tuning:** the encoder and classifier are trained together.
3. **LoRA fine-tuning:** the pretrained weights are frozen and low-rank matrices are added to each attention block's fused query/key/value projection. The LoRA parameters and classifier are trained.

In the case of a pretrained weight matrix, LoRA updates the weight matrix by adding the term `W' = W + (alpha / r) BA`, where `r` is the rank and `alpha` is the scale factor of the weight update. The speeds in inference are not guaranteed and the adaptor itself stores task-specific weights, not the pretrained backbone.

The splits used for training, validation and testing are the official splits. Validation macro-F1 is used to select checkpoints and the test split for the final evaluation. The following results are reported on a Tesla T4 GPU with a combination of mixed precision.

## Results

| Configuration | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen encoder | 0.8285 | 0.8643 | 0.7799 | 0.7979 | 0.8176 | 0.9237 |
| Full fine-tuning | 0.8638 | 0.9054 | 0.8201 | 0.8408 | 0.8559 | **0.9779** |
| LoRA fine-tuning | **0.8750** | **0.9096** | **0.8359** | **0.8557** | **0.8689** | 0.9718 |

LoRA had the highest accuracy and macro-F1 in this run and the best ROC-AUC was obtained by full fine-tuning. The gap between those results is a reminder that there's more to the behavior of a classifier than any one metric.

Here, the test split is imbalanced so Macro-F1 is useful considering all classes equally. The per-class results of LoRA indicate that the remaining gap is observed in the cases of recalling normal images (LoRA's result: 0.68) and pneumonia images (LoRA's result: 0.99). The confusion matrix has 75 images of normal with pneumonia labels and 3 images of pneumonia with normal labels. It is important to note that this overall score should not be interpreted as a result of even performance from both classes.

### Per-class results

| Configuration | Normal precision | Normal recall | Normal F1 | Pneumonia precision | Pneumonia recall | Pneumonia F1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen encoder | 0.93 | 0.59 | 0.72 | 0.80 | 0.97 | 0.88 |
| Full fine-tuning | 0.99 | 0.65 | 0.78 | 0.82 | 0.99 | 0.90 |
| LoRA fine-tuning | 0.98 | 0.68 | 0.80 | 0.84 | 0.99 | 0.91 |

## Efficiency measurements

| Configuration | Total parameters | Trainable parameters | Trainable (%) | Elapsed fit time (s) | Peak GPU memory (MB) | Checkpoint (MB) | Adapter + head (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen encoder | 5,524,802 | 386 | 0.007 | 64.767 | 170.079 | 21.132 | — |
| Full fine-tuning | 5,524,802 | 5,524,802 | 100.000 | 44.169 | 1,239.522 | 21.131 | — |
| LoRA fine-tuning | 5,598,530 | 74,114 | 1.324 | 67.293 | 960.548 | 21.421 | 0.291 |

In this training, LoRA used approximately 22.5% less GPU memory and trained approximately 1.32% of the total parameters. It did not see an improvement in training time either: LoRA would take approximately 67 seconds, while full fine-tuning would take approximately 44 seconds. The full fine-tuning run was terminated after three epochs, and frozen and LoRA training were completed after five epochs; this is not a fair comparison of elapsed fit times, as the number of epochs trained varies and is not a fair measure of speed per epoch.

The adapter-plus-head file is smaller than the entire LoRA checkpoint but cannot be used as a standalone model. It can be loaded after training, but features still need to be loaded into the use of the backbone. This report does not include inference latency.

The notebook also restores the saved LoRA adapter and classifier to the pretrained backbone and runs a comparison of logits for a batch of 8 test images. The absolute difference of logsits in the run was the most: 0.00000000, which means that the exported adapter reproduced the saved model's outputs on the run with the most absolute difference.

## Figures and generated files

When run, the notebook writes results and figures to `/content/pneumoniamnist-lora/`:

- `results/model_comparison.csv`
- `results/per_class_metrics.csv`
- `results/efficiency_comparison.csv`
- `figures/class_distribution.png`
- `figures/sample_images.png`
- `figures/training_curves.png`
- `figures/model_comparison.png`
- `figures/confusion_matrices.png`
- `figures/per_class_f1.png`
- `figures/roc_curves.png`

This repository does not provide fabricated or prefilled result artifacts, the CSV files and plots are generated by the notebook. The training checkpoints and the dataset files are also not committed here and are written under the Colab output directory.

## Run the notebook

1. Open `pneumoniamnist_lora_colab.ipynb` in Google Colab and select a GPU runtime if available.
2. Run the cells in order. The notebook checks the runtime, installs the required packages, downloads PneumoniaMNIST and the pretrained checkpoint, runs a forward/backward smoke test, and trains the three configurations.
3. Retrieve the generated CSVs and figures from `/content/pneumoniamnist-lora/` if you want to include them in a repository or portfolio.

If a runtime disconnects, re-run the setup cell and data-loading cells and then proceed. Checkpoints are truncated after stopping runtime in Colab, copy checkpoints to persistent storage if you need to reuse it. Only use REUSE_CHECKPOINTS = True if checkpoint files can be matched.

## Limitations

- PneumoniaMNIST is small and heavily preprocessed. Its 28 × 28 images do not contain the detail of original-resolution radiographs.
- The results are from one run and one random seed. Repeated runs are needed to assess how stable the differences are.
- The class-level results show substantially lower recall for normal images than for pneumonia images.
- This experiment is not clinically validated and must not be used for diagnosis or treatment decisions.

## References

- [MedMNIST project and dataset documentation](https://medmnist.com/)
- [MedMNIST source repository and metadata](https://github.com/MedMNIST/MedMNIST)
- [ViT-Tiny model card](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
