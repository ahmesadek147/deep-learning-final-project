# Published Benchmark Comparison

## Purpose

This is a contextual comparison between the corrected P5 ResNet18 evaluation and a published CUB-200-2011 result. It is not a controlled benchmark: the two evaluations use different protocols and the paper's result was not reproduced in this repository.

## Results

| Model | Backbone | Dataset | Reported accuracy | Evaluation protocol |
|---|---|---|---:|---|
| Corrected P5 (this project) | ImageNet-pretrained ResNet18; 200-class head; frozen backbone | `Multimodal-Fatima/CUB_test:test` (5,794 images) | **44.9603%** | Held-out test evaluation after 10-epoch training using the deterministic `stratified_group_v2` train/validation split |
| Published reference | Standard black-box ResNet18 | CUB-200-2011 | **79.9%** | Reported in *Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts*; ImageNet-pretrained model, 200-class classifier head, 100 training epochs, default train/validation split |

Additional corrected P5 test metrics are accuracy `44.9603%`, macro precision `52.8782%`, macro recall `45.2582%`, macro F1 `44.1557%`, and loss `2.18914`. Evidence: `outputs/evaluation/p5_corrected/metrics.json` and MLflow run `de57392f5f5e45a7861f39760775986e`.

## Interpretation and limitations

The published value is shown as a reference point only. The accuracies should not be treated as directly comparable because:

- the project uses a duplicate-safe `stratified_group_v2` split, while the paper reports its default train/validation protocol;
- the project run used a 10-epoch limit, while the paper states 100 epochs;
- preprocessing, augmentation, optimization details, checkpoint selection, and other implementation details may differ;
- the project result is measured on the untouched Hugging Face test split, whereas the paper's cited evaluation protocol is not fully reproduced here.

These differences prevent attributing the accuracy gap to any single factor. In particular, this comparison does not establish that either implementation, split, training duration, or preprocessing caused the difference.

## Reference

*Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts.* European Conference on Computer Vision (ECCV), 2024. [Paper PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11588.pdf).

## Figure

`figures/published_benchmark_reference.png` visualizes the two reported accuracy values. The published bar is explicitly labeled as an external reference, not a controlled experiment.
