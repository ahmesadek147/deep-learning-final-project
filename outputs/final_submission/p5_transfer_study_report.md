# P5 Transfer-Learning Study

Controlled comparison: one factor changes per run, everything else (split, seed, image size, augmentation, batch size, epoch limit, optimizer, weight decay) is held constant by the `common` section of `configs/p5_transfer_study.yaml`. All figures are **validation** metrics on the duplicate-safe `stratified_group_v2` split; no test data was used.

## Results

| Run | Architecture | Backbone | Learning rate | Best epoch | Val accuracy | Val macro F1 | Val loss | Duration | Peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| efficientnet_b0-finetune-lr1e-4 | efficientnet_b0 | fine-tuned | 0.0001 | 9 | 72.75% | 70.76% | 1.0015 | 854.9s | 434.4 MB |
| resnet18-finetune-lr1e-4 | resnet18 | fine-tuned | 0.0001 | 9 | 64.42% | 63.05% | 1.3675 | 634.3s | 332.8 MB |
| resnet18-finetune-lr3e-5 | resnet18 | fine-tuned | 3e-05 | 10 | 63.67% | 61.11% | 1.4859 | 655.2s | 466.5 MB |
| efficientnet_b0-frozen-lr1e-3 | efficientnet_b0 | frozen | 0.001 | 9 | 47.42% | 45.11% | 2.1109 | 585.3s | 81.1 MB |
| resnet18-frozen-lr1e-3 | resnet18 | frozen | 0.001 | 10 | 44.25% | 40.95% | 2.3354 | 551.5s | 93.7 MB |

## Reading

The strongest configuration is **efficientnet_b0-finetune-lr1e-4** at 72.75% validation accuracy and 70.76% macro F1.
Best frozen-backbone run: efficientnet_b0-frozen-lr1e-3 at 45.11% macro F1. Best fine-tuned run: efficientnet_b0-finetune-lr1e-4 at 70.76%, 25.65 points above it. Compare this against the original random search, where every fine-tuned trial used a learning rate of 1e-3 or 3e-3 and scored between 1.92% and 5.76% accuracy: the collapse there was caused by the learning rate, not by fine-tuning itself.
Across architectures: efficientnet_b0 reaches 70.76% macro F1 (efficientnet_b0-finetune-lr1e-4); resnet18 reaches 63.05% macro F1 (resnet18-finetune-lr1e-4).

## Files

- `study_results.csv`, `study_results.json`: per-run metrics
- `figures/transfer_study.png`: validation accuracy and macro F1 per run
