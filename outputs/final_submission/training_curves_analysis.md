# Training and Validation Curves

These curves use the actual epoch-level values recorded in MLflow. No values were interpolated or fabricated.

## Exact runs

| Part | Experiment | Run name | Run ID |
|---|---|---|---|
| P4 Scratch CNN (Grid trial 003) | CUB_P4_GridSearch | grid-18850aea-trial-003 | `d8bebc0669234e97ae94a3b4ed581a97` |
| P5 ResNet18 Transfer (Random trial 005) | CUB_P5_RandomSearch | random-618382b5-trial-005 | `a54f1b97a45e402fa7570045567dfcb9` |

## Best checkpoint epochs

- **P4:** epoch 15, based on the lowest recorded validation loss (`4.615377`) in the search summary/checkpoint selection.
- **P5:** epoch 8, based on the lowest recorded validation loss (`2.162699`) in the search summary/checkpoint selection. P5's epoch-10 validation accuracy is higher than epoch 8, but that does not change the loss-selected checkpoint.

## Convergence analysis

- **P4 Scratch CNN:** training loss decreases steadily from 5.278348 to 4.262284 and training accuracy rises from 1.00% to 8.44%. Validation loss also declines overall from 5.122367 to 4.615377, while validation macro F1 rises from 0.005439 to 0.043671. The curves show slow learning and substantial underfitting; there is no sustained train/validation divergence indicating strong overfitting.
- **P5 ResNet18 Transfer Learning:** training loss falls from 4.830311 to 1.908178 and training accuracy rises to 49.23%. Validation loss reaches its minimum at epoch 8, then increases at epochs 9–10. Validation macro F1 peaks at epoch 8 (0.433089) before a slight decline, while accuracy fluctuates and reaches 46.41% at epoch 10. This is consistent with useful transfer-learning convergence followed by mild validation-loss overfitting after epoch 8.

## Comparison

P5 starts and converges much faster because the pretrained ResNet18 backbone provides transferable visual features; P4 must learn visual features from scratch and remains underfit within its 15-epoch limit. These curves compare different architectures and training configurations, so they are not an isolated optimizer or search-strategy comparison.

## Files

- `loss_curves.png`: training versus validation loss.
- `accuracy_curves.png`: training versus validation accuracy.
- `validation_macro_f1.png`: validation macro F1 by logged epoch.
- `metric_histories.json`: exact extracted MLflow values used for the plots.
