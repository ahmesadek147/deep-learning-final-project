# Grid Search vs Random Search (Milestone 1)

This comparison uses the persisted search CSV summaries and MLflow run timestamps. No search was rerun, and no checkpoint was changed.

## Experimental context

P4 is an 8-configuration exhaustive Grid Search over the Scratch CNN (`simple_cnn`). P5 is an 8-trial seeded Random Search over a 36-configuration space using ResNet18 Transfer Learning (pretrained, with the sampled `freeze_backbone` setting). Because model family and training setup differ, these results cannot isolate Grid Search versus Random Search.

## Search-level comparison

| Quantity | P4 Grid Search | P5 Random Search |
|---|---:|---:|
| Completed trials | 8 | 8 |
| Search-space cardinality | 8 | 36 |
| Failed trials | 0 | 0 |
| Wall-clock span (s) | 7732.527 | 5238.028 |
| Sum of trial runtimes (s) | 7715.024 | 5218.765 |
| Average trial runtime (s) | 964.378 | 652.346 |
| Trial runtime population SD (s) | 39.680 | 67.625 |

## Validation trial statistics

| Metric | P4 mean | P4 SD | P4 best | P5 mean | P5 SD | P5 best |
|---|---:|---:|---:|---:|---:|---:|
| Validation Macro F1 | 0.038389 | 0.003496 | 0.043671 | 0.110077 | 0.161038 | 0.433089 |
| Validation Accuracy | 0.053527 | 0.003625 | 0.058431 | 0.133347 | 0.165414 | 0.453255 |
| Validation Loss | 4.656291 | 0.025361 | 4.615377 | 4.092829 | 0.868516 | 2.162699 |

## Exact timing evidence

Times below are UTC values recorded in MLflow. The wall-clock span is first trial start to last trial end; the runtime sum excludes inter-trial gaps. No timing estimates were used.

### P4 Grid Search
- First start: `2026-10-29T19:50:19.845000+00:00`
- Last end: `2026-10-29T21:59:12.372000+00:00`
- Wall-clock span: `7732.527` seconds
- Sum of trial runtimes: `7715.024` seconds

| Trial | Start (UTC) | End (UTC) | Duration (s) |
|---|---|---|---:|
| grid-18850aea-trial-001 | 2026-10-29T19:50:19.845000+00:00 | 2026-10-29T20:06:08.895000+00:00 | 949.050 |
| grid-18850aea-trial-002 | 2026-10-29T20:06:11.425000+00:00 | 2026-10-29T20:21:48.406000+00:00 | 936.981 |
| grid-18850aea-trial-003 | 2026-10-29T20:21:50.880000+00:00 | 2026-10-29T20:37:32.499000+00:00 | 941.619 |
| grid-18850aea-trial-004 | 2026-10-29T20:37:34.721000+00:00 | 2026-10-29T20:53:06.841000+00:00 | 932.120 |
| grid-18850aea-trial-005 | 2026-10-29T20:53:09.529000+00:00 | 2026-10-29T21:09:26.478000+00:00 | 976.949 |
| grid-18850aea-trial-006 | 2026-10-29T21:09:28.862000+00:00 | 2026-10-29T21:26:48.438000+00:00 | 1039.576 |
| grid-18850aea-trial-007 | 2026-10-29T21:26:50.974000+00:00 | 2026-10-29T21:43:46.108000+00:00 | 1015.134 |
| grid-18850aea-trial-008 | 2026-10-29T21:43:48.777000+00:00 | 2026-10-29T21:59:12.372000+00:00 | 923.595 |

### P5 Random Search
- First start: `2026-10-29T22:09:57.969000+00:00`
- Last end: `2026-10-29T23:37:15.997000+00:00`
- Wall-clock span: `5238.028` seconds
- Sum of trial runtimes: `5218.765` seconds

| Trial | Start (UTC) | End (UTC) | Duration (s) |
|---|---|---|---:|
| random-618382b5-trial-001 | 2026-10-29T22:09:57.969000+00:00 | 2026-10-29T22:21:32.936000+00:00 | 694.967 |
| random-618382b5-trial-002 | 2026-10-29T22:21:35.501000+00:00 | 2026-10-29T22:33:09.953000+00:00 | 694.452 |
| random-618382b5-trial-003 | 2026-10-29T22:33:13.485000+00:00 | 2026-10-29T22:44:50.780000+00:00 | 697.295 |
| random-618382b5-trial-004 | 2026-10-29T22:44:53.397000+00:00 | 2026-10-29T22:56:31.650000+00:00 | 698.253 |
| random-618382b5-trial-005 | 2026-10-29T22:56:34.141000+00:00 | 2026-10-29T23:06:42.037000+00:00 | 607.896 |
| random-618382b5-trial-006 | 2026-10-29T23:06:44.877000+00:00 | 2026-10-29T23:14:58.683000+00:00 | 493.806 |
| random-618382b5-trial-007 | 2026-10-29T23:15:01.227000+00:00 | 2026-10-29T23:25:40.994000+00:00 | 639.767 |
| random-618382b5-trial-008 | 2026-10-29T23:25:43.668000+00:00 | 2026-10-29T23:37:15.997000+00:00 | 692.329 |

## Best hyperparameters

- **P4 Grid, trial 003:** `learning_rate=0.001`, `weight_decay=0.00001`, `dropout=0.2`; best validation accuracy `0.0584307`, macro F1 `0.0436706`, loss `4.6153766`.
- **P5 Random, trial 005:** `learning_rate=0.001`, `weight_decay=0.0001`, `dropout=0.2`, `freeze_backbone=True`; best validation accuracy `0.4532554`, macro F1 `0.4330889`, loss `2.1626992`.

## Strengths and limitations

Grid Search is exhaustive over its declared discrete space and provides complete coverage, but its cost grows multiplicatively with each dimension. Random Search explores more distinct combinations per fixed trial budget when the space is larger and can be reproducible with a seed, but it does not guarantee coverage or identify the global best configuration.

Random Search should not be called superior from these results: P5 uses a different model family (ResNet18 Transfer Learning), different effective batch size, and a different search space. The large score gap primarily demonstrates the model/training difference, not an isolated search-strategy effect.

## Conclusion

Within the recorded experiments, P4 exhaustively evaluated all 8 Scratch CNN configurations and P5 sampled 8 of 36 ResNet18 Transfer Learning configurations. P5 achieved the stronger validation results, but the comparison is descriptive rather than causal. A fair search-strategy comparison would hold architecture, data, budget, and search dimensions constant, then compare exhaustive and random selection under matched compute.

## Charts

- `figures/metric_comparison.png`
- `figures/trial_score_distributions.png`
- `figures/trial_durations.png`
