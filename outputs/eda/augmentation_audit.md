# Data Augmentation Audit

Source: `Multimodal-Fatima/CUB_train:train`; no source images or labels were modified.

## Verified training pipeline

The configured training transform is:

```text
RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.75, 1.3333))
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=(-15, +15))
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3))
```

Augmentations are sampled during training only. `RandomResizedCrop`, rotation,
and color jitter always sample a transform; horizontal flip has probability
0.5; random erasing has probability 0.1. No vertical flip is used.

## Suitability for bird classification

- RandomResizedCrop (scale 0.6–1.0, aspect ratio 3:4–4:3) models framing and
  bird scale changes while retaining most of the subject.
- Horizontal flip (0.5) preserves species identity because left/right
  orientation is not a class attribute.
- Rotation ±15° models modest camera tilt; it does not create upside-down birds.
- ColorJitter uses modest brightness/contrast/saturation changes and hue ±0.02,
  preserving plumage cues better than aggressive color changes.
- RandomErasing (0.1; area 2–15%) encourages use of multiple visual cues;
  its small area limits damage to the class-defining bird appearance.

## Validation/test separation and normalization

Validation and test use only `Resize((224, 224))`, `ToTensor()`, and the same
ImageNet normalization. They contain no random crop, flip, rotation, jitter,
or erasing. Reapplying the evaluation transform to the same image produced
identical tensors in this audit: `True`.

Normalization is applied once after `ToTensor()` with mean
`(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. The visualization
denormalizes only for display; it does not alter training data.

## Visual evidence and limitations

The eight original/augmented pairs are saved at
`outputs/eda/augmentation_examples.png`. The examples are representative
samples, not a statistical estimate of all augmentation outcomes. Augmented
images are stochastic, so a different seed can produce different crops and
colors. No augmentation was added or changed as part of this audit.
