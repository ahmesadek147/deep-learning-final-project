# P5 Error Analysis

## Measured evaluation

Computed from `outputs/evaluation/p5/classification_report.json` and `confusion_matrix.npy` on the 5,794-image held-out CUB test split. No training or validation images were used.

### Top 10 classes by F1

| Class | F1 | Support |
|---|---:|---:|
| Brown Pelican | 0.8814 | 30 |
| White Pelican | 0.8780 | 20 |
| Green Violetear | 0.8727 | 30 |
| Heermann Gull | 0.8727 | 30 |
| Clark Nutcracker | 0.8710 | 30 |
| Geococcyx | 0.8710 | 30 |
| Green Jay | 0.8679 | 27 |
| European Goldfinch | 0.8621 | 30 |
| Pied billed Grebe | 0.7937 | 30 |
| Dark eyed Junco | 0.7925 | 30 |

### Bottom 10 classes by F1

| Class | F1 | Support |
|---|---:|---:|
| Common Tern | 0.0000 | 30 |
| Fish Crow | 0.0000 | 30 |
| Long tailed Jaeger | 0.0000 | 30 |
| Rusty Blackbird | 0.0000 | 30 |
| Western Gull | 0.0000 | 30 |
| Herring Gull | 0.0513 | 30 |
| Western Wood Pewee | 0.0526 | 30 |
| American Crow | 0.0571 | 30 |
| Pelagic Cormorant | 0.0571 | 30 |
| Mockingbird | 0.0606 | 30 |

### Most frequent confusions

| True class | Predicted class | Count |
|---|---|---:|
| Artic Tern | Forsters Tern | 24 |
| Common Tern | Forsters Tern | 23 |
| Western Gull | California Gull | 22 |
| Glaucous winged Gull | California Gull | 20 |
| Caspian Tern | Forsters Tern | 20 |
| Herring Gull | California Gull | 18 |
| Ruby throated Hummingbird | Anna Hummingbird | 15 |
| Pelagic Cormorant | Brandt Cormorant | 15 |
| Least Tern | Forsters Tern | 14 |
| Elegant Tern | Forsters Tern | 14 |
| Scarlet Tanager | Summer Tanager | 13 |
| Ring billed Gull | California Gull | 13 |
| Red eyed Vireo | Warbling Vireo | 13 |
| Whip poor Will | Chuck will Widow | 12 |
| Rufous Hummingbird | Anna Hummingbird | 12 |

The dominant measured error groups are visually related seabirds: terns (especially Forster's Tern), gulls (especially California Gull), cormorants, and hummingbirds. Other repeated pairs include tanagers, vireos, sparrows, and albatross/gull images.

## Grad-CAM quality review

All eight panels were inspected. Each has three labeled columns—Original, Grad-CAM heatmap, and Overlay—and a title containing true class, predicted class, and confidence. Correct panels use green titles; wrong panels use red titles.

Visual interpretation: most reviewed heatmaps are nearly uniform dark blue, so they provide weak localization evidence. One wrong example shows a concentrated hotspot in the lower-left background rather than on the bird, suggesting a background-driven decision. These are qualitative observations, not additional metrics.

## Limitations

Grad-CAM is a coarse attribution map from the final ResNet convolutional block, not a segmentation mask. The panel sample is four correct and four incorrect predictions, not a statistically representative summary of all 5,794 images. Class-name similarity judgments are visual/domain interpretations of the measured confusion pairs.
