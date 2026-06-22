# Fruit-Leaf Species Classification — A Model-Selection Study

**A controlled comparison of a convolutional neural network against a classical Random Forest baseline on a ten-class fruit-leaf image dataset. The central question is not "how accurate a CNN can be built" but the prior one every practitioner should answer first: whether the task warrants deep learning at all. On this dataset it does not — a Random Forest on raw pixels attains 92.6% test accuracy against the CNN's 73.5% — and the repository is organised around establishing that result rigorously rather than around the model that wins.**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)

---

## Motivation

A common failure mode in applied machine learning is to reach for a high-capacity model before establishing that a simpler one is insufficient. Deep convolutional networks are the default choice for image classification, but they are data-hungry, slower to train, and harder to interpret than classical alternatives; their advantages are realised only under conditions: large datasets, strong spatial structure, or transfer learning — that do not hold for every task labelled "image classification".

This repository treats a ten-class fruit-leaf classification problem as a test of that principle. A small CNN is trained from scratch and then evaluated against a Random Forest baseline fitted on the same data. The comparison is deliberately constructed to be fair, the disparity between the two models is investigated rather than asserted, and a data-augmentation experiment is conducted to give the deep model a reasonable opportunity to close the gap. The conclusion — that classical machine learning is the appropriate tool for this dataset — is the principal contribution of the repository.

---

## Data

The dataset is the **Multi-Class Fruit Leaf Classification** collection from Mendeley Data ([DOI 10.17632/4gxzx6h7gv.2](https://data.mendeley.com/datasets/4gxzx6h7gv/2), CC BY 4.0). It comprises **3,173 images across ten balanced classes** (302–343 images per class): *Aegle marmelos, Black plum, Custard Apple, Guava, Jackfruit, Lotkon, Lychee, Mango, Plum,* and *Star Fruit*. Each class occupies its own directory, making the collection directly compatible with `torchvision.datasets.ImageFolder`, and the task is whole-image single-label classification.

The source images are high resolution and were resized to 64×64 pixels for this study to keep training tractable on CPU. The raw data (approximately 9.73 GB) is not redistributed here; it should be downloaded from the link above and placed at `data/fruit_leaves/` to reproduce the results.

---

## Method

A compact convolutional network (`model/small_cnn.py`) was trained as the candidate deep model: two convolutional blocks (`Conv(3→16)` and `Conv(16→32)`, each followed by ReLU and 2×2 max-pooling) feeding two fully connected layers (`Linear(8192→64) → ReLU → Linear(64→10)`). It was optimised with cross-entropy loss and Adam (learning rate 0.001) over eight epochs on an 80/20 train–test split.

A **Random Forest** (100 trees, scikit-learn) was fitted as the baseline, taking each image flattened to a 12,288-dimensional pixel vector. The baseline serves the essential diagnostic role of indicating whether the deep model earns its additional complexity: a classical model trained on identical inputs establishes the accuracy that must be exceeded before deep learning can be justified.

The disparity between the two models was then examined along three lines of evidence: the CNN's train–test accuracy gap as a measure of overfitting; an explicit check that the train and test partitions do not overlap and together exhaust the dataset, confirming the baseline's score is not an artefact of leakage; and per-class confusion matrices for both models. Finally, the CNN was retrained with data augmentation applied to the training set only (random horizontal flips, rotation up to 20°, and brightness/contrast jitter), with the test set left unaltered, to test whether augmentation could recover the deficit.

---

## Results

| Model | Test accuracy | Train accuracy | Train–test gap |
|---|---|---|---|
| CNN (8 epochs, no augmentation) | 73.5% | 87.1% | +13.6% |
| CNN (8 epochs, with augmentation) | 67.1% | 63.0% | −4.1% |
| **Random Forest (100 trees)** | **92.6%** | 100.0% | — |

The Random Forest outperforms the from-scratch CNN by approximately nineteen percentage points on held-out data. The leakage check confirms the partitions are disjoint and complete, so the baseline's margin reflects genuine generalisation rather than contamination. The confusion matrices localise the CNN's errors to a small number of visually similar species — most prominently *Black plum* and *Mango* — and show that both models confuse the same look-alike classes, indicating the difficulty is intrinsic to the data rather than specific to one model.

![CNN and Random Forest confusion matrices](figures/confusion_matrices.png)

The augmentation experiment is instructive in its own right. Augmentation removed the CNN's overfitting entirely — the train–test gap moved from +13.6% to −4.1%, the test set now being easier than the deliberately perturbed training set — but eight epochs were insufficient for the model to recover accuracy on the harder augmented inputs, so overall test accuracy fell to 67.1% and still did not approach the baseline. The intervention behaved exactly as theory predicts; it simply did not change the conclusion.

---

## Interpretation

This result should not be read as a general claim that Random Forests outperform convolutional networks. It is a claim about the match between a model and a dataset. Several properties of this dataset favour the classical model: the sample is small (roughly 3,000 images, where convolutional networks typically require substantially more), and the discriminative signal lies largely in colour and texture — hue, sheen, and venation — which a Random Forest reads effectively from raw pixel values without needing to learn spatial feature hierarchies from scratch.

Conditions under which the deep model would be expected to prevail are precisely those absent here: a markedly larger training set, a task with strong spatial dependencies, or the use of a pretrained convolutional backbone via transfer learning. Identifying which regime a given problem occupies, and selecting the model accordingly, is the judgement this study is intended to demonstrate.

---

## Repository structure

```
fruit-leaf-classification-cnn/
├── eda.ipynb              # the full study, read top to bottom
├── model/
│   └── small_cnn.py       # the SmallCNN architecture
├── figures/
│   └── confusion_matrices.png
├── train.py               # standalone CNN training script
├── evaluate.py            # standalone CNN evaluation script
├── requirements.txt
└── README.md
```

The complete analysis resides in `eda.ipynb`, which proceeds in the order of the argument above: data loading and inspection, CNN training and evaluation, the Random Forest baseline, the three-part diagnosis, and the augmentation experiment. The `train.py` and `evaluate.py` scripts reproduce the core CNN training and evaluation steps for use outside the notebook.

## Reproduction

```bash
pip install -r requirements.txt
# download the dataset (link above) into data/fruit_leaves/
jupyter notebook eda.ipynb     # run cells in order
```

---

## Limitations and extensions

The deep model is trained from scratch; a pretrained backbone (for example a ResNet fine-tuned on this dataset) would give convolutional networks their strongest opportunity and is the natural next comparison. The augmented model would also benefit from a longer training schedule to test whether it eventually recovers and surpasses the un-augmented configuration. Finally, reporting per-class precision and recall alongside aggregate accuracy would better characterise performance on the small set of visually similar species that dominate both models' errors.

---

*Dataset: Multi-Class Fruit Leaf Classification (Mendeley Data, CC BY 4.0). This repository was developed to practise disciplined model comparison and the honest reporting of results, including those that contradict the initial hypothesis.*
