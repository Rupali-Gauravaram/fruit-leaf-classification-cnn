# Fruit-Leaf Species Classification: A Model-Selection Study

**I compared a small convolutional neural network (CNN) with a Random Forest on a ten-class fruit-leaf image dataset. The question was not "how accurate can I make a CNN?" but "does this task need deep learning at all?" On this dataset it does not: the Random Forest reached 92.6% test accuracy and the CNN reached 73.5%. This repository shows how I tested that result.**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)

---

## Motivation

It is easy to reach for a deep learning model before checking whether a simpler one would do. CNNs are the usual choice for image classification, but they need a lot of data, take longer to train, and are harder to interpret. They do best with large datasets, strong spatial patterns, or transfer learning, and not every image task has those.

This project tests that idea on a fruit-leaf classification problem. I trained a small CNN from scratch and compared it with a Random Forest trained on the same data. I kept the comparison fair, looked into why the two models performed differently, and tried data augmentation to give the CNN a fair chance to catch up. The main finding is that classical machine learning is the better tool for this dataset.

---

## Data

The dataset is the **Multi-Class Fruit Leaf Classification** collection from Mendeley Data ([DOI 10.17632/4gxzx6h7gv.2](https://data.mendeley.com/datasets/4gxzx6h7gv/2), CC BY 4.0). It has **3,173 images in ten balanced classes** (302 to 343 images per class): *Aegle marmelos, Black plum, Custard Apple, Guava, Jackfruit, Lotkon, Lychee, Mango, Plum,* and *Star Fruit*. Each class has its own folder, so it works directly with `torchvision.datasets.ImageFolder`. Each image has one label.

The original images are high resolution. I resized them to 64×64 pixels so training would run on a CPU. The raw data (about 9.73 GB) is not included here. To reproduce the results, download it from the link above and place it at `data/fruit_leaves/`.

---

## Method

**The CNN** (`model/small_cnn.py`) has two convolutional blocks (`Conv(3→16)` and `Conv(16→32)`, each followed by ReLU and 2×2 max-pooling) and two fully connected layers (`Linear(8192→64) → ReLU → Linear(64→10)`). I trained it with cross-entropy loss and the Adam optimiser (learning rate 0.001) for eight epochs on an 80/20 train-test split.

**The baseline** is a Random Forest (100 trees, scikit-learn). Each image is flattened into a vector of 12,288 pixel values. The baseline answers a simple question: is the deep model worth its extra complexity? If a classical model on the same inputs does better, deep learning is not justified here.

I then looked at the difference between the two models in three ways:

1. The CNN's gap between train and test accuracy, as a measure of overfitting.
2. A check that the train and test sets do not overlap and together cover the whole dataset, so the baseline's score is not caused by data leakage.
3. Confusion matrices for both models, to see which classes each one gets wrong.

Finally, I retrained the CNN with data augmentation on the training set only (random horizontal flips, rotation up to 20°, and brightness and contrast changes). The test set was left unchanged.

---

## Results

| Model | Test accuracy | Train accuracy | Train-test gap |
|---|---|---|---|
| CNN (8 epochs, no augmentation) | 73.5% | 87.1% | +13.6% |
| CNN (8 epochs, with augmentation) | 67.1% | 63.0% | −4.1% |
| **Random Forest (100 trees)** | **92.6%** | 100.0% | n/a |

The Random Forest beats the CNN by about nineteen percentage points on the test set. The leakage check confirmed that the train and test sets are separate and complete, so this is a real difference. The confusion matrices show that most of the CNN's errors come from a few species that look alike, mainly *Black plum* and *Mango*. Both models confuse the same pairs, which suggests the difficulty comes from the data and not from one model.

![CNN and Random Forest confusion matrices](figures/confusion_matrices.png)

The augmentation experiment was useful too. It removed the CNN's overfitting: the train-test gap went from +13.6% to −4.1%, because the augmented training images are harder than the test images. But eight epochs were not enough for the model to learn from the harder images, so test accuracy dropped to 67.1%, still well below the baseline. Augmentation worked as expected. It just did not change the conclusion.

---

## Interpretation

This does not mean Random Forests are better than CNNs in general. It means this model suits this dataset. Two things favour the classical model here:

- The dataset is small (about 3,000 images). CNNs usually need much more.
- The useful signal is mostly colour and texture, which a Random Forest can read from raw pixel values without learning spatial features from scratch.

A CNN would be expected to win with a much larger training set, a task with strong spatial patterns, or a pretrained model used through transfer learning. Working out which situation a problem is in, and choosing the model to match, is what this study is meant to show.

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

The full analysis is in `eda.ipynb`, in the same order as above: loading and inspecting the data, training and evaluating the CNN, the Random Forest baseline, the three checks, and the augmentation experiment. `train.py` and `evaluate.py` run the CNN training and evaluation outside the notebook.

## Reproduction

```bash
pip install -r requirements.txt
# download the dataset (link above) into data/fruit_leaves/
jupyter notebook eda.ipynb     # run cells in order
```

---

## Limitations and next steps

- The CNN is trained from scratch. A pretrained model (for example a ResNet fine-tuned on this dataset) would give deep learning its best chance, and is the natural next comparison.
- The augmented model could be trained for more epochs, to see whether it eventually catches up.
- Reporting precision and recall per class, as well as overall accuracy, would describe performance on the look-alike species more clearly.

---

*Dataset: Multi-Class Fruit Leaf Classification (Mendeley Data, CC BY 4.0). I built this project to practise comparing models carefully and reporting results honestly, including results that went against what I expected.*
