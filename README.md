# Machine Learning Models

Small from-scratch machine-learning projects exploring decision trees,
perceptrons, and classification evaluation in Python.

## Projects

### Titanic survival prediction

Implemented a depth-limited binary decision tree using weighted Gini impurity
to select feature splits. Preprocessing includes encoding passenger sex,
filling missing ages with the median, and extracting cabin deck. The model
creates a Kaggle submission using passenger features such as sex, ticket
class, age, and fare.

- **Recorded Kaggle accuracy:** 77.99%
- **Recorded leaderboard position:** 3,387 of 11,351
- **Notebook:** [`DecisionTree-titanic.ipynb`](decision_tree_kaggle_titanic/DecisionTree-titanic.ipynb)

### MNIST handwritten digit classification

Implemented and compared linear classifiers for the ten MNIST digit classes.
Images are normalized, binarized, and flattened from 28 × 28 pixels into
784-element inputs. Experiments include independent perceptrons and a
multiclass linear classifier trained with stochastic gradient descent,
softmax, and cross-entropy loss.

- **Best recorded test accuracy:** 90.22%
- **Recorded precision / recall / F1:** 0.9034 / 0.9022 / 0.9015
- **Notebook:** [`MulticlassClassifier.ipynb`](linear_perceptron/MulticlassClassifier.ipynb)
- **Experiment comparison:** [`report.ipynb`](report.ipynb)

## Supporting evaluation utilities

[`metrics.py`](metrics.py) computes per-class and aggregated precision,
recall, and F-scores from a confusion matrix, with micro, macro, and weighted
averaging options. It also includes specificity calculations.

## Tools

Python, NumPy, pandas, Matplotlib, and Hugging Face Datasets.
