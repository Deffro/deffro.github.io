---
title: "Feature Selection for Machine Learning: 3 Categories and 12 Methods"
date: 2021-06-09
author_profile: true
excerpt: "Learn basic theory about the 3 types of feature selection in machine learning namely filters, wrappers, and embedders."
tags: [features, machine learning,]
header:
  image: "images/projects/fs.jpg"
  teaser: "images/teasers/fs.jpg"
categories:
  - Machine Learning
  - Feature Selection
mathjax: "true"
---

**The project is available online on [Towards Data Science](https://towardsdatascience.com/feature-selection-for-machine-learning-3-categories-and-12-methods-6a4403f86543)**.

Most of the content of this article is from my recent paper entitled:
“An Evaluation of Feature Selection Methods for Environmental Data”, available [here](https://www.sciencedirect.com/science/article/abs/pii/S1574954121000157) for anyone interested.

## The 2 approaches for Dimensionality Reduction
There are two ways to reduce the number of features, otherwise known as dimensionality reduction.

The first way is called feature extraction and it aims to transform the features and create entirely new ones based on combinations of the raw/given ones.
The most popular approaches are the Principle Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Multidimensional Scaling. However, the new feature space can hardly provide us with useful information about the original features.
The new higher-level features are not easily understood by humans, because we can not link them directly to the initial ones, making it difficult to draw conclusions and explain the variables.

The second way for dimensionality reduction is feature selection.
It can be considered as a pre-processing step and does not create any new features, but instead selects a subset of the raw ones, providing better interpretability.
Finding the best features from a significant initial number can help us extract valuable information and discover new knowledge.
In classification problems, the significance of features is evaluated as to their ability to resolve distinct classes.
The property which gives an estimation of each feature’s handiness in discriminating the distinct classes is called feature relevance.

Continue reading on [Towards Data Science](https://towardsdatascience.com/feature-selection-for-machine-learning-3-categories-and-12-methods-6a4403f86543).