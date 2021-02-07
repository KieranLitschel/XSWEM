# XSWEM

[![Build Status](https://img.shields.io/travis/KieranLitschel/XSWEM/main.svg?label=main)](https://travis-ci.org/KieranLitschel/XSWEM) [![Build Status](https://img.shields.io/travis/KieranLitschel/XSWEM/develop.svg?label=develop)](https://travis-ci.org/KieranLitschel/XSWEM)

A simple and explainable deep learning model for NLP implemented in TensorFlow.

Based on SWEM-max as proposed by Shen et al. in [Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms, 2018](https://arxiv.org/pdf/1805.09843.pdf).

This package is currently in development. The purpose of this package is to make it easy to train and explain SWEM-max. 

You can find demos of the functionality we have implemented in the [notebooks](https://github.com/KieranLitschel/XSWEM/blob/main/notebooks) directory of the package. Each notebook has a badge that allows you to run it yourself in Google Colab. We will add more notebooks as new functionality is added.

For a demo of how to train a basic SWEM-max model see [train_xswem.ipynb](https://github.com/KieranLitschel/XSWEM/blob/main/notebooks/train_xswem.ipynb).

## Local Explanations

We are currently implementing some methods we have developed for local explanations.

### local_explain_most_salient_words

So far we have only implemented the local_explain_most_salient_words method. This method extracts the words the model has learnt as most salient from a given input sentence. Below we show an example of this method using a sample from the [ag_news](https://huggingface.co/datasets/viewer/?dataset=ag_news) dataset. This method is explained in more detail in the [local_explain_most_salient_words.ipynb](https://github.com/KieranLitschel/XSWEM/blob/main/notebooks/local_explain_most_salient_words.ipynb) notebook.

![local_explain_most_salient_words.png](https://github.com/KieranLitschel/XSWEM/blob/main/resources/images/local_explain_most_salient_words.png)

## Global Explanations

We have implemented the global explainability method proposed in section 4.1.1 of the original paper. You can see a demo of this method in the notebook [global_explain_embedding_components.ipynb](https://github.com/KieranLitschel/XSWEM/blob/main/notebooks/global_explain_embedding_components.ipynb).

## How to install

This package is hosted on [PyPI](https://pypi.org/project/xswem/) and can be installed using pip.

```
pip install xswem
```
