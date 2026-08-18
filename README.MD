# PYTORCH ESSENTIALS
This repo covers the essentials functionalities/concepts of the Pytorch library along with some implementations. This is to be used for learning purposes and not as the end production code.

# Interesting Modules
## src/pytorch
Provides various tutorials on pytorch - compiled mostly from the Coursera's 'Pytorch for Deep Learning' specialization course by Laurence Moroney. It contains fundamentals to advanced Pytorch topics.

## src/dnn/transformer 
This is a basic setup for a language based transformer, closely based on the seminal paper "Attention is All You Need". It is highly annotated with examples, maths, and comments and is to be used as a 
learning tool to understand transformers, tokenizer, and the attention framework. Studying the paper side by side while playing around with this
is highly recommended. Still a WIP...the model is not ready yet.

## src/pyessentials
Contains some essentials and often overlooked basic topics in Python like classes,concurreny, decorator and numbers.

## src/vision
Includes some modules on AI based vision models like point detectors, examples of torch data handling, point detectors etc. There is some work in progress here.

## Setup
Create a Python venv (>=3.12) environment and then activate the environment. 

```bash
pytorch-essentials> python -m venv .venv
pytorch-essentials> source .venv/bin/activate
```

Install the requirements

```bash
(.venv)pytorch-essentials> pip install -r requirements.txt
```

Helpful Pytorch Links
Documentation: https://docs.pytorch.org/docs/stable/index.html
All Datasets: https://docs.pytorch.org/vision/stable/datasets.html
