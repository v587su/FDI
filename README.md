# Feedback Data Injection
This repository contains the code for the paper "FDI: Attack Neural Code Generation Systems through User Feedback Channel"

## Data

The data for the prompt injection experiment is available in the `prompt_search/dataset` folder, while the data for the continual learning experiments can be downloaded from this [link](https://huggingface.co/datasets/code-search-net/code_search_net).

## Usage

For the continual learning experiments, the running scripts are available in the `continual_learning/scripts`.
Please revise the scripts to set the correct paths to the data.

For the prompt injection experiments, it can be run using the following command:

```python run.py```
