# CogSci Thesis - Probing Language Models

This repository contains code for a cognitive science thesis focused on probing the internal representations of the Mistral language model. The project investigates how psychological dimensions are encoded within the model's activations.

## Project Overview

[cite_start]This project explores the relationship between the internal states of a large language model and psychological concepts. [cite_start]By analyzing the model's activation layers, it aims to uncover how concepts like emotion, concreteness, and others are represented.

The experiment involves:
* [cite_start]Extracting layer activations from the Mistral model for specific words.
* [cite_start]Aligning these activations with psycholinguistic data from the Glasgow Norms.
* [cite_start]Training probing models to predict psychological dimensions from model activations.
* [cite_start]Analyzing the results across different model layers to understand where and how this information is encoded.

## Key Features

* [cite_start]**Data Import**: A data pipeline imports and processes large activation files using Dask and Pandas. [cite_start]It can gather distributed Parquet files into a single dataset.
* [cite_start]**Glasgow Norms Integration**: The system loads and enriches the Glasgow Norms dataset. [cite_start]It adds features for part-of-speech tagging and identifying polysemic words.
* [cite_start]**Model Management**: Manages the Mistral model, including loading the model and tokenizer, generating text, and clearing memory resources.
* **Probing**: A system for conducting probing experiments, which includes:
    * [cite_start]Data preprocessing with standardization and dimensionality reduction using PCA.
    * [cite_start]Execution of probing tasks using machine learning models such as Ridge, MLPRegressor, and GradientBoostingRegressor.
    * [cite_start]Systematic evaluation across different model layers and psychological dimensions.
* [cite_start]**Visualization**: Generates plots to visualize results, including R² scores across layers and detailed prediction plots for each psychological dimension.

## Directory Structure

```
└── cogsci/
    [cite_start]├── activations/      # Manages the import of model activation data. 
    ├── ai/               # Handles interactions with the AI model. 
    ├── glasgow_norms/    # Loads and processes the Glasgow Norms dataset. 
    ├── notebooks/        # For experiments and analysis. 
    └── probing/          # Contains the core logic for the probing analysis. 
```
