# Constructing Interpretable Prediction Models with 1D DNNs

Welcome to the GitHub repository the paper "Constructing Interpretable Prediction Models with 1D DNNs." This repository contains the code and resources necessary to replicate the experiments and analyses presented in the paper.

## Overview

This project presents a novel methodology for creating an interpretable prediction model for irregular ECG classification using features extracted by a one-dimensional deep neural network (1D DNN). Given the rising prevalence of cardiovascular disease, there is a critical need for models that provide transparent, clinically relevant predictions to enhance automated diagnostic tools.

We integrate these features into a simple logistic regression (LR) model to predict abnormal ECG patterns, demonstrating that the features align with clinical knowledge and enable reliable classification of conditions like atrial fibrillation (AF) and myocardial infarction (MI). Notably, our results show that the logistic regression model achieves similar predictive accuracy to more complex models, illustrating the effective integration of explainable artificial intelligence (XAI) with traditional regression techniques

## Scripts

The scripts provided include the Python modules developed for either pre-processing the data or performing the analysis.

1. The file `ECG_data.py` is a Python module containing all scripts developed for pre-processing.
2. The file `Main_Repository.py` includes all scripts developed for the analysis.
3. An `.ipynb` file is included to demonstrate how all analyses were performed.

## Data

Data are freely available at https://physionet.org/content/ecg-arrhythmia/1.0.0/ .

