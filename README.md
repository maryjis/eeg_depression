# eeg_depression

The repository contains machine learning models for classification of Major Depression Disorder patients from healthy controls.


The repository provides two approaches: a standard feature-extracted approach and a deep learning approach.
Feature extraction and preparing data were made with https://github.com/ledovsky/eeg-research 

Deep learning approach includes 3 notebooks with such models:
  - 3D Autoencoder on spectrum EEG data
  - 2D Autoencoder on spectrum EEG data
  - 2D CNN model

Standard feature-extracted approach  includes notebook with training different ml models ( Random Forest, Logistic Regression, KNN, Gradient Boosting etc.) , notebook with extracting important features recieved on the best model and notebook with attemt to use transfer learning from one eeg dataset to another
