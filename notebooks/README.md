"""
Diabetic Retinopathy Detection - Notebook Renaming and Ordering
===============================================================

This file serves as a placeholder to indicate the new order of notebooks in the pipeline.
Use this file to guide your execution flow.

Notebook Order:
1. 01_eda_and_preprocessing.ipynb : Initial data exploration and preprocessing checks.
2. 02_quality_control.ipynb       : (New) Assessment of image quality and creation of filtered manifest.
3. 03_model_training.ipynb        : (Planned) Training the model using the (optionally) filtered dataset.
4. 04_model_evaluation.ipynb      : (Renamed from 02) Evaluation of model performance.
5. 05_inference_demo.ipynb        : (Renamed from 03) Inference on single images.

Note:
- 'src/data/make_dataset.py' now includes 'load_filtered_dataset' helper.
- 'src/config.py' now includes 'QUALITY_THRESHOLD'.
"""
