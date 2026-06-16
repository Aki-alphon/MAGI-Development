# HeatMAP — Edge AI & ML Preprocessing Evaluator

Dashboard and code for evaluating raw sensor preprocessing pipelines and machine learning classification thresholds (MAYA project telemetry).

---

## Running the Dashboard

1. Install requirements:
   ```bash
   pip install streamlit pandas numpy plotly
   ```
2. Launch the Streamlit dashboard:
   ```bash
   cd HeatMAP
   streamlit run visauls.py
   ```

---

## Features

### Preprocessing Pipeline Simulator
Simulates image pipeline transformations before feeding inputs to the `MobileNetV2` neural network.
* **Toggles**: Enables testing of image downsampling (to 224px), CLAHE contrast normalization, tensor float32 scaling (0.0 to 1.0), and data augmentation tilt.
* **Warnings**: Emits warnings for memory bounds (e.g. running 12MP raw images on a Raspberry Pi) and convergence risks (e.g. integer scaling).

### ML Model Evaluation
Interactive tool using a mock dataset of 100 validation samples (50 healthy, 50 diseased).
* **Threshold Slider**: Adjusts classification thresholds dynamically to update Accuracy, Precision, and Recall metrics.
* **Confusion Matrix Plot**: Plotly 2x2 scatter matrix visualizing True/False Positive/Negative distribution with custom layout jitter to prevent overlapping coordinates.

---

## Files

* **[visauls.py](file:///home/aki/Downloads/MAGI/HeatMAP/visauls.py)**: Main Streamlit interface.
* **[Heatmap_algo.ipynb](file:///home/aki/Downloads/MAGI/HeatMAP/Heatmap_algo.ipynb)**: Jupyter notebook containing the baseline heatmap algorithm and spectral calculations.
* **[Ass servo.SLDPRT](file:///home/aki/Downloads/MAGI/HeatMAP/Ass%20servo.SLDPRT)**: SolidWorks part assembly file for the joints.
