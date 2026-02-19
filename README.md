# Cybercrime Intelligence System

This project analyzes fraud complaint parameters and predicts the most probable withdrawal hotspot using a trained Machine Learning model.

## Project Structure

- `train_model.py`: Script to train the Random Forest Classifier model.
- `app.py`: Streamlit application for the web interface.
- `cybercrime.csv`: Dataset used for training.
- `requirements.txt`: List of Python dependencies.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model and generate the necessary encoder files, run:

```bash
python train_model.py
```

This will create `cybercrime_model.pkl` and several `*_encoder.pkl` files.

## Running the Application

To start the web application, run:

```bash
streamlit run app.py
```

This will launch a local server and open the application in your default web browser. You can input crime details to predict the likely location of the fraud withdrawal.
