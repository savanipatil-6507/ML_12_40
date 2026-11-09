# Penguin Species Prediction using Decision Tree

## About the Project
This project uses Machine Learning to predict the species of penguins based on their physical measurements.

A Decision Tree Classifier is used to classify penguins into three species:
- Adelie
- Chinstrap
- Gentoo

## Dataset
The project uses the `penguins.csv` dataset.

The features used for prediction are:
- Bill Length
- Bill Depth
- Flipper Length
- Body Mass

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Pickle

## Machine Learning Algorithm
**Decision Tree Classifier**

The model is trained using the penguin dataset and evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

## Project Structure

```text
Penguin-Species-Prediction/
│
├── penguins.csv
├── train_model.py
├── penguin_species_model.pkl
└── README.md
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/savanipatil-6507/ML_12_40.git
```

2. Install the required libraries:

```bash
pip install pandas scikit-learn
```

3. Run the Python file:

```bash
python train_model.py
```

4. The model will be trained, evaluated, and saved as `penguin_species_model.pkl`.

## Output
The program displays:
- Dataset preview and shape
- Model training status
- Accuracy, Precision, Recall, and F1-Score
- Predicted penguin species
- Model saved successfully message

