# Traffic Accident Fatality Prediction

## Project Overview
This project implements a binary classification model that predicts whether a person involved in a traffic accident will suffer a fatal injury or not. The model uses neural networks to analyze various factors related to the person, vehicle, and accident circumstances.

## Dataset Description
The model uses the NHTSA crash dataset, specifically the person-level data which includes information about individuals involved in traffic accidents. The dataset contains:
- 455,336 person records
- Features covering demographic information, safety equipment usage, and accident circumstances
- Binary target variable: Fatal (1) vs Non-Fatal (0) injury
- Class distribution: 
  - Non-Fatal: 253,122 (55.6%)
  - Fatal: 202,214 (44.4%)

## Model Architecture
The model is a deep neural network implemented using TensorFlow and Keras with the following architecture:

```
Model: Sequential
_________________________________________________________________
Layer (type)                    Output Shape         Param #
=================================================================
Dense (64 neurons, ReLU)        (None, 64)           10,112
Dropout (0.3)                   (None, 64)           0
Dense (32 neurons, ReLU)        (None, 32)           2,080
Dropout (0.3)                   (None, 32)           0
Dense (16 neurons, ReLU)        (None, 16)           528
Dense (1 neuron, Sigmoid)       (None, 1)            17
=================================================================
Total params: 12,737
```

- **Input Layer**: 157 features (after preprocessing)
- **Hidden Layers**: Three dense layers with ReLU activation
- **Regularization**: Dropout layers (0.3 rate) to prevent overfitting
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam with learning rate of 0.001

## Feature Selection and Preprocessing
The model uses the following features:

### Person Characteristics
- AGE
- SEX
- PER_TYP (Person type: driver, passenger, pedestrian, etc.)

### Safety Factors
- SEAT_POS (Seating position)
- REST_USE (Restraint system use)
- AIR_BAG (Airbag deployment)

### Risk Factors
- EJECTION (Whether person was ejected from vehicle)
- DRINKING (Alcohol involvement)
- DRUGS (Drug involvement)

### Time Factors
- MONTH, DAY, HOUR, MINUTE (Time of accident)

### Preprocessing Steps
1. **Missing values**: Rows with missing values in essential columns are removed
2. **Categorical features**: One-hot encoded using scikit-learn's OneHotEncoder
3. **Numeric features**: Standardized using scikit-learn's StandardScaler
4. **Feature expansion**: After preprocessing, the 13 original features expand to 157 features

## Training Process
- **Train-Test Split**: 80% training, 20% testing with stratified sampling
- **Validation**: 20% of training data used for validation
- **Batch Size**: 64
- **Epochs**: 20
- **Early Stopping**: Not implemented, but could be added for further optimization

## Performance Metrics and Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 81.02% |
| Precision | 81.51% |
| Recall    | 74.06% |
| F1 Score  | 77.61% |
| AUC       | 0.88   |

The model achieves over 81% accuracy in predicting fatal vs. non-fatal outcomes. The precision (81.51%) is higher than recall (74.06%), indicating that the model is more conservative in predicting fatalities but more reliable when it does predict a fatal outcome.

## Visualizations
The model generates several visualizations to help understand its performance:

1. **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
2. **Training History**: Plots of loss and accuracy during training
3. **ROC Curve**: Shows the trade-off between true positive rate and false positive rate
4. **Metrics Summary**: Bar chart of key performance metrics

## File Structure
```
fatality_binary_classification/
├── data/
│   └── cleaned_combined_person.csv
├── output/
│   ├── models/
│   │   └── fatality_classification_model.h5
│   └── plots/
│       ├── confusion_matrix.png
│       ├── training_history.png
│       ├── roc_curve.png
│       └── metrics_summary.png
├── fatality_classifier.py
└── README.md
```

## Running the Model
To run the model:
```
cd fatality_binary_classification
python fatality_classifier.py
```

The script will:
1. Load and preprocess the data
2. Train the model
3. Evaluate performance
4. Generate visualizations
5. Save the trained model

## Dependencies
- Python 3.6+
- TensorFlow 2.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Potential Improvements
- Implement class balancing techniques
- Add feature importance analysis
- Try different model architectures
- Implement hyperparameter tuning
- Add more features or feature engineering
- Implement cross-validation
- Try ensemble methods
