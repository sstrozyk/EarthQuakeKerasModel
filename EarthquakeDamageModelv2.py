# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:45:04 2023
@author: sstro
EarthQuakeDamageModelv2.py
"""

# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


# Define CSV header for data import
CSV_HEADER = [
    # Building characteristics
    'building_id', 'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
    'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
    # Superstructure types
    'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
    'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
    'has_superstructure_timber', 'has_superstructure_bamboo',
    'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
    'has_superstructure_other',
    # Other building details
    'count_families', 'has_secondary_use', 'has_secondary_use_agriculture',
    'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution',
    'has_secondary_use_school', 'has_secondary_use_industry', 'has_secondary_use_health_post',
    'has_secondary_use_gov_office', 'has_secondary_use_use_police', 'has_secondary_use_other',
    # Ground characteristics
    'land_surface_condition_n', 'land_surface_condition_o', 'land_surface_condition_t',
    # Foundation, roof, floor types
    'foundation_type_h', 'foundation_type_i', 'foundation_type_r',
    'foundation_type_u', 'foundation_type_w', 'roof_type_n', 'roof_type_q',
    'roof_type_x', 'ground_floor_type_f', 'ground_floor_type_m',
    'ground_floor_type_v', 'ground_floor_type_x', 'ground_floor_type_z',
    # Other structural characteristics
    'other_floor_type_j', 'other_floor_type_q', 'other_floor_type_s',
    'other_floor_type_x', 'position_j', 'position_o', 'position_s',
    'position_t', 'plan_configuration_a', 'plan_configuration_c',
    'plan_configuration_d', 'plan_configuration_f', 'plan_configuration_m',
    'plan_configuration_n', 'plan_configuration_o', 'plan_configuration_q',
    'plan_configuration_s', 'plan_configuration_u',
    # Ownership and damage grades
    'legal_ownership_status_a', 'legal_ownership_status_r',
    'legal_ownership_status_v', 'legal_ownership_status_w',
    'damage_grade_1', 'damage_grade_2', 'damage_grade_3','instance_weight'
]


# Numeric feature names
NUMERIC_FEATURE_NAMES = [
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "count_floors_pre_eq", "age",
    "area_percentage", "height_percentage", "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick", "has_superstructure_timber",
    "has_superstructure_bamboo", "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered", "has_superstructure_other", "count_families",
    "has_secondary_use", "has_secondary_use_agriculture", "has_secondary_use_hotel",
    "has_secondary_use_rental", "has_secondary_use_institution", "has_secondary_use_school",
    "has_secondary_use_industry", "has_secondary_use_health_post", "has_secondary_use_gov_office",
    "has_secondary_use_use_police", "has_secondary_use_other"
]


# Categorical feature names
CATEGORICAL_FEATURE_NAMES = {
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status"
}


# Load and preprocess training data

def get_categorical_features_with_vocabulary(train_data):
    """
    Loads, preprocesses, and splits the dataset into training, validation, and test sets.
    Also handles one-hot encoding of categorical features and normalization of numeric features.
    
    Parameters:
    train_data: string
        csv of training data

    Returns: CATEGORICAL_FEATURES_WITH_VOCABULARY : list of lists
    list of each categorical feature and its vocabulary
    """
    # Pull vocabulary for categorical features from training data
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "land_surface_condition": train_data["land_surface_condition"].unique(),
        "foundation_type": train_data["foundation_type"].unique(),
        "roof_type": train_data["roof_type"].unique(),
        "ground_floor_type": train_data["ground_floor_type"].unique(),
        "other_floor_type": train_data["other_floor_type"].unique(),
        "position": train_data["position"].unique(),
        "plan_configuration": train_data["plan_configuration"].unique(),
        "legal_ownership_status": train_data["legal_ownership_status"].unique()
    }
    return CATEGORICAL_FEATURES_WITH_VOCABULARY


def load_data(train_data_url,train_labels_url,test_data_url):
    """
    Loads, preprocesses, and splits the dataset into training, validation, and test sets.
    Also handles one-hot encoding of categorical features and normalization of numeric features.
    
    Parameters:
    train_data_url: string
        URL of training data
    train_labels_url: string
        URL of training labels
    test_data_url: string
        URL of test data

    Returns:
    x_train, y_train, x_valid, y_valid, x_test: csv data sets from given urls for training and validation data sets
    """
    # Load data from CSV files
    train_data = pd.read_csv(train_data_url)
    labels_data = pd.read_csv(train_labels_url)
    test_data = pd.read_csv(test_data_url)

    # Merge training data with labels
    data = train_data.merge(labels_data, on="building_id")

    # Splitting data into training and validation sets
    train_df, valid_df = train_test_split(data, test_size=0.15, random_state=42)

    # One-hot encoding of categorical features
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURE_NAMES)
    valid_df = pd.get_dummies(valid_df, columns=CATEGORICAL_FEATURE_NAMES)
    test_data = pd.get_dummies(test_data, columns=CATEGORICAL_FEATURE_NAMES)

    # Ensure all columns in test_data are also in train_df and valid_df
    for column in set(train_df.columns) - set(test_data.columns):
        test_data[column] = 0
    test_data = test_data[train_df.columns]

    # Normalizing numeric features
    scaler = StandardScaler()
    train_df[NUMERIC_FEATURE_NAMES] = scaler.fit_transform(train_df[NUMERIC_FEATURE_NAMES])
    valid_df[NUMERIC_FEATURE_NAMES] = scaler.transform(valid_df[NUMERIC_FEATURE_NAMES])
    test_data[NUMERIC_FEATURE_NAMES] = scaler.transform(test_data[NUMERIC_FEATURE_NAMES])

    # Separate features and target variable
    x_train = train_df.drop(['damage_grade'], axis=1)
    y_train = to_categorical(train_df['damage_grade'])
    x_valid = valid_df.drop(['damage_grade'], axis=1)
    y_valid = to_categorical(valid_df['damage_grade'])
    x_test = test_data.drop(['building_id'], axis=1)  # Assuming 'building_id' is not a feature

    return x_train, y_train, x_valid, y_valid, x_test


def create_model(input_shape, num_classes):
    """
    Creates and compiles a neural network model.

    Parameters:
    input_shape: tuple
        The shape of the input features. It does not include the batch size.
        For example, for a dataset with 10 features, input_shape would be (10,).
    num_classes: int
        The number of classes in the output layer. This should be equal to the number of unique labels
        in the classification task.

    Returns:
    model: keras.Model
        A compiled Keras sequential model with the specified architecture.
    """

    # Create a Sequential model instance
    model = keras.Sequential([
        # First dense layer with 128 neurons and ReLU activation. 
        # 'input_shape' is required only for the first layer.
        Dense(128, activation='relu', input_shape=input_shape),

        # Dropout layer to reduce overfitting by randomly setting a fraction of input units to 0 at each update during training
        Dropout(0.5),

        # Second dense layer with 64 neurons and ReLU activation
        Dense(64, activation='relu'),

        # Another dropout layer for regularization
        Dropout(0.5),

        # Output layer with 'num_classes' neurons, one for each class, with softmax activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    # Compiling the model with 'adam' optimizer, 'categorical_crossentropy' loss (common choice for multi-class classification),
    # and tracking 'accuracy' metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Evaluating the model
def evaluate_model_performance_with_loss(model, x_test, y_true):
    """
    Evaluates the performance of the trained model on test data.

    Parameters:
    model: Trained Keras model.
    X_test: Test features.
    y_true: True labels for the test data.
    """
    loss, accuracy = model.evaluate(x_test, y_true)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return accuracy,loss

def predict_values(model, x_test):
    """
    Predicts the class labels for the test data.

    Parameters:
    model: Trained Keras model.
    X_test: Test features.

    Returns:
    Array of predicted class labels.
    """
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def evaluate_model_performance(model, x_test, y_true):
    """
    Evaluates the performance of the trained model on test data.

    Parameters:
    model: Trained Keras model.
    X_test: Test features.
    y_true: True labels for the test data.
    """
    # Predicting labels for the test set
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)

    #More detailed performance analysis can be added later

    print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy



def get_unique_model_name(base_name="model"):
    """
    Generates a unique model name by appending the current timestamp.

    Parameters:
    base_name: str
        Base name for the model.

    Returns:
    unique_model_name: str
        A unique model name with a timestamp.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_model_name = f"{base_name}_{current_time}"
    return unique_model_name





def train_and_evaluate(train_data_url,train_labels_url,test_data_url):
    """
    Trains the model and evaluates its performance.
    
    Parameters:
    train_data_url: string
        URL of training data
    train_labels_url: string
        URL of training labels
    test_data_url: string
        URL of test data
        
    returns:
        model : Keras model
        model_file_name: str
            A unique model fiile name with a timestamp.
        model_accuracy: num
            accuracy of model
        model_loss: num
            loss of model
    """
    # Load the data
    x_train, y_train, x_valid, y_valid, x_test = load_data(train_data_url,train_labels_url,test_data_url)

    # Create the model
    model = create_model(x_train.shape[1:], y_train.shape[1])

    # Training the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, batch_size=32, callbacks=[early_stopping])

    # Saving the model
    model_file_name = get_unique_model_name("earthquake_damage_model") + ".h5"
    model.save(model_file_name)

    # Evaluate the model
    model_accuracy, model_loss = evaluate_model_performance_with_loss(model, x_valid, y_valid)

    # Predict values
    predicted_classes = predict_values(model, x_test)

    return model, model_file_name, model_accuracy, model_loss

def main():
    """
    Main function to run the script.
    """
    # Data file paths
    train_data_url = "C:/Users/sstro/Documents/Python Scripts/EarthquakeDamData/train_values.csv"
    train_labels_url = "C:/Users/sstro/Documents/Python Scripts/EarthquakeDamData/train_labels.csv"
    test_data_url = "C:/Users/sstro/Documents/Python Scripts/EarthquakeDamData/test_values.csv"
    model, model_file_name, model_accuracy, model_loss = train_and_evaluate(train_data_url,train_labels_url,test_data_url)
    model.save(model_file_name)
    print(model_accuracy)
    print(model_loss)

    # Additional code if needed for pre saved models
    # loaded_model = keras.models.load_model(model_file_name)
    # predictions = loaded_model.predict(X_new_data) # For new data predictions

if __name__ == "__main__":
    main()
