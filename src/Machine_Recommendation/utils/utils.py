import os
import sys
import pickle
import numpy as np
from src.Machine_Recommendation.logger import logging
from src.Machine_Recommendation.exception import customexception

# Assuming you'll use a clustering metric, I'm importing silhouette_score
from sklearn.metrics import silhouette_score 

import os
import pickle


model = pickle.load(open('artifacts/model.pkl','rb'))
book_name = pickle.load(open('artifacts/book_name.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (any): The Python object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Assuming you have a logging and customexception setup
        logging.info(f"Exception occurred in save_object: {e}")  
        raise customexception(e, sys) from e

# Recommendation System (Modified for clustering)
def evaluate_model(X_train, X_test, models):
    """
    Evaluates clustering models using silhouette score.

    Args:
        X_train: Training data features.
        X_test: Testing data features.
        models (dict): A dictionary of models where keys are model names 
                       and values are model instances.

    Returns:
        dict: A dictionary with model names as keys and silhouette scores 
              as values.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train) 

            # Predict cluster labels for training and testing data
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)

            # Calculate silhouette scores
            train_score = silhouette_score(X_train, train_labels)
            test_score = silhouette_score(X_test, test_labels)

            report[model_name] = {
                "train_score": train_score,
                "test_score": test_score
            }

        return report

    except Exception as e:
        logging.info(f'Exception occurred during model evaluation: {e}')
        raise customexception(e, sys) from e

def load_object(file_path):
    """
    Loads a Python object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        any: The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info(f'Exception occurred in load_object: {e}')
        raise customexception(e, sys) from e






def recommend_book(book_name,book_pivot):
    book_id = np.where(book_pivot.index == book_name)[0][0]


    # Prediction for a specific data point
    data_point = book_pivot.iloc[book_id, :].values.reshape(1, -1)
    cluster_label = model.predict(data_point)


    # Get all points belonging to the same cluster
    labels_all = model.predict(book_pivot)

    # Get the indices of all rows in the same cluster
    same_cluster_indices = np.where(labels_all == cluster_label)[0]

    # Retrieve the original row IDs from the DataFrame based on the indices
    original_ids = book_pivot.index[same_cluster_indices]
    book_list = []
    for idx in original_ids[:6]:  # Get at least 6 IDs
        # Ensure that idx is used as an index in book_pivot, not as a positional index
      
      book_list.append(idx)
    # same_cluster_indices
    poster_url = []
    for Title in original_ids[0:6]:
        url_value = final_rating[final_rating['Book-Title'] == Title]['Image-URL-L'].values[0]
        poster_url.append(url_value)
        
    
    return book_list, poster_url
