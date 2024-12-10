import os
import sys
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from src.Machine_Recommendation.logger import logging
from src.Machine_Recommendation.exception import customexception
from dataclasses import dataclass
from src.Machine_Recommendation.utils.utils import save_object
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import   silhouette_score
from scipy.sparse import csr_matrix
# Base Clustering Model Class
import pickle

class BaseClustering:
    def __init__(self, name):
        self.name = name
        self.best_model = None
        self.best_params = None
        self.best_score = -1

    def fit(self, X_train, X_test):
        raise NotImplementedError("The fit method must be implemented by subclasses")

    def evaluate(self, X_test):
        if self.best_model is None:
            raise ValueError(f"{self.name}: Model not trained yet.")
        labels_test = self.best_model.predict(X_test)
        if len(set(labels_test)) > 1:
            score = silhouette_score(X_test, labels_test)
        else:
            score = -1
        return score

# KMeans Clustering
class KMeansClustering(BaseClustering):
    def __init__(self):
        super().__init__("K-Means")


    def fit(self, X_train, X_test):
        # Convert sparse matrices to dense arrays if necessary
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()


        # Parameter search space
        param_dist = {
            'n_clusters': np.arange(2, 10),
            'init': ['k-means++', 'random'],
            'n_init': np.arange(5, 15),
            'max_iter': np.arange(100, 500)
        }
        
        # Perform Randomized Search
        search = RandomizedSearchCV(
            KMeans(random_state=42),
            param_dist,
            n_iter=20,
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train)

        # Save best model, parameters, and silhouette score
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = silhouette_score(X_test, self.best_model.predict(X_test))

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    raw_data_path=(Path(os.path.join("artifacts","book_pivoted.csv")))
    book_name_path = os.path.join('artifacts','book_name.pkl')
    final_rating_path = os.path.join('artifacts','final_rating.pkl')
    book_pivot_path = os.path.join('artifacts','book_pivot.pkl')
   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self):
        try:
            # Convert to string before creating CSR matrix
            data =  pd.read_csv(self.model_trainer_config.raw_data_path)
            data.fillna(0, inplace=True)
            # Set 'Title' as the index if it isn't already
            if data.index.name != 'Title':
                data.set_index('Title', inplace=True)
            book_name = data.index
            book_pivoted = data 
            book_sparse = csr_matrix(data)
            X_train, X_test = train_test_split(book_sparse, test_size=0.2, random_state=42)
            logging.info('Splitting Dependent and Independent variables from train and test data')
            kmeans_model = KMeansClustering()
            print("\n--- K-Means Clustering ---")
            kmeans_model.fit(X_train, X_test)  # Use train_array and test_array

            # Evaluate on test data
            test_score = kmeans_model.evaluate( X_test)  # Use test_array
            print(f"K-Means Best Parameters: {kmeans_model.best_params}")
            print(f"K-Means Silhouette Score on Test Set: {test_score}")
            final_rating=pd.read_csv(Path(os.path.join("artifacts","final_rating.csv")))
            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=kmeans_model.best_model
              
                
            )
            save_object(self.model_trainer_config.book_name_path, book_name)
            save_object(self.model_trainer_config.final_rating_path, final_rating)
            save_object(self.model_trainer_config.book_pivot_path, book_pivoted)

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e,sys) from e