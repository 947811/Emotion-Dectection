Data Preparation
Load and preprocess the data:

bash
Copy code
python src/data_preparation.py
This script reads the dataset, handles missing values, and saves the cleaned data.

Data Visualization
Visualize the dataset distribution:

bash
Copy code
python src/data_visualization.py
This script generates visualizations to understand the distribution of labels in the dataset.

Model Training
Train various models:

bash
Copy code
python src/model_training.py
This script trains different machine learning models (e.g., SVM, Logistic Regression, Naive Bayes, XGBoost, Random Forest, KNN) on the dataset and saves the trained models and vectorizers.

Model Inference
Make predictions with the trained model:

bash
Copy code
python src/model_inference.py
This script loads the trained Logistic Regression model and TF-IDF vectorizer to predict the class of new input text.

File Descriptions
data_preparation.py: Loads and preprocesses the dataset.
data_visualization.py: Generates bar plots, histograms, and pie charts to visualize data distributions.
model_training.py: Trains various models, including SVM, Logistic Regression, Naive Bayes, XGBoost, Random Forest, and KNN, and evaluates their performance.
model_inference.py: Uses the trained Logistic Regression model to predict the class of new input text.
utils.py: Contains utility functions (if any).
Models
The trained models and TF-IDF vectorizer are saved in the models/ directory. These include:

logistic_regression_model.pkl: The serialized Logistic Regression model.
tfidf_vectorizer.pkl: The serialized TF-IDF vectorizer.
Other models are also saved as specified in the model_training.py script.
