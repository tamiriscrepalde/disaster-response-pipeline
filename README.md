# Disaster Response Pipeline Project

This project was developed as part of the Udacity Data Scientist Nanodegree. 

### Context
The project consists in a Flask app that uses an NLP machine learning pipeline to classify emergency messages. The machine learning model is trained using pre-labeled data provided by Figure Eight.

### Instructions:
Follow the guidelines below to run the app:

1. It's recommended to start a virtual environment.

2. Install the required libraries by running the command: `pip install -r requirements.txt`.
   
3. Run the ETL pipeline that cleans the data and stores on a database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   
4. Run the ML pipeline that trains the classifier and saves the model: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`.
   
5. Go to `app`directory: `cd app`.

6. Run the Flask app: `python run.py`.

7. Click the `PREVIEW` button to open the homepage.
   