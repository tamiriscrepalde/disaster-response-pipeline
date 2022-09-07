# Disaster Response Pipeline Project

This project was developed as part of the Udacity Data Scientist Nanodegree. 

### Context
The project consists in a Flask app that uses a NLP machine learning pipeline to classify emergency messages. The machine learning model is trained using pre-labeled data provided by Figure Eight.

### Instructions:
Follow the guidelines below to run the app:

1. It's recommended to start an virtual environment.

2. Install the required libraries by running the command: `pip install requirements.txt`.
   
3. Run the ETL pipeline that cleans the data and stores in database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
4. Run the ML pipeline

5. Run your web app: `python run.py`

6. Click the `PREVIEW` button to open the homepage