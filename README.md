# Disaster Response Pipeline Project

In this project messages sent during disaster events are categorised in order to send the messages to an appropriate disaster relief agency. 
It includes an ETL pipeline for loading and cleaning of messages for modelling and a machine learning pipeline to train a random forest classifier for classification to all unique categories. 
New messages can be categorised using the model and the result is visualised in a web app.

The file data/disaster_messages.csv contains messages that were used in this project and data/disaster_categories.csv contains the corresponding categories for all messages. 
You can easily replace them with new data structured in the same way.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
