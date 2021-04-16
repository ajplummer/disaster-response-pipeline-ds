# Disaster Response Pipeline Project

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Installation
The main content of this repository are Python files as well as a web application utilizing Flask and Python as the backend.

To run this project you will need:
- Python 3.0
- Numpy
- Pandas
- SQLAlchemy
- NLTK
- Pickle
- Plotly
- Flask
- Sklearn
- A Web Browser

## Project Motivations
This is a required project for the Udacity Data Scientist for Enterprise nanodegree but is also one I am interested in personally.

This seems to be a great real-world example of where data science can provide benefit to humanity. In a disaster, there will be many messages received or trending on social media and it is important that responders can sift through the all of the data to find the more urgent or important ones to focus on. Data science can automate that process to save time and allow the responders to focus on what they do best: helping people.

## File Descriptions
- app
   - template
     - master.html - main page of web app
     - go.html - classification result page of web app
   - run.py - Flask file that runs app

- data
   - disaster_categories.csv - data to process 
   - disaster_messages.csv - data to process
   - process_data.py - Code for processing the two data files above, cleaning up the data and storing it in a database.

- models
   - train_classifier.py - Code for training the model based upon the data processed in the prior step.

- README.md - This file.

## Licensing, Authors, Acknowledgements, Etc.
The data was and file structure was provided by Udacity as part of their Data Science for Professionals Nanodegree.

The code was written by me, A.J. Plummer, with some functions copied (either word-for-word or in spirit) from Udacity coursework. Those functions are noted in the comments in the code.

Thank you to Udacity for putting together a great "real world" exercise for us and to my company, Mars, Inc., for enabling me to take on this course!
