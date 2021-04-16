import json
import plotly
import pandas as pd
import joblib
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
with open("../models/classifier.pkl", 'rb') as file:
    model = pickle.load(file)
#model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Find the top 10 categories with the highest % of messages and display [Copied from https://github.com/harshdarji23/Disaster-Response-WebApplication]
    top_cats_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)#.sort_values(ascending = False)[:,0:10]
    top_cats_df=(top_cats_df.sort_values(ascending=False)[0:10])
    top_cats_names = list(top_cats_df.index)
    top_cats_proportions = top_cats_df[0]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # [Copied from https://github.com/harshdarji23/Disaster-Response-WebApplication]
        {
            'data': [
                Bar(
                    x = top_cats_names,
                    y = top_cats_proportions
                )
            ],

            'layout': {
                'title': 'Top 10 Categories by Proportion of Messages Received',
                'yaxis': {
                    'title': "Proportions"
                },
                'xaxis': {
                    'title': "Message Types"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_labels)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=df.columns[4:],
                    y=classification_labels
                )
            ],

            'layout': {
                'title': 'Genre Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids,
        graphJSON=graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
