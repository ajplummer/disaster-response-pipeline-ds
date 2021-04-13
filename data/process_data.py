# import libraries
import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function load_data takes in two filepaths and reads the data from them into dataframes

    Parameters:
    messages_filepath: Filepath of the Messages file
    categories_filepath: Filepath of the Categories file

    Returns:
    df: The merged dataframe of messages and categories
    categories: The original categories dataframe

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df, categories



def upd_cat_vals(cat):
    """
    Function upd_cat_vals takes in a category value string and returns just the integer

    Parameters:
    cat: A category value

    Returns:
    val: The integer value found in the category string

    """
    # set each value to be the last character of the string
    val = cat[cat.index('-')+1:]

    # convert column from string to numeric
    return int(val)



def clean_data(df, categories):
    """
    Function clean_data takes in two dataframes and cleans up the data in the primary one

    Parameters:
    df: The primary, merged dataframe to be used in this program
    categories: The original categories dataframe

    Returns:
    df: The merged dataframe of messages and categories after it has been cleaned

    """
    # create a dataframe of the 36 individual category columns
    old_categories = categories
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list()
    for col in row:
        category_colnames.append(col[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(upd_cat_vals)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    categories['id'] = old_categories['id']
    df = df.merge(categories, on='id')

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Function save_data takes in the primary dataframe and the desired filename for the output database and
    then saves the dataframe to the database.

    Parameters:
    df: The primary dataframe
    database_filename: Desired filename of the output database

    Returns:
    N/A

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
