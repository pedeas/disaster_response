# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories data and returns a mergred dataframe

    Parameters:
    messages_filepath (str): path for messages file
    categories_filepath (str): path for categories file

    Returns:
    DataFrame: Merged dataframe

   """
    # load messages dataset
    messages = pd.read_csv(messages_filepath).set_index('id')

    # load categories dataset
    categories = pd.read_csv(categories_filepath).set_index('id')

    # merge datasets
    df = messages.join(categories)

    return df


def clean_data(df):
    """Create dummy columns for each category and remove duplicates

    Parameters:
    df (DataFrame): dataframe with messages and categories

    Returns:
    DataFrame: structured dataframe with a dummy column for each category

   """
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.str.split('-', 1, True).iloc[:, 0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category to number
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Saves a dataframe in a sqlite database

    Parameters:
    df (DataFrame): dataframe to save
    database_filename (str): path to save database

   """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(type(df))
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()