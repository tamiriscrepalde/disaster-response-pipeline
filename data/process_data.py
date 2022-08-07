import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Loads data from messages and categories files.

    Args:
        messages_filepath (_type_): Filepath of the messages dataset.
        categories_filepath (_type_): Filepath of the categories dataset.

    Returns:
        df: Pandas DataFrame containing the merged data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = merge_data(messages, categories)
    return df


def merge_data(messages, categories) -> pd.DataFrame:
    """Merges messages and categories datasets.

    Args:
        messages (_type_): DataFrame containing the messages dataset.
        categories (_type_): DataFrame containing the categories dataset.

    Returns:
        df: Pandas DataFrame containing the merged data.
    """
    df = pd.concat([messages, categories.iloc[:, 1:]], axis=1)
    categories = extract_categories(df)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    return df


def extract_categories(df) -> pd.DataFrame:
    """Extracts categories from the merged dataset.

    Args:
        df: Merged dataset.

    Returns:
        categories: Dataset containing the categories as columns."""
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    categories = make_categories_dummies(categories)
    return categories


def make_categories_dummies(categories) -> pd.DataFrame:
    """Creates dummy variables from the categories dataset.

    Args:
        categories (_type_): Dataset containing the categories.

    Returns:
        categories: Dataset containing the categories as columns."""
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    return categories


def clean_data(df) -> pd.DataFrame:
    """Cleans the dataset.

    Args:
        df: Pandas dataset to clean.

    Returns:
        df: Cleaned pandas dataset.
    """
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename) -> None:
    """Saves the data at a SQLite database.

    Args:
        df: Pandas dataset to save.
        database_filename: Filepath of the database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)


def main() -> None:
    """Main function."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {database_filepath}')
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
