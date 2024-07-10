import pandas as pd

def remove_rows_with_languages(csv_file_path, languages):
    df = pd.read_csv(csv_file_path)

    filtered_df = df[~df['spoken_language'].isin(languages)]

    filtered_df.to_csv(csv_file_path, index=False)

csv_file_path = './assets/signsuisse/index.csv'
languages_to_remove = ['it', 'fr']
remove_rows_with_languages(csv_file_path, languages_to_remove)