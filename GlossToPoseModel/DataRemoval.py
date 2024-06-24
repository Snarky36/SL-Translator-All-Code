import pandas as pd

def remove_rows_with_languages(csv_file_path, languages):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Filter out rows with specified languages
    filtered_df = df[~df['spoken_language'].isin(languages)]

    # Write the filtered DataFrame back to the CSV file
    filtered_df.to_csv(csv_file_path, index=False)

# Example usage:
csv_file_path = './assets/signsuisse/index.csv'
languages_to_remove = ['it', 'fr']
remove_rows_with_languages(csv_file_path, languages_to_remove)