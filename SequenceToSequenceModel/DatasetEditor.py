import pandas as pd

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file, sep='|')
    df = df[['translation', 'orth']]
    df = df.rename(columns={'translation': 'text', 'orth': 'target'})

    df.to_csv(output_file, index=False)


input_file_path = '../Phoenix/PHOENIX-2014-T.train-complex-annotation.corpus.csv'
output_file_path = '../Phoenix/PHOENIX-2014-T.train-complex-annotation.corpus2.csv'
process_csv(input_file_path, output_file_path)