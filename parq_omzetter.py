import pandas as pd, csv

bestand = "train_data_bio_prodict.parq"

def parq_lezer(bestand):
    text = pd.read_parquet(bestand, engine='pyarrow')
    #  print(text)

    df = pd.DataFrame(text)
    #print(df)
    df.to_csv('test.csv')

def main():
    parq_lezer(bestand)
main()
