import pandas as pd

bestand = "train_data_bio_prodict.parq"

def parq_lezer(bestand):
    #problemen geeft
    text = pd.read_parquet(bestand, engine='pyarrow')
    print(text)

def main():
    parq_lezer(bestand)
main()
