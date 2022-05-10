import pandas as pd, csv

bestand = "Files/train_data_bio_prodict.parq"

def parq_lezer(bestand):
    col_names = ["mutation", "conservationPro", "conservationAla",
                 "conservationHis",
                 "conservationThr", "conservationGln", "conservationTyr",
                 "conservationGly",
                 "conservationArg", "conservationVal", "consWildType",
                 "conservationGlu",
                 "conservationMet", "conservationLys", "conservationIle",
                 "conservationPhe",
                 "conservationLeu", "conservationAsn", "conservationSer",
                 "conservationAsp",
                 "conservationCys", "consVariant", "conservationTrp", "source",
                 "label"]
    text = pd.read_parquet(bestand, engine='pyarrow')
    # print(text[0])

    df = pd.DataFrame(text)
    list = []
    mutation_names = df.index
    print(df.index)
    for name in mutation_names:
        alt = (name[-3:])
        for i in range(len(name)):
            try:
                int(name[(-4-i)])
            except ValueError:
                ref = (name[(-6-i):(-3-i)])
                break
        print(df[df["conservationGly"]])
    # print(df.columns)


    # print(df.columns)
    # df.to_csv('test.csv')

def main():
    parq_lezer(bestand)
main()
