# Thijs Ermens
# 26-4-2022
# Functie die een csv omzet naar een bestand met one-hot-encoding

import pandas as pd


def one_hot_encoding(file):
    print(file)
    file = open(file)
    for line in file:
        print(line)
        split = line.split(',')
        # print(split)


if __name__ == '__main__':
    file = "VoorbeeldDataset.csv"
    one_hot_encoding(file)
