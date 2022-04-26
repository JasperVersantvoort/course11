import blosum as bl
matrix = bl.BLOSUM(62)

bestand = open("gnomad_data.csv")
f = open("gnomad_data_new.csv", "w")
f.write(bestand.readline().rstrip() + ",Blosum\n")
for regel in bestand:
    if "," in regel:
        ref = regel.split(",")[1][0]
        mut = regel.split(",")[1][-1]
        score = matrix[ref + mut]
        f.write(regel.rstrip() + "," + str(score) + "\n")
