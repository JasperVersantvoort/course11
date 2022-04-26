# Thijs Ermens
# 19-4-2022
# Dit file maakt een nieuw file van de vcf.bgz en zorgt ervoor dat de 3
# juiste labels zijn toegevoegd

def gnome_reader(file):
    file = open(file)
    f = open("gnomad_data.csv", 'w')
    f.write("Gene,Mutation,Label\n")
    for line in file:
        if not line.startswith("\"#"):
            newline = line.split("|")
            if len(newline) > 148 and bool(newline[128]) and bool(newline[
                                                                      127]) \
                    and bool(newline[149]) and bool(newline[117]):
                name = newline[117]
                ref = newline[128][0]
                alt = newline[128][-1]
                location = newline[127]
                mutation = ref + location + alt

                label = newline[149]
                if label.startswith("probably"):
                    label = "pathogenic"
                    f.write(name + "," + mutation + "," + label + "\n")
                elif label.startswith("benign"):
                    label = "benign"
                    f.write(name + "," + mutation + "," + label + "\n")

                print("name = ", name)
                print("mutation = ", mutation)
                print("label = ", label)


def clinvar_reader(file):
    print("Dit is clinvar", file)
    file = open(file)
    for i in file:
        print(i)


if __name__ == '__main__':
    gnomad = "gnomad.exomes.r2.1.1.sites.Y.vcf"
    gnome_reader(gnomad)

