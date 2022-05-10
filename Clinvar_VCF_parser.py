import csv
import sys
from operator import contains

'''
Takes clinvar VCF file, splits collumns on tab
Splits the last "info" collumn on ';' and searches that collumn for Benign and Pathogenic variants.
Also searches the "info" collumn for gene name.
Writes the Benign and Pathogenic variants, as well as their corresponding gene names to a CSV file

input: Clinvar VCF file
output: CSV file containing: gene name, variant, pathogenic/benign
'''
def parse(clinvar_bestand):
    # open file
    clinvar_output = open(clinvar_bestand, 'r')
    for x in clinvar_output:
        if x.startswith('#CHROM'):
            header = x
            break
    header = header.split("\t")
    column_headers = ['Gene Name','Significance','Variation']
    outputs = []
    teller = 0
    
    for line in clinvar_output:
        teller += 1
        # Splits collumns on tabs 
        line_split = line.split("\t")
        
        # Splits info collumn on ';'
        line_info_split = line_split[7].split(';')
        print("Dit is lijntje nummero: "+ str(teller))
        for i in range (len(line_info_split)):
            # Checks info collumn for benign or pathogenic only
            if line_info_split[i] == 'CLNSIG=Benign' or line_info_split[i] == 'CLNSIG=Pathogenic' :
                signif = line_info_split[i].strip("CLNSIG=")
                
                # Checks info collumn for geneinfo, aka the gene name
                for j in range (len(line_info_split)):
                    if "GENEINFO=" in line_info_split[j]:
                        genID = line_info_split[j]
                        genIDsplit = genID.split(':')
                        genID = genIDsplit[0].strip("GENEINFO=")
                        outputline = [genID, signif, line_split[3]+' -> '+line_split[4]]
                        outputs.append(outputline)
                        print(outputline)

    # Writes the nested list of outputlines to a CSV
    with open("ClinvarVCFparse.csv", "w", newline='' ) as my_csv:    
        w = csv.writer(my_csv)
        w.writerow(column_headers)
        w.writerows(outputs)


if __name__ == '__main__':
    clinvar_bestand = "./clinvar.vcf"
    parse(clinvar_bestand)



#Discarded medgen parse, because medgen id was deemed irrelevant, mayhaps usefull someday
    # genstrip = line_info_split[i-4].lstrip("CLNDISDB=")
    # gensplit = genstrip.split(",")
    # print(gensplit)
    # for j in range(len(gensplit)):
    #     if "MedGen" in gensplit[j]:
    #         medgenID = gensplit[j]
    #         print(medgenID)
