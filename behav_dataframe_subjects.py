## NEMO analyses -- create a dataframe over subjects (and do some descriptive stats)##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# file locations / subjects
base_dir = "D:/NEMO_analyses/behav/"
subjs_all = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34",
        "NEM_35","NEM_36","NEM_37"]
excluded = ["NEM_19","NEM_21","NEM_30","NEM_32","NEM_33","NEM_37"]
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
#subjs = ["NEM_10","NEM_11"]

# create SUPER DATAFRAME
subjects = {'Subjects':subjs}
NEMO = pd.DataFrame(subjects,columns=['Subjects'])

## Part 1 -- Picture Ratings

pic_val = []
pic_ars = []

for i,sub in enumerate(subjs):
    #read into pandas data frame
    rat = pd.read_csv(base_dir+"rat_{sub}.txt".format(sub=sub),sep="\t")
    #create seperate positive and negative dataframes
    ratsorted = rat.sort_values(by='Trial')
    ratsorted.index = list(range(32))
    ratpos = ratsorted.ix[:15]
    ratneg = ratsorted.ix[16:]
    # add variables (diff values Neg-Pos) to group list
    pic_val.append((ratneg['PicVal'].mean())-(ratpos['PicVal'].mean()))
    pic_ars.append((ratneg['PicArs'].mean())-(ratpos['PicArs'].mean()))

NEMO['Pic_Val'] = pic_val
NEMO['Pic_Ars'] = pic_ars

#compute group statitstics
print("T-Test Picture Valence")
print(stats.ttest_1samp(NEMO['Pic_Val'],0.0))
print("T-Test Picture Arousal")
print(stats.ttest_1samp(NEMO['Pic_Ars'],0.0))
print("Pearson Correlation Picture Valence & Arousal")
print(stats.pearsonr(NEMO['Pic_Val'],NEMO['Pic_Ars']))

## Part 2 -- Experiment Tone Ratings

ton_laut = []
ton_ang = []
emo_val = []
emo_ars = []

for i,sub in enumerate(subjs):
    #read into pandas data frame
    res = pd.read_csv(base_dir+"res_{sub}.txt".format(sub=sub),sep="\t")
    #ignore baseline ratings on tones and look only at experimental ratings
    res = res[4:]
    #build pos/neg data subsets
    respos = res.query('Cat == "P"')
    resneg = res.query('Cat == "N"')
    # add variables (diff values Neg-Pos) to group list
    ton_laut.append((resneg['Laut'].mean())-(respos['Laut'].mean()))
    ton_ang.append((resneg['Angenehm'].mean())-(respos['Angenehm'].mean()))
    emo_val.append((resneg['Valence'].mean())-(respos['Valence'].mean()))
    emo_ars.append((resneg['Arousal'].mean())-(respos['Arousal'].mean()))

NEMO['Ton_Laut'] = ton_laut
NEMO['Ton_Ang'] = ton_ang
NEMO['Emo_Val'] = emo_val
NEMO['Emo_Ars'] = emo_ars

## Part 3 -- Psychological & Emotion Regulation Questionnaire Data

sek_27 = {"ER_ges" : [60,56,65,77,87,77,93,55,65,86,67,81,83,75,70,71,62,78,60,69], "Aufmerksamkeit" : [9,4,7,6,8,6,9,7,11,7,7,10,8,9,6,10,8,8,6,4],
          "Klarheit" : [6,4,6,9,12,10,12,6,9,11,8,8,9,9,10,11,10,9,4,8], "Körperwahrnehmung" : [6,6,8,8,10,9,10,7,6,9,8,6,11,10,7,12,9,12,5,5],
          "Verstehen" : [7,3,9,10,11,10,12,5,9,12,7,9,10,8,10,8,8,9,4,11], "Akzeptanz" : [7,8,8,6,10,8,11,2,7,9,8,11,9,9,9,8,7,7,8,11],
          "Resilienz" : [8,10,7,8,10,9,8,6,5,11,7,10,8,6,7,5,3,6,9,11], "Selbstunterstützung" : [4,6,6,12,9,8,10,8,6,9,6,9,9,8,6,5,5,9,6,5],
          "Konfrontationsbereitschaft" : [10,10,7,9,10,11,12,6,8,11,9,10,11,10,9,6,4,11,10,6], "Regulation" : [3,5,7,9,7,6,9,8,4,7,7,8,8,6,6,6,8,7,8,8]}

scl_90 = {"Aggressivität" : [5,1,1,0,2,0,1,4,3,0,4,0,3,1,2,3,0,2,6,1], "Ängstlichkeit" : [6,2,3,6,5,4,1,3,4,3,3,1,5,3,0,1,7,3,7,6], "Depressivität" : [15,13,4,2,5,4,1,12,19,7,8,8,5,2,1,17,10,5,2,10],
          "Paranoides_Denken" : [5,3,6,0,2,1,0,1,4,1,3,1,3,1,0,5,0,0,5,2], "Phobische_Angst" : [4,4,0,1,0,2,0,1,0,2,2,0,6,1,0,0,0,1,2,0], "Psychotizismus" : [7,8,0,0,1,1,0,1,4,1,0,2,5,0,0,6,0,3,3,3],
          "Somatisierung" : [6,8,0,5,2,2,0,3,1,11,3,4,13,6,5,1,6,6,8,5], "Soziale_Unsicherheit" : [9,11,3,0,2,4,0,4,5,0,5,1,7,3,1,9,0,2,4,6], "Zwanghaftigkeit" : [13,8,2,5,6,5,0,14,12,4,9,8,9,0,7,7,3,6,9,12],
          "Psycho_ges" : [70,58,19,19,25,23,3,43,52,29,37,25,56,17,16,49,26,28,46,45], "Angst_ges" : [15,9,9,7,7,7,1,5,8,6,8,2,14,5,0,6,7,4,14,8]}

NEMO['ER_ges'] = sek_27['ER_ges']
NEMO['Angst_ges'] = scl_90['Angst_ges']
NEMO['Psycho_ges'] = scl_90['Psycho_ges']
