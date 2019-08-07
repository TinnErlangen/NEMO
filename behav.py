## NEMO behav analyses ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# file locations / subjects
base_dir = "C:/Users/kimca/Documents/MEG_analyses/NEMO/behav/"
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34"]
#subjs = ["NEM_35","NEM_36","NEM_37"]

# create SUPER DATAFRAME
subjects = {'Subjects':subjs}
NEMO = pd.DataFrame(subjects,columns=['Subjects'])

## Part 1 -- Picture Ratings

#delete first lines in ratlog & save as new file
# for sub in subjs:
#     oldfile = open("ratlog_{sub}.txt".format(sub=sub))
#     x = oldfile.readlines()
#     del x[0]
#     newfile = "rat_{sub}.txt".format(sub=sub)
#     with open(newfile,"w") as file:
#         for line in x:
#             file.write(line)

for sub in subjs:
    #read into pandas data frame
    rat = pd.read_csv("rat_{sub}.txt".format(sub=sub),sep="\t")
    #add rank variable for ratings
    rat['ValRank'] = rat['PicVal'].rank(ascending=False)
    rat['ArsRank'] = rat['PicArs'].rank()
    #create seperate positive and negative dataframes
    ratsorted = rat.sort_values(by='Trial')
    ratsorted.index = list(range(32))
    ratpos = ratsorted.ix[:15]
    ratneg = ratsorted.ix[16:]
    #plot scatterplot with positive and negative dots
    plt.plot(ratpos['PicVal'],ratpos['PicArs'],'go',ratneg['PicVal'],ratneg['PicArs'],'ro')
    plt.axis([-1,1,0,1])
    plt.xlabel('Picture Valence')
    plt.ylabel('Picture Arousal')
    plt.title('Picture Rating {sub}'.format(sub=sub))
    plt.savefig('picrat_{sub}.png'.format(sub=sub))
    plt.close()






# ## Part 2 -- Experiment Tone Ratings
# for sub in subjs:
#     #delete first lines in ratlog & save as new file
#     oldfile = open("reslog_{sub}.txt".format(sub=sub))
#     x = oldfile.readlines()
#     del x[0]
#     newfile = "res_{sub}.txt".format(sub=sub)
#     with open(newfile,"w") as file:
#         for line in x:
#             file.write(line)
#     #read into pandas data frame
#     res = pd.read_csv("res_{sub}.txt".format(sub=sub),sep="\t")
