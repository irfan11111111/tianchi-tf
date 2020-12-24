
# import pandas as pd
# import numpy as np
# data=pd.read_csv("./OCEMOTION_a.csv")
# print(data.columns)

import csv
sample_list=[]
c=open("./OCNLI_a.csv","r",encoding="utf-8")
read=csv.reader(c)
for line in read:
    sample=(str(line[0])+"	0").split("	")
    print(sample)
    sample_list.append(sample)
c.close()
c=open("./test.csv","w",encoding="utf-8",newline='')
for sample in sample_list:
    writer=csv.writer(c)
    writer.writerow(sample)
c.close()