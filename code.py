import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#dividing the input dataset into majority - minority dataset
datainput = pandas.read_csv('creditcard.csv')
minority = datainput[datainput['Class'] == 1]
majority = datainput[datainput['Class'] == 0]

#initialising matplotlib in plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
str1='Majority Dataset('+str(len(majority))+')'
str2='Minority Dataset('+str(len(minority))+')'

#plotting majority and minority datasets on bar graph
fieldName = [str1, str2]
fieldSize = [len(majority),len(minority)]
ax.bar(fieldName,fieldSize,color=['deepskyblue','lime'])
plt.show()

#plotting majority and minority datasets on x-y axis using scatter of malplotlib
plt.plot(majority["V1"], majority["V2"], "o", color="deepskyblue")
plt.plot(minority["V1"], minority["V2"], "o", color="lime")
plt.show()

#creating kmeans template
kmeans = KMeans(2)
kmeans.fit(minority)

#copying minority dataset 
clustors = minority.copy()
clustors['cluster_pred'] = kmeans.fit_predict(minority)

#plotting minority datasets as k means clusters
plt.scatter(clustors['V1'], clustors['V2'], c=clustors['cluster_pred'], cmap='autumn')
plt.show()
