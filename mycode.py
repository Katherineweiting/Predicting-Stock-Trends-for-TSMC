import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

UP = pd.read_excel("TSMC/5.xlsx", sheet_name = "UP")
DOWN = pd.read_excel("TSMC/5.xlsx", sheet_name = "DOWN")
excel_data_df2 = pd.read_excel("TSMC3.xlsx", sheet_name = "all")
#print(wordlist)
U=UP['詞'][:150].tolist()
D=DOWN['詞'][:150].tolist()
wordlist=U+D
updowns = excel_data_df2["updowns"].tolist()
#print(updowns)
convertlist2 = excel_data_df2["content"].tolist()
sentencelist = []
for i in range(len(convertlist2)):
	Str = convertlist2[i]
	Str1 = re.sub('\W+', '', Str).replace("_", '')
	Chinese = re.compile("[A-Za-z0-9\!\%\[\]\,\。]")
	Str2 = re.sub(Chinese,"",Str1)
	Str3 = Str2.replace(" ","")
	sentencelist.append(Str3)
#print(sentencelist)

vector = {}
vector_values = []
for i in range(len(sentencelist)):
	wordfreq = []
	for j in wordlist:
		wordfreq.append(sentencelist[i].count(j))
	vector[i+1] = wordfreq
	vector_values.append(wordfreq)

df = pd.DataFrame(vector_values)
col = df.columns.to_numpy()
for i in col:
	df[i] -= df[i].mean()
	df[i] /= df[i].std()
X = df.to_numpy()

#print(vector_values)

#df = pd.DataFrame(vector)
#df.to_excel('dict.xlsx', sheet_name = 'vectors')

"""
sentence_list = vector.values()
word_dict = {"content": sentence_list, "mark":updowns}
word_dict_df = pd.DataFrame(word_dict)
#print(word_dict_df)

from sklearn.model_selection import train_test_split

X = word_dict_df["content"]
y = word_dict_df["mark"]

"""
for i in range(len(vector_values)):
	vector_values[i] = np.array(vector_values[i])
#X = np.array(vector_values)
y = np.array(updowns)
#print(X)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 101)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import ensemble, preprocessing, metrics
from sklearn.naive_bayes import GaussianNB

print("======SVC======")
model = SVC(kernel = "poly")
model.fit(train_X,train_y)
predictions = model.predict(test_X)
print(confusion_matrix(test_y, predictions))
print('\n')
print(classification_report(test_y, predictions))


print("======RANDOM FOREST======")
model = ensemble.RandomForestClassifier(n_estimators = 100)
model.fit(train_X,train_y)
predictions = model.predict(test_X)
print(confusion_matrix(test_y, predictions))
print('\n')
print(classification_report(test_y, predictions))

print("======Naise Bayse======")
model = GaussianNB()
model.fit(train_X,train_y)
predictions = model.predict(test_X)
print(confusion_matrix(test_y, predictions))
print('\n')
print(classification_report(test_y, predictions))























