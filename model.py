import pandas as pd
import numpy as np
import numbers
import decimal
import scipy.stats as ss
import matplotlib.pyplot as plt
from statistics import stdev
from statistics import mean
import time
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import mutual_info_classif
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn import svm



# X = pd.read_csv('datasetSentences.txt', sep="\t", header=None, index_col=0)
# Y = pd.read_csv('datasetSplit.txt', sep=",", header=0, index_col=0)

# sttReviewsX = pd.read_csv('original_rt_snippets.txt', sep='\n',header=None)
# sttReviewsY = pd.read_csv('sentiment_labels.txt', sep='|',header=0).iloc[:,-1]
# print(sttReviewsY)
# print(sttReviewsX)




tTrain1 = pd.read_csv('TRECtrain1.txt', sep="\\", header=None)
tTrain2 = pd.read_csv('TRECtrain2.txt', sep="\\", header=None)
# print(tTrain2)

trecTrainX1 = tTrain1.iloc[:,-1]
# print(trecTrainX)
trecTrainY1 = tTrain1.iloc[:,0]
# print(trecTrainY)

trecTrainX2 = tTrain2.iloc[:,-1]
# print(trecTrainX)
trecTrainY2 = tTrain2.iloc[:,0]
# print(trecTrainY)




neg_data = np.loadtxt("rt-polarity.neg", dtype='str', delimiter='\n', encoding='latin1')
pos_data = np.loadtxt("rt-polarity.pos", dtype='str', delimiter='\n', encoding='latin1')


reshaped_neg = np.reshape(neg_data, (neg_data.shape[0], 1))
reshaped_pos = np.reshape(pos_data, (pos_data.shape[0], 1))

appended_neg = np.c_[reshaped_neg, np.zeros(reshaped_neg.shape[0], dtype=int)]
appended_pos = np.c_[reshaped_pos, np.ones(reshaped_pos.shape[0], dtype=int)]

data = np.concatenate((appended_neg, appended_pos), axis=0)
np.random.shuffle(data)


# PositiveMRx = pd.read_csv(x1, sep='\n', header=None)
# NegativeMRx = pd.read_csv(x2, sep='\n', header=None)
# print(PositiveMRx)
# print(NegativeMRx)

# filenames = ['rtPositive.txt', 'rtNegative.txt']
# with open('reviewSnippets.txt', 'w') as outfile:
#     for fname in filenames:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)

# f1 = open("reviewLabels.txt", "w")

# for i in range(10660):
#     if(i < 5330):
#         f1.writelines('fresh\n')
#     else:
#         f1.writelines('rotten\n')
# f1.close()

reviewX = open('reviewSnippets.txt', "r",encoding='utf8')
reviewY = open('reviewLabels.txt', "r",encoding='utf8')
# x.encode('utf8')

MRx = pd.read_csv(reviewX, sep='\n', header=None, dtype=str) #.values
# MRx = [str (item) for item in MRx]
MRy = pd.read_csv(reviewY, sep='\n', header=None, dtype=str)
# print(MRx)

rX = MRx.values
rY = MRy.iloc[::]
# print(rY)

#####Text Preprocessing techniques, using tfidf and countvectorizer
# tfidf = TfidfVectorizer(stop_words="english", lowercase=False, smooth_idf=False, sublinear_tf=True, use_idf=True, max_features=6550,) # ngram_range=(1,3)) #,max_df=0.1,) # max_features=65500)  
# cv = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1,2))
tfidf1 = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
cv1 = CountVectorizer(lowercase=False, ngram_range=(1,2))

tfidf2 = TfidfVectorizer(lowercase=False, ngram_range=(1,1))
cv2 = CountVectorizer(lowercase=False, ngram_range=(1,1))

tfidf3 = TfidfVectorizer(lowercase=False, ngram_range=(1,3))
cv3 = CountVectorizer(lowercase=False, ngram_range=(1,3))

tfidf4 = TfidfVectorizer(lowercase=False, ngram_range=(2,3))
cv4 = CountVectorizer(lowercase=False, ngram_range=(2,3))

# ##### SKlearn Models created and used for training 
lr = LogisticRegression(solver="sag", penalty="l2", class_weight="balanced")
multiNB = MultinomialNB(alpha=0.3)
dtc = tree.DecisionTreeClassifier()
kf = StratifiedKFold(n_splits=5)


svm3 = svm.SVC(kernel='linear', decision_function_shape='ovo', probability=True, verbose=True)

lr2 = LogisticRegression(solver='sag', penalty='l2', class_weight='balanced')


# ###Helper method created to specify model being used, and to fit using said model and return a test score
def getScoretWithModel(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


###################Train Test split 80/20 Using multinomial NB, fits model and runs prediction returning score##################
#############(or any model can be inserted as the first parameter in the getscorewithmodel method)###################

# x_train, x_test, y_train, y_test = train_test_split(trecTrainX1, trecTrainY1, test_size=0.2, random_state=4)
# DataTrainTF = cv.fit_transform(x_train)
# DataTestTF = cv.transform(x_test)
# print("Accuracy:", getScoretWithModel(lr, DataTrainTF, DataTestTF, y_train, y_test))

# x_train, x_test, y_train, y_test = train_test_split(trecTrainX2, trecTrainY2, test_size=0.2, random_state=4)
# DataTrainTF = cv.fit_transform(x_train)
# DataTestTF = cv.transform(x_test)
# print("Accuracy:", getScoretWithModel(svm, DataTrainTF, DataTestTF, y_train, y_test))





# x_train, x_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.3, random_state=4)
# DataTrainTF = tfidf1.fit_transform(x_train)
# DataTestTF = tfidf1.transform(x_test)
# print("Accuracy:", getScoretWithModel(svm, DataTrainTF, DataTestTF, y_train, y_test))

# x_train, x_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.3, random_state=4)
# DataTrainTF = tfidf1.fit_transform(x_train)
# DataTestTF = tfidf1.transform(x_test)
# print("Accuracy:", getScoretWithModel(svm1, DataTrainTF, DataTestTF, y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.3, random_state=4)
DataTrainTF = tfidf1.fit_transform(x_train)
DataTestTF = tfidf1.transform(x_test)
print("Accuracy:", getScoretWithModel(lr2, DataTrainTF, DataTestTF, y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(trecTrainX2, trecTrainY2, test_size=0.2, random_state=4)
DataTrainTF = tfidf1.fit_transform(x_train)
DataTestTF = tfidf1.transform(x_test)
print("Accuracy:", getScoretWithModel(lr2, DataTrainTF, DataTestTF, y_train, y_test))


