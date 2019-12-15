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


tTrain1 = pd.read_csv('TRECtrain1.txt', sep="\\", header=None)
tTrain2 = pd.read_csv('TRECtrain2.txt', sep="\\", header=None)

trecTrainX1 = tTrain1.iloc[:,-1]
trecTrainY1 = tTrain1.iloc[:,0]
trecTrainX2 = tTrain2.iloc[:,-1]
trecTrainY2 = tTrain2.iloc[:,0]




neg_data = np.loadtxt("rt-polarity.neg", dtype='str', delimiter='\n', encoding='latin1')
pos_data = np.loadtxt("rt-polarity.pos", dtype='str', delimiter='\n', encoding='latin1')


reshaped_neg = np.reshape(neg_data, (neg_data.shape[0], 1))
reshaped_pos = np.reshape(pos_data, (pos_data.shape[0], 1))

appended_neg = np.c_[reshaped_neg, np.zeros(reshaped_neg.shape[0], dtype=int)]
appended_pos = np.c_[reshaped_pos, np.ones(reshaped_pos.shape[0], dtype=int)]

data = np.concatenate((appended_neg, appended_pos), axis=0)
np.random.shuffle(data)



reviewX = open('reviewSnippets.txt', "r",encoding='utf8')
reviewY = open('reviewLabels.txt', "r",encoding='utf8')

MRx = pd.read_csv(reviewX, sep='\n', header=None, dtype=str) #.values
MRy = pd.read_csv(reviewY, sep='\n', header=None, dtype=str)

rX = MRx.values
rY = MRy.iloc[::]

#####Text Preprocessing techniques, using tfidf and countvectorizer
# tfidf = TfidfVectorizer(stop_words="english", lowercase=False, smooth_idf=False, sublinear_tf=True, use_idf=True, max_features=6550,) # ngram_range=(1,3)) #,max_df=0.1,) # max_features=65500)  
# cv = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1,2))


tfidf1 = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
cv1 = CountVectorizer(lowercase=False, ngram_range=(1,2))

tfidf2 = TfidfVectorizer(lowercase=False, strip_accents="ascii", ngram_range=(1,2),analyzer="word")
cv2 = CountVectorizer(lowercase=False, strip_accents="ascii", ngram_range=(1,2), analyzer="word")

tfidf3 = TfidfVectorizer(lowercase=False, ngram_range=(1,3))
cv3 = CountVectorizer(lowercase=False, ngram_range=(1,3))

tfidf4 = TfidfVectorizer(lowercase=False, ngram_range=(2,3))
cv4 = CountVectorizer(lowercase=False, ngram_range=(2,3))

# ##### SKlearn Models created and used for training 
lr = LogisticRegression(solver="sag", penalty="l2", class_weight="balanced")
dtc = tree.DecisionTreeClassifier()
kf = StratifiedKFold(n_splits=5)
linSVM = svm.LinearSVC()


# svm3 = svm.SVC(kernel='linear', decision_function_shape='ovo',)
# svm3 = svm.SVC(kernel='linear', decision_function_shape='ovo', shrinking=False, C= 1.7,) #Best

svm3 = svm.SVC(kernel='linear', decision_function_shape='ovo', probability=True, class_weight='balanced' )

# lr2 = LogisticRegression(solver='newton-cg', penalty='l2', class_weight='balanced', )

# lr2 = LogisticRegression(solver="newton-cg", penalty='none', class_weight='balanced') #Best

# lr2 = LogisticRegression(solver="sag", penalty='none', class_weight='balanced') #Best

# lr2 = LogisticRegression(solver="liblinear")




multiNB = MultinomialNB(alpha=0.3)

# ###Helper method created to specify model being used, and to fit using said model and return a test score
def getScoretWithModel(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


###################Train Test split 80/20, fits model and runs prediction returning score##################
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
DataTrainTF = tfidf2.fit_transform(x_train)
DataTestTF = tfidf2.transform(x_test)
print("SVM Accuracy:", getScoretWithModel(svm3, DataTrainTF, DataTestTF, y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(trecTrainX2, trecTrainY2, test_size=0.2, random_state=4)
DataTrainTF = tfidf2.fit_transform(x_train)
DataTestTF = tfidf2.transform(x_test)
print("SVM Accuracy:", getScoretWithModel(svm3, DataTrainTF, DataTestTF, y_train, y_test))

# x_train, x_test, y_train, y_test = train_test_split(trecTrainX2, trecTrainY2, test_size=0.2, random_state=4)
# DataTrainTF = tfidf2.fit_transform(x_train)
# DataTestTF = tfidf2.transform(x_test)
# print("MNB Accuracy:", getScoretWithModel(multiNB, DataTrainTF, DataTestTF, y_train, y_test))
