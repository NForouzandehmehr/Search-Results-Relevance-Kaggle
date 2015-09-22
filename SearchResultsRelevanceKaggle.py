import pandas as pd
import numpy as np 
import gc
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
import re
from sklearn.feature_extraction import text
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
import math

#method1

# Loading the training and test file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Droping ID columns and labels
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
relevance = train.median_relevance.values

train= train.drop(['median_relevance', 'relevance_variance'], axis=1)
train_query = list(train.apply(lambda x:'%s' % (x['query']),axis=1))
train_prodTitle= list(train.apply(lambda x:'%s' % (x['product_title']),axis=1))
train_prodDesc= list(train.apply(lambda x:'%s' % (x['product_description']),axis=1))
test_query = list(test.apply(lambda x:'%s' % (x['query']),axis=1))
test_prodTitle= list(test.apply(lambda x:'%s' % (x['product_title']),axis=1))
test_prodDesc= list(test.apply(lambda x:'%s' % (x['product_description']),axis=1))
del train
del test
# Create tfidf vectoriser
tfv = TfidfVectorizer(min_df=0,  max_features=None, strip_accents='unicode', 
        analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 5), use_idf=1,
        smooth_idf=1,sublinear_tf=1, stop_words = 'english')


#Fit TFIDF on train and test
tfv.fit(train_prodDesc)
train_q=  tfv.transform(train_query)
test_q=  tfv.transform(test_query)

train_t=  tfv.transform(train_prodTitle)
test_t=  tfv.transform(test_prodTitle)
train_d=  tfv.transform(train_prodDesc)
test_d=  tfv.transform(test_prodDesc)
train_tfidf=hstack([train_q,train_t,train_d])
test_tfidf=hstack([test_q,test_t,test_d])

del train_query
del train_prodDesc
del train_prodTitle
del test_query
del test_prodDesc;
del test_prodTitle

#Calculate Cosine Similarty of Query and Product Title and Cosine Similarity of Query and Product Description
norms_q_train = (train_q.data ** 2).sum()
norms_d_train = (train_d.data ** 2).sum()
norms_t_train = (train_t.data ** 2).sum()
norms_q_test = (test_q.data ** 2).sum()
norms_d_test = (test_d.data ** 2).sum()
norms_t_test = (test_t.data ** 2).sum()
train_q_norm= train_q/norms_q_train
train_d_norm= train_d/norms_d_train
train_t_norm= train_t/norms_t_train
test_q_norm= train_q/norms_q_train
test_d_norm= train_d/norms_d_train
test_t_norm= train_t/norms_t_train
similarity_qd_train=(train_q_norm.dot(train_d_norm.T)).todense()
similarity_qd_test=(test_q_norm.dot(test_d_norm.T)).todense()
mean_sim_qd_train=similarity_qd_train.sum(axis=1)
mean_sim_qd_test=similarity_qd_test.sum(axis=1)
del train_d_norm
del test_q_norm
del similarity_qd
gc.collect()
similarity_qt_train=(train_q_norm.dot(train_t_norm.T)).todense()
similarity_qt_test=(test_q_norm.dot(test_t_norm.T)).todense()
mean_sim_qt_train=similarity_qt_train.sum(axis=1)
mean_sim_qt_test=similarity_qt_test.sum(axis=1)
del train_q_norm
del train_t_norm
del similarity_qt
del test_q_norm
del test_t_norm
del similarity_qt
gc.collect()
mean_sim_train=np.concatenate((mean_sim_qd_train,mean_sim_qt_train),axis=1)
mean_sim_test=np.concatenate((mean_sim_qd_test,mean_sim_qt_test),axis=1)
train_tfidf_similarity= hstack([train_tfidf,mean_sim_train])
test_tfidf_similarity= hstack([train_tfidf,mean_sim_test])
del train_tfidf
del mean_sim
gc.collect()

#The quadratic weighted kappa calculation function
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


#Grid search for tuning learning algorithm
param_grid = {
    'seed': [0],
    'loss': ['hinge'],
    'penalty': ['elasticnet'],
    'alpha': [0.01,0.001, 0.0001]
}
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
model = GridSearchCV(estimator = SGDClassifier(), param_grid=param_grid, scoring=kappa_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

model.fit(train_tfidf_similarity, relevance)
best_model = model.best_estimator_
best_model.fit(train_tfidf_similarity, relevance)
pr = best_model.predict(test_tfidf_similarity)

#method2


# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','head']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)


    

train = pd.read_csv('train.csv').fillna("")
test=pd.read_csv('test.csv').fillna("")


#porter stemmer
stemmer = PorterStemmer()
    ## Stemming functionality
class stemmerUtility(object):
        """Stemming functionality"""
@staticmethod
def stemPorter(review_text):
    porter = PorterStemmer()
    preprocessed_docs = []
    for doc in review_text:
       final_doc = []
       for word in doc:
         final_doc.append(porter.stem(word))
                        #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
         preprocessed_docs.append(final_doc)
    return preprocessed_docs
    
    
 
for i in range(len(train.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
    
for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
#create pipeline, fit, predit test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(s_data, s_labels)
    t_labels = clf.predict(t_data)
    

#Ensemble of two methods
pred= []
for i in range(len(preds)):
    x = (int(t_labels[i]) + pr[i])/2
    x = math.floor(x)
    pred.append(int(x))
    
        
    
  

# Create submission file
submission = pd.DataFrame({"id": idx, "prediction": pred})
    submission.to_csv("prediction.csv", index=False)
