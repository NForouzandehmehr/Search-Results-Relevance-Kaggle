library(readr)
library(Metrics)
library(tm)
library(SnowballC)
library(e1071)
library(Matrix)
library(SparseM)

#Loading the training and test file
train = read_csv("C:/Users/najmeh.f/Documents/R/kaggle/train.csv")
test  = read_csv("C:/Users/najmeh.f/Documents/R/kaggle/test.csv")

# Droping labels
ids = test$id
rtrain = nrow(train)
rtest =nrow(test)
relevance = train$median_relevance
variance = train$relevance_variance
train$median_relevance = NULL
train$relevance_variance = NULL

# Combine train and test set 
cdata=rbind(train,test)


# Create tfidf vectors for query, product_title and product_description

data_corpus <- Corpus(VectorSource(cdata$query))
dtm<-DocumentTermMatrix(data_corpus,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.99)
df_q<-Matrix(as.matrix(dtm),sparse=T)
df_q<-as.data.frame(as.matrix(dtm))
colnames(df_q)=paste("q_",colnames(df_q),sep="")

data_corpus <- Corpus(VectorSource(cdata$product_title))
dtm<-DocumentTermMatrix(data_corpus,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.95)
df_pt<-Matrix(as.matrix(dtm),sparse=T)
df_pt<-as.data.frame(as.matrix(dtm))
colnames(df_pt)=paste("pt_",colnames(df_pt),sep="")

data_corpus <- Corpus(VectorSource(cdata$product_description))
dtm<-DocumentTermMatrix(data_corpus,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.9)
df_pd<-as.data.frame(as.matrix(dtm))
colnames(df_pd)=paste("pd_",colnames(df_pd),sep="")

cdata=cbind(df_q,df_pt,df_pd)

rm(df_q)
rm(df_pt)
rm(df_pd)
rm(data_corpus)
rm(corpus)
rm(dtm)

# Stacking learners using H2O Ensemble
train = cdata[1:10158,]
test = cdata[10159:32671,]
rm(cdata)

library(h2o) 
library(h2oEnsemble)
library(SuperLearner) 
localH2O <- h2o.init(ip = "localhost", port = 54321)
train2 <- as.h2o(localH2O, object = train )
test2 <- as.h2o(localH2O, test)

family <- "gaussian"
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.randomForest.1 <- function(..., ntrees = 1000, nbins = 100, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.glm.1 <- function(...,family = c("gaussian"), alpha = 1.0, prior =NULL) {h2o.glm.wrapper(...,family=family, alpha = alpha, prior= prior)}
h2o.glm.2 <- function(...,family = c("gaussian"), alpha = 0.003, prior =NULL) {h2o.glm.wrapper(...,family=family, alpha = alpha, prior= prior)}
learner <- c("h2o.randomForest.1","h2o.deeplearning.1", "h2o.deeplearning.2" )

metalearner <- c("SL.glm")
print(length(metalearner)>1)
print(!is.character(metalearner))
print(!exists(metalearner))
y <- "relevance"
x <- setdiff(names(train2), y)

fit <- h2o.ensemble(x = x, y = y, data = train2, family = family, learner = learner, metalearner = metalearner)
pred <- predict(object = fit, newdata = test2)
predictions=as.data.frame(pred$pred)[,1]

rm(test2)
rm(train2)
rm(trains)
# Create submission file
submission = data.frame(id=test_id, prediction = predictions)
write.csv(Newsubmission,"prediction.csv",row.names=F)
