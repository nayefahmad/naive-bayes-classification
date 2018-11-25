
#**************************************************
# Naive Bayes Classification for Sentiment Analysis 
# of Movie Reviews
# 2018-11-24 
#**************************************************

library(here)
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
# Library for parallel processing
# library(doMC)
# registerDoMC(cores=detectCores())



# Reference: https://rpubs.com/cen0te/naivebayes-sentimentpolarity 


# read in data: ----------- 
df1.reviews <- read_data(here("data", 
                           "movie-pang02.csv"))
glimpse(df1.reviews)

# randomize data: 
set.seed(1)
df1.reviews <- df1.reviews[sample(nrow(df1.reviews)), ]
df1.reviews <- df1.reviews[sample(nrow(df1.reviews)), ]
glimpse(df1.reviews)


# Convert the 'class' variable from character to factor.
df1.reviews$class <- as.factor(df1.reviews$class)


# Bag of words tokenization -----
# ?Corpus
corpus <- Corpus(VectorSource(df1.reviews$text))

# str(corpus, max.level = 1)  # not useful


# Inspect the corpus: --------
corpus
## <<VCorpus>>
## Metadata:  corpus specific: 0, document level (indexed): 0
## Content:  documents: 2000
inspect(corpus[1:3])

corpus[[2]]



# data cleanup: --------
corpus.clean <- 
    corpus %>% 
    tm_map(content_transformer(tolower)) %>% 
    tm_map(removePunctuation) %>% 
    tm_map(removeNumbers) %>% 
    tm_map(removeWords, 
           stopwords(kind = "en")) %>% 
    tm_map(stripWhitespace)

inspect(corpus.clean[1:3])


# matrix representation of bag of words: -------
dtm <- DocumentTermMatrix(corpus.clean)

inspect(dtm[1:5, 50:70])



# partitioning the data: --------
# Next, we create 75:25 partitions of the dataframe, corpus 
# and document term matrix for training and testing purposes.

df2.train <- df1.reviews[1:1500, ]
df2.test <- df1.reviews[1501:2000, ]

dtm.train <- dtm[1:1500, ] 
dtm.test <- dtm[1501:2000, ]

corpus.clean.train <- corpus.clean[1:1500]
corpus.clean.test <- corpus.clean[1501:2000]


# Feature selection: --------
dim(dtm.train)  

# 38k features is too much; drop the ones that are 
# used in less than 5 documents 

# ?findFreqTerms
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

# now subset dtm.train again, using the terms in fivefreq: 
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, 
                                   control = list(dictionary = fivefreq))

# inspect(dtm.train.nb[1:5, 100:120])
dim(dtm.train.nb)

# do the same for the test set: 
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test,
                                  control=list(dictionary = fivefreq))

dim(dtm.test.nb)


# Boolean feature multinomial naive bayes -------
# In this method, the term frequencies are replaced by Boolean 
# presence/absence features. The logic behind this being that
# for sentiment classification, word occurrence matters more 
# than word frequency.

# Function to convert the word frequencies to yes (presence) and no
# (absence) labels
convert_count <- function(x) {
    y <- ifelse(x > 0, 1,0)
    y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
    y
}


trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

str(trainNB)
dim(trainNB)
trainNB[1, 1:20]



# training the naive bayes model --------
# To train the model we use the naiveBayes function from 
# the ‘e1071’ package. Since Naive Bayes evaluates products 
# of probabilities, we need some way of assigning non-zero
# probabilities to words which do not occur in the sample.
# We use Laplace 1 smoothing to this end.

# Train the classifier: -----
system.time(classifier <- naiveBayes(trainNB, 
                                      df2.train$class, 
                                      laplace = 1) )

# str(classifier, max.level = 1)
# summary(classifier)


# test the predictions: ----
# Use the NB classifier we built to make predictions on the test set.
system.time(pred <- predict(classifier,
                            newdata=testNB))


