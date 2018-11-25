
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


# read data: 
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

