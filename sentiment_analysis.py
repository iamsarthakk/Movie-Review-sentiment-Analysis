
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import re  ##Regular Expressions
import nltk ##Natural Language Toolkit
#pip install contractions
import contractions ## Short forms. eg. can't, etc
#pip install inflect
#import inflect

data_length = 1000  ##Rows of training set to be used.

##Reading File
dataset = pd.read_csv('train.tsv', delimiter = '\t') ### 
X = dataset.iloc[:data_length, :-1]
#y = dataset.iloc[:1000, 3]
y = dataset.iloc[:data_length, 3].values

#Sentiments Frequency Plot
dataset['Sentiment'].value_counts().plot(kind='bar')
plt.show()
print(dataset.describe())

#Dimension Reduction
##from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr #pearsonr(x, y) Pearson correlation coefficient and the p-value for testing non-correlation. 

correlation0 = pearsonr(dataset['PhraseId'], dataset['Sentiment'])
correlation1 = pearsonr(dataset['SentenceId'], dataset['Sentiment'])

dataset = dataset.iloc[:data_length, [2, 3]]  ##Dimension Reduction. Irrelevant Columns

##Stopword Elimination. Irrelevant words for sentence meaning.
#nltk.download('stopwords')  ##Articles, conjuctions, etc.. 
#nltk.download('wordnet')    ## Lemmatization. Word Net Lemmatizer

from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer ##Stemming
#from nltk.stem.snowball import SnowballStemmer ##Stemming
from nltk.stem import WordNetLemmatizer

corpus = [] ### Array to be filled with filtered reviews.

##Iterate through the rows.
for i in range(0, data_length):
    
    contractions.fix(dataset['Review'][i]) ##Remove Contractions
    
    ###re.sub(phrase, replace, string)
    review = re.sub('[^a-zA-Z ]', '', dataset['Review'][i]) ##Remove everything apart from letters and spaces from the 'i'th review.
    ###Does not subtract leters from a-z. Space should not be removed
    review = review.lower()  ## Convert all letters to lower case
    
    #review  = review.split()  ## Split string to list of words. Regular Expressions function.
    #nltk.download('punkt') ##The punkt.zip file contains pre-trained Punkt sentence tokenizer models that detect sentence boundaries.
    review = nltk.word_tokenize(review, language = 'english') ##Split larger text into segments.
   
    #ps = PorterStemmer() 
    #ss = SnowballStemmer("english")
    wnl = WordNetLemmatizer() ##wordnet
    
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review = [ss.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [wnl.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    ###Stemming/Lemmatization clubs words with same roots. 
    
    review = ' '. join(review)  ##Convert the list of words back into a string, separating each word with a "space".
    
    corpus.append(review)  ##Append the 'i'th review into the corpus array. 


#Creating the Bag of Words Model
##Convert a collection of text documents to a sparse matrix of token counts
from sklearn.feature_extraction.text  import CountVectorizer
countvect = CountVectorizer(max_features = 800)
X_matrix = countvect.fit_transform(corpus).toarray()
##print (countvect.vocabulary_)
##word2num_mapping = countvect.get_feature_names() ## list
##word2num_mapping = np.asarray(countvect.get_feature_names())
##w2n_freq = X_matrix.sum(axis=0)
word_frequency = pd.DataFrame(data = list(zip(X_matrix.sum(axis=0), np.asarray(countvect.get_feature_names()))), columns = ['Frequency', 'Word'])

##Normalized representation
from sklearn.feature_extraction.text import TfidfVectorizer
tfidvect = TfidfVectorizer(stop_words='english', max_features = 800)   ## Limit the number of words to 200
X_matrix_normalized = tfidvect.fit_transform(corpus).toarray()

#Data Visualization
Reviews_Dataframe = pd.DataFrame(data = list(zip(np.asarray(corpus), dataset['Sentiment'])), columns = ['Review', 'Sentiment'])
Reviews_Dataframe = Reviews_Dataframe[Reviews_Dataframe['Review'] != ''] ##Remove Rows with empty 'Review'
##Segment based on Sentiment Value
Sent_0, Sent_1, Sent_2, Sent_3, Sent_4 = [x for _, x in Reviews_Dataframe.groupby(Reviews_Dataframe['Sentiment'])]

##Visualization Function ## Top 20 words for a 'sentiment'
def sentiment_word_iteraction(dataframe):
    countvect = CountVectorizer(max_features = 20)
    dataframe_matrix = countvect.fit_transform(dataframe['Review']).toarray()
    word_freq_dataframe = pd.DataFrame(data = list(zip(dataframe_matrix.sum(axis=0), np.asarray(countvect.get_feature_names()))), columns = ['Frequency', 'Word'])
    plt.pie(word_freq_dataframe['Frequency'], labels = word_freq_dataframe['Word'])
    plt.show()
    plt.bar(word_freq_dataframe['Word'], word_freq_dataframe['Frequency'] )
    plt.show()
##---------------------------------##
    
sentiment_word_iteraction(Sent_0)
sentiment_word_iteraction(Sent_1)
sentiment_word_iteraction(Sent_2)
sentiment_word_iteraction(Sent_3)
sentiment_word_iteraction(Sent_4)

#File Outputs 
Reviews_Dataframe.to_csv('Reviews_Dataframe.csv')
word_frequency.to_csv('word_frequency.csv')
Sent_0.to_csv('Sent_0.csv')
Sent_1.to_csv('Sent_1.csv')
Sent_2.to_csv('Sent_2.csv')    
Sent_3.to_csv('Sent_3.csv')    
Sent_4.to_csv('Sent_4.csv')
X_matrix.to_csv('X_matrix.csv')
X_matrix_normalized.to_csv('X_matrix_normalized.csv')






