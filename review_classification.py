######################################################### Classification of Reviews using Bag of Words Model #############################################


### Libraries for operations ###
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


#Reading into the panda dataframe  - quoting = 3 helps in ignoring the double quotes, header=0 ensures the first row consists of the headings
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

#Training data dimensions
training_shape = train.shape

#On printing the below line various HTML markups will be remained which have to be cleaned
#print train['review'][0]

#Removal of HTML markup by the use of BeautifulSoup

#This removes the HTML markup from the reviews
example = BeautifulSoup(train['review'][0],"html.parser")


### Function to convert the reviews into words after pre-processing ###
def conversion_to_words(review):

	#Removal of HTML markups 
	review = BeautifulSoup(review,"html.parser").get_text()

	#Removal of punctuations/non letters and cleaning the text
	new_review = re.sub("[^a-zA-Z]"," ",review)

	#Conversion of all the letters to lower cases and spliiting into words
	word_list = new_review.lower().split()

	#Removal of stop words from word_list
	stops = set(stopwords.words("english"))
	new_word_list = [w for w in word_list if not w in stops]

	#Conversion of the word list into text
	review_final = " ".join(new_word_list)
    
    #Returning th final cleaned-up review
	return review_final
 


#Number of labelled reviews available for training
number_of_reviews = train['review'].size

#Storing all the cleaned up reviews in a list
cleaned_reviews = []

for i in xrange(0,number_of_reviews):
	#Appending the cleaned up reviews to an empty list for further processing
	cleaned_reviews.append(conversion_to_words(train['review'][i]))



#Usage of Machine Learning Techniques to Classify the reviews

#Before that the reviews have to be converted to numerical / feature format to apply ML algorithms

#Initialization of CountVectorizer for features
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 1000)

#Curation of training features
training_features = vectorizer.fit_transform(cleaned_reviews)

#Conversion into numpy array for easy processing for ML algorithms
features = training_features.toarray()

#Initialising the Random Forest Classifier - 100 Trees 
forest = RandomForestClassifier(n_estimators = 100)

#Providing the model with training data - labelled
forest = forest.fit(features, train["sentiment"])

#Reading the test data into dataFrame
test= pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

n_reviews = len(test["review"])

clean_test_reviews = []

for j in xrange(0,n_reviews):
	clean_test_reviews.append(conversion_to_words(test['review'][j]))


test_data_features = vectorizer.fit_transform(clean_test_reviews)

test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

print result





