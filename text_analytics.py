# %% [markdown]
# ## Import required libraries

# %%
import nltk
import re

# %%
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("wordnet")

# %%
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

# %% [markdown]
# ## Reading input1.txt and input2.txt

# %%
with open('A7Input1.txt', 'r') as f:
    doc1 = f.read()

print(doc1)

# %%
with open('A7Input2.txt', 'r') as f:
    doc2 = f.read()

print(doc2)

# %% [markdown]
# ## Using re library to remove fullstop, etc

# %%
doc1 = re.sub('[\W_]+', ' ', doc1)
print(doc1)

# %%
doc2 = re.sub('[\W_]+', ' ', doc2)
print(doc2)

# %% [markdown]
# ## using the nltk library to tokenize doc1 and doc2 content

# %%
doc1Token = nltk.word_tokenize(doc1)
doc2Token = nltk.word_tokenize(doc2)

# %%
print(doc1Token)

# %%
print(doc2Token)

# %% [markdown]
# ## POS Tagging

# %%
doc1POS = nltk.pos_tag(doc1Token)

# %%
doc2POS = nltk.pos_tag(doc2Token)

# %%
print(doc1POS)

# %%
print(doc2POS)

# %% [markdown]
# ## Removing stop-words

# %%
stop_words_all = nltk.corpus.stopwords.words()
print(stop_words_all)

# %%
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

# %%
token1 = [token for token in doc1Token if token not in stop_words]
print(token1)

# %%
print("Before removal of stopwords : ", len(doc1Token))
print("After removal of stopwords : ", len(token1))

# %%
token2 = [token for token in doc2Token if token not in stop_words]
print(token2)

# %%
print("Before removal of stopwords : ", len(doc2Token))
print("After removal of stopwords : ", len(token2))

# %% [markdown]
# ## Stemming

# %%
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_token_doc1 = [stemmer.stem(token) for token in token1]

# %%
print(token1)

# %%
print(stemmed_token_doc1)

# %%
print("Before stemming : ",len(token1))
print("AFter stemming : ", len(stemmed_token_doc1))

# %%
different = 0
for i in range(539):
    if (token1[i] != stemmed_token_doc1[i]):
        print(token1[i], stemmed_token_doc1[i])
        different+=1

# %%
print("Number of differences found while comparing token1 and stemmed_token_doc1 : ", different)

# %%
# for document 2
stemmed_token_doc2 = [stemmer.stem(token) for token in token2]

# %%
print(token2)

# %%
print(stemmed_token_doc2)

# %%
print("Before stemming : ",len(token2))
print("AFter stemming : ", len(stemmed_token_doc2))

# %%
different = 0
for i in range(356):
    if (token2[i] != stemmed_token_doc2[i]):
        print(token2[i], stemmed_token_doc2[i])
        different+=1


# %%
print("Number of differences found while comparing token2 and stemmed_token_doc2 : ", different)

# %% [markdown]
# ## Lemmatizer

# %%
from nltk.stem import WordNetLemmatizer

# %%
lemmatizer = WordNetLemmatizer()
lemmatized_token_doc1 = [lemmatizer.lemmatize(token) for token in token1]
print(lemmatized_token_doc1)

# %%
print("Before lemmatizing : ", len(lemmatized_token_doc1))
print("After lemmatizing : ", len(token1))

# %%
different = 0
for i in range(356):
    if (lemmatized_token_doc1[i] != token1[i]):
        print(token1[i], lemmatized_token_doc1[i])
        different += 1

# %%
print("Number of differences found while comparing token1 and lemmatized_token_doc1 : ", different)

# %%
# for doc2
lemmatized_token_doc2 = [lemmatizer.lemmatize(token) for token in token2]
print(lemmatized_token_doc2)

# %%
print("Before lemmatizing : ", len(lemmatized_token_doc2))
print("After lemmatizing : ", len(token2))

# %%
different = 0
for i in range(356):
    if (lemmatized_token_doc2[i] != token2[i]):
        print(token2[i], lemmatized_token_doc2[i])
        different += 1

# %%
print("Number of differences found while comparing token2 and lemmatized_token_doc2 : ", different)

# %% [markdown]
# ## Calculating Term Frequency

# %%
# Term Frequency

def tf_calculate(document):
    freq = {}
    for word in document:
        if (word not in freq):
            freq[word] = document.count(word) / len(document)
    
    return freq

tf1 = tf_calculate(token1)
# print(tf1)
for key, value in tf1.items():
    print(key, value)

# %%
tf2 = tf_calculate(token2)
# print(tf2)
for key, value in tf2.items():
    print(key, value)

# %%
for key, value in tf1.items():
    print(key, value)

# %% [markdown]
# ## Calculating Inverse Document Frequency

# %%
# inverse document frequency (idf)

import math

# def idf_calculate():
#     N = 2 # number of documents
#     all_tokens = token1 + token2
#     list(set(all_tokens))
#     f = 1
#     idf = {}
#     for word in all_tokens:
#         if (word in token1):
#             f+=1
#         if (word in token2):
#             f+=1
#         idf[word] = math.log(N / f)
    
#     return idf

def idf_calculate():
    N = 2 # number of documents
    all_tokens = token1 + token2
    list(set(all_tokens))
    f = 0
    idf = {}
    for word in all_tokens:
        if (word in token1):
            f+=1
        if (word in token2):
            f+=1
        if (f==0):
            f+=1
        idf[word] = math.log(N / f)
    
    return idf


idf = idf_calculate()
print(idf)
        

# %% [markdown]
# ## tf-idf calculation

# %%
# tf-idf calculation

print(tf1)

# %%
print(idf)

# %%
tf_idf_doc1 = {}
for word in token1:
    if word not in tf_idf_doc1:
        tf_idf_doc1[word] = tf1[word] * idf[word]
print(tf_idf_doc1)

# %%
tf1

# %%
type(tf1)

# %%


# %%
lst = [2, 4, 1, 6, 9]
sorted(lst)

# %%
sorted(tf1.values(), reverse = True)

# %%
x = sorted(tf1.values(), reverse = True)
print(x[0:10])
# print(tf1)
top_three = x[0:4]
i = 0
lst = []
for key, value in tf1.items():
    if (value == top_three[i]):
        i+=1
        lst.append(key)
print(lst)

# %%
descending_order = sorted(tf1.items(), key = lambda x : x[1], reverse = True)
descending_order = dict(descending_order)
print(descending_order)



**********************************************************************************************************************

# %% [markdown]
# # 7. Text Analytics
# 1. Extract Sample document and apply following document preprocessing
# methods: Tokenization, POS Tagging, stop words removal, Stemming and
# Lemmatization.
# 2. Create representation of document by calculating Term Frequency and Inverse
# Document Frequency.
Text analytics, also known as text mining or natural language processing (NLP), is the process of extracting useful insights and information from unstructured text data.
# %%
import nltk
import string
import math

# %%
# Read the docs
# Remove all non-ASCII characters

with open( "doc_01" , "r" ) as file:
    doc_1 = file.read()

with open( "doc_02" , "r" ) as file:
    doc_2 = file.read()

alphabet = string.printable
def clean_doc( doc: str ) -> str:
    return ''.join( [ c for c in doc.lower() if c in alphabet ] )

doc_1 = clean_doc( doc_1 )
doc_2 = clean_doc( doc_2 )

# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# %% [markdown]
# ## 7.1. Document Preprocessing

# %% [markdown]
# ### 7.1.1. Tokenization

# %%
"""
word tokenization
default tokenizer: Penn Treebank Tokenizer
Ref: https://docs.ropensci.org/tokenizers/reference/ptb-tokenizer.html
This tokenizer uses regular expressions to tokenize text similar to the tokenization used in the Penn Treebank. It assumes that text has already been split into sentences. 
The tokenizer does the following:
- splits common English contractions, e.g. don't is tokenized into do n't and they'll is tokenized into -> they 'll,
- handles punctuation characters as separate tokens,
- splits commas and single quotes off from words, when they are followed by whitespace,
- splits off periods that occur at the end of the sentence.
"""
tokens_1 = nltk.word_tokenize( doc_1 )
tokens_2 = nltk.word_tokenize( doc_2 )
print( tokens_1 )

# %% [markdown]
# ### 7.1.2. POS Tagging

# %%
# default pos tagger: Perceptron Tagger
# Ref: https://explosion.ai/blog/part-of-speech-pos-tagger-in-python/
pos_tags = dict( zip( tokens_1 , nltk.pos_tag(tokens_1) ) )
print( pos_tags )

# %% [markdown]
# ### 7.1.3. Sentence Tokenization
# 

# %%
"""
sentence tokenization
default tokenizer: Punkt tokenizer
Ref: Unsupervised Multilingual Sentence Boundary Detection (Kiss and Strunk (2005)
"""
print( nltk.sent_tokenize( doc_1 ) )

# %% [markdown]
# ### 7.1.4. Lemmatization

# %%
# lemmatization
# Lemmatization reduces words to their base or dictionary form
# Lemmatization, on the other hand, involves reducing words to their base or dictionary form (lemma) based on their part of speech.
lemmatizer = nltk.stem.WordNetLemmatizer()
tokens_1 = [ lemmatizer.lemmatize(token) for token in tokens_1 ]
tokens_2 = [ lemmatizer.lemmatize(token) for token in tokens_2 ]
print( tokens_1 )

# %% [markdown]
# ### 7.1.5. Stemming

# %%
# The Porter Stemmer is an algorithm that applies a series of rules to remove suffixes from words to obtain their root forms.
stemmer = nltk.stem.PorterStemmer()
tokens_1 = [ stemmer.stem( token ) for token in tokens_1 ]
tokens_2 = [ stemmer.stem( token ) for token in tokens_2 ]
print( tokens_1 )

# %% [markdown]
# ### 7.1.6. Stop Word Removal

# %%
# stop word removal
def remove_stop_words( tokens ):
    return [ token for token in tokens if token not in nltk.corpus.stopwords.words('english') ]

tokens_1 = remove_stop_words( tokens_1 )
tokens_2 = remove_stop_words( tokens_2 )
print( tokens_1 )

# %% [markdown]
# ## 7.2. TF and IDF

# %%
# Returns a map containing term-frequencies of each token
# present in `doc`
# tf( token ) = freq( token ) / num_tokens_in_doc
def term_freq( doc_tokens ):
    N = len( doc_tokens )
    token_freq = dict( [ ( token , 0 ) for token in doc_tokens ] )
    for token in doc_tokens:
        token_freq[ token ] += 1
    tf = dict( [ ( token , count / N ) for token , count in token_freq.items() ] )
    return tf

tf_1 = term_freq( tokens_1 )
tf_2 = term_freq( tokens_2 )

# %%
# Calculate inverse-document frequency
# IDF( token ) = log( N / (num_docs_where_token_occurs) )
all_tokens = tokens_1 + tokens_2 # list concatenation
def inverse_doc_freq():
    N = 2
    idf = {}
    for token in all_tokens:
        token_doc_freq = 0
        if token in tokens_1:
            token_doc_freq += 1
        if token in tokens_2:
            token_doc_freq += 1
        idf[ token ] = math.log( N / token_doc_freq )
    return idf

idf = inverse_doc_freq()

# %%
# TFIDF( token ) = TF( token ) * IDF( token )
doc_1_repr = []
for token in tokens_1:
    doc_1_repr.append( tf_1[ token ] * idf[token] )
doc_2_repr = []
for token in tokens_2:
    doc_2_repr.append( tf_2[ token ] * idf[token] )

# %%
print( doc_1_repr )

# %%







