'''
Created on 20-Dec-2018

@author: Vishnu
'''

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import re

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def Keywords(text):
    try:
        doc_complete = [text]
        doc_clean = [clean(doc).split() for doc in doc_complete]
        dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=50)
        key = ldamodel.print_topics(num_topics=1, num_words=10)
        key = re.split(r'[^\w]', key[0][1])
        keywords = [i for i in key if re.search(r'[a-zA-Z]', i)]
    except:
        keywords = "No keyword found"
    result = {}
    result['result'] = keywords
    result['success'] = True
    return result
