from nltk import *
from nltk.corpus import *
import os
from collections import Counter
import math  
import datetime
from colorama import Fore, Style
from nltk import pos_tag

#novelty part 1
def get_snippet(file_path, query_token, window=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    #taking the whole file
    text_lower = text.lower()
    x = ""
    
    #printing the line where any one of the query terms appear
    for term in query_token:
        if term in text_lower.split():
            idx = text_lower.index(term)
            start = max(0, idx - window)
            end = min(len(text), idx + window)
            snippet = text[start:end].replace("\n", " ")
            x = "..." + snippet + "..."
            break
    #if no word in file matches the query terms- then just print the first few lines
    if not x:
        x = (text[:100] + "...").replace("\n", " ")
    #highlighting the query terms
    for word in x.split():
        clean_word = ''.join(ch for ch in word if ch.isalnum())  # remove punctuation
        lemma = word_len.lemmatize(clean_word.lower())
        if lemma in query_token or soundex(lemma) in query_token:
            print(Style.BRIGHT + Fore.BLUE + word + Style.RESET_ALL, end=" ")
        else:
            print(Fore.BLUE + word + Style.RESET_ALL, end=" ")

def soundex(name):
    if not name:
        return "0000"
    codes = {
        "bfpv": "1",
        "cgjkqsxz": "2",
        "dt": "3",
        "l": "4",
        "mn": "5",
        "r": "6"
    }
    soundex_code = name[0]

    #replace letters with digits
    for char in name[1:]:
        for key, val in codes.items():
            if char in key:
                code = val
                break
        else:
            code = ""
        if code != soundex_code[-1]:
            soundex_code += code

    soundex_code = soundex_code[0] + ''.join([c for c in soundex_code[1:] if c.isdigit()])
    soundex_code = (soundex_code + "000")[:4]
    return soundex_code

#novelty part 2
#categorizing each term as a verb, noun, adverb, etc
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  #default
    
word_len= WordNetLemmatizer()

#documents ki list
docs=os.listdir("corpus")
N=len(docs)

#stop words
stopword_s = set(stopwords.words("english"))

postings={}
i=0
tf_for_each={}

#for every document- preprocess
for file in os.listdir("corpus"):
    with open(os.path.join("corpus", file), "r", encoding="utf-8") as f:
        terms=word_tokenize(f.read().lower())
        tagged_words = pos_tag(terms)
        x=[]
        for word,t in tagged_words:
            if word.isalpha() and word not in stopword_s:
                lemma = word_len.lemmatize(word, pos=get_wordnet_pos(t))
                x.append(lemma)
                x.append(soundex(lemma))
        docs_tf=Counter(x)
        sorted_tf = dict(sorted(docs_tf.items()))
        tf_for_each[i]=sorted_tf
        for term, freq in docs_tf.items():
            postings.setdefault(term, []).append((i+1, freq))
        i+=1
#after this i have posting list,tf for all documents
 
#query ka pre-processing
query= input("Enter the query: ")
query_words=word_tokenize(query.lower())
tagged_words = pos_tag(query_words)
query_token = []    
for x, tag in tagged_words:
    if x.isalpha() and x not in stopword_s:
        lemma = word_len.lemmatize(x, pos=get_wordnet_pos(tag))
        query_token.append(lemma)
        query_token.append(soundex(lemma))
tf_query = Counter(query_token)
#after this i have a processed query list and tf of the query

#calculation of tf-idf scores
log_tf={}  
idf_query={}
weight_query={}
for term in query_token:
    if term in postings:
        log_tf[term] = 1 + math.log10(tf_query[term])
        idf_query[term] = math.log10(N / len(postings[term]))
        weight_query[term]=idf_query[term]*log_tf[term]
        
#calculation of tdf-if for every documents
values={}
for i in tf_for_each:
    tf_doc=tf_for_each[i]
    doc_log_tf={}
    doc_norm={}
    normalized=0
    cos=0
    for j in tf_doc:
        doc_log_tf[j]=(1+math.log10(tf_doc[j]))
        normalized+=(doc_log_tf[j]**2)
    normalized=math.sqrt(normalized)
    for k in tf_doc:
        doc_norm[k]=doc_log_tf[k]/normalized
    for l in weight_query:
        if l in tf_doc:
            cos+=weight_query[l]*doc_norm[l]
    values[i]=cos

#add the document name, score and its timestamp to a list
top_docs = []
for doc_id, score in values.items():
    file = docs[doc_id]
    path = os.path.join("corpus", file)
    mtime = os.path.getmtime(path)
    top_docs.append((file, score, mtime))
import time

#novelty part 3
now = time.time()  #current timestamp
alpha = 0.9        #weight for cosine score
beta = 0.1         #weight for timestamp

combined_docs = []
for file, score, mtime in top_docs:
    recency_score = 1 / (1 + (now - mtime)/86400)  # 86400 seconds in a day
    combined_score = alpha*score + beta*recency_score #new score= aplha.tf-idf_score+beta.recency_score
    combined_docs.append((file, combined_score, score, mtime))

# sort by combined score
combined_docs = sorted(combined_docs, key=lambda x: -x[1])[:10]

#format and print the top 10 documents
i=1
print()
print(Style.BRIGHT +"-"*50+"RESULTS"+"-"*50+ Style.RESET_ALL)
for file, comb_score, score, mtime in combined_docs:
    dt = datetime.datetime.fromtimestamp(mtime)
    timestamp_with_ms = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

    file_path = os.path.join("corpus", file)
    print(Style.BRIGHT+Fore.RED + str(i)+"." + Style.RESET_ALL,end="")
    print(Fore.RED,Style.BRIGHT,file.capitalize(),"\t",f"Combined Score: {comb_score:.4f}", "\t",f"Tf-Idf Score: {score:.4f}","\t","Timestamp:", timestamp_with_ms,Style.RESET_ALL)
    get_snippet(file_path, query_token)
    print("\n")
    i+=1