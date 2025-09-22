# Vector_Space_Model
# Information Retrieval – Assignment 1  
**Course:** CSD358 – Information Retrieval Lab  
**Group Project**  

##  Overview  
This project implements a **ranked retrieval system using the Vector Space Model (VSM)**. The system indexes a given corpus, computes term weights using **lnc.ltc** weighting, and retrieves the **top-10 most relevant documents** for a free-text query.  

The project extends the standard VSM implementation with:  
- **Soundex-based spelling tolerance** (for handling spelling variations of names/words).  
- **Recency-based ranking** (combining TF-IDF similarity with document timestamp).  
- **Contextual snippets** (highlighting query terms in retrieved documents).  

---

##  Features  
- **Indexing**  
  - Tokenization, stopword removal, POS tagging.  
  - Lemmatization using WordNet.  
  - Dictionary & postings list with `(docID, term frequency)` tuples.  
  - Soundex codes are indexed alongside terms.  

- **Query Processing**  
  - Tokenization, lemmatization, and Soundex expansion of query terms.  
  - Query weights computed using `ltc` scheme:  
    ```
    tf = 1 + log10(freq)
    idf = log10(N / df)
    tf-idf = tf * idf
    ```  
  - Document weights computed using `lnc` scheme (log tf + cosine normalization, no idf).  
  - Cosine similarity scoring between query and documents.  

- **Ranking & Results**  
  - Documents ranked by **cosine similarity** (TF-IDF score).  
  - Recency incorporated into final ranking:  
    ```
    combined_score = α * cosine_score + β * recency_score
    ```  
    where recency favors recently modified files.  
  - Up to **top-10 documents** returned.  

- **Output**  
  - Results displayed with:  
    - Document name  
    - Combined score  
    - TF-IDF score  
    - Timestamp  
  - **Snippet generation**: Highlights query terms and nearby context.  

---

## Project Structure  
```
├── corpus/ 
├── main.py 
├── README.md 
```

## Getting Started

- Install Dependencies
   
   ```
   pip install nltk colorama
   ```

 - Modules Required

   ```
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   nltk.download('omw-1.4')   

   ```

 - Run The Program

   ```
    python main.py
   ```
   
## Example Query





