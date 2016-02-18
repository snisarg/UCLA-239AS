from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
docs_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)

tf_idf = TfidfVectorizer(stop_words='english',use_idf=True)

#get Tf-idf-weighted document-term matrix
#rows represent documents, columns represent terms, each row is a vector for a document
tf_idf_data = tf_idf.fit_transform(docs_train.data)
print tf_idf_data.shape     # Get no of keywords
