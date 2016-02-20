from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import utility

# Load the dataset
docs_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)

tf_idf = TfidfVectorizer(preprocessor=utility.clean_word, use_idf=True)

# get Tf-idf-weighted document-term matrix
# rows : documents, columns : terms, each row is a vector for a document
tf_idf_data = tf_idf.fit_transform(docs_train.data)
print tf_idf_data.shape     # Get no of keywords
print docs_train.target[:20]

# returns top 'n' significant terms for the given category
def get_significant_terms(category, n):



    return


docs_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
n = 10
for category in categories:
    terms = get_significant_terms(category, n)
    print terms
