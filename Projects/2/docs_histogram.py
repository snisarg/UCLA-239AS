from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as pyplot
import collections

def category_docs_frequency_count(category):

    category_train = fetch_20newsgroups(subset='train', categories=category, shuffle=True, random_state=42)
    frequency = []
    frequency = collections.Counter(category_train.target) #count frequency of category ids
    docs_count = sum(frequency.values()) #sum up frequencies of docs of a category
    return docs_count


twenty_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)

# Plot histogram of all categories
pyplot.hist(twenty_train.target)
pyplot.axis([0,20,0,1500])
pyplot.title('Distribution of Documents')
pyplot.xlabel('Classes')
pyplot.ylabel('Frequency Count')
pyplot.show()

# get frequency of docs in Comp Tech category
comp_tech_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.windows.x']
comp_tech_freq = category_docs_frequency_count(comp_tech_categories)
print comp_tech_freq

# get frequency of docs in Recreational category
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
rec_freq = category_docs_frequency_count(rec_categories)
print rec_freq
