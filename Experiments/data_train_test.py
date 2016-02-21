from sklearn.datasets import fetch_20newsgroups


categories = ['soc.religion.christian', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print data_train.target[:40]
print data_train.target_names