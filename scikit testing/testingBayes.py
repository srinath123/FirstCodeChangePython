from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
y_pred = mnb.fit(iris.data,iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"  % (iris.data.shape[0],(iris.target != y_pred).sum()))
