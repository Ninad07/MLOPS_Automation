from sklearn.datasets import load_wine
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbours import KNeighboursClassifier

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=0)

model = KNeighboursClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
result = model.score(x_test, y_test)
print(result)

fp = open("sk_accuracy.txt", "w")
fp.write(str(result*100))
fp.close()
