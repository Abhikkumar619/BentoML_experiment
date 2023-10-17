import bentoml
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model= KNeighborsClassifier()
iris= load_iris()

x=iris.data
y=iris.target

model.fit(x,y)

bentoml_model=bentoml.sklearn.save_model("KNeighbors", model)
print(f"Model saved : {bentoml_model}")
