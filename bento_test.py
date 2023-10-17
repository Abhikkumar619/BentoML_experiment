import bentoml

clf=bentoml.sklearn.get('kneighbors').to_runner()

clf.init_local()

result=clf.predict.run([[2.3, 1.2,4.3,6.3]])
print(result)