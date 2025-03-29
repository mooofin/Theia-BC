import pickle
from KNN import knn
model_name = "knn_model.pkl"
with open(model_name,"wb") as file :
    pickle.dump(knn,file)
print(f"model working as {model_name}")



