import numpy as  np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

from lightfm.datasets import fetch_movielens
import pickle
from sklearn.externals import joblib


#fetch data and format it

data = fetch_movielens(min_rating=2.0)

#create model
model = LightFM(loss='bpr')

model.fit(data['train'],epochs=30,num_threads=2)



# save the model to disk
filename = 'bpr_model.pkl'
pickle.dump(model, open(filename, 'wb'))

