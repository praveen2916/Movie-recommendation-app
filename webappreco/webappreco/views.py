"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, url_for, request
from webappreco import app

"""
For Predictions.
"""
import numpy as  np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k


from lightfm.datasets import fetch_movielens
import pickle

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/predict', methods = ['POST'])
def predict():
    """Renders the contact page."""
    data = fetch_movielens(min_rating=2.0)

    if request.method == 'POST':
        model = request.form['model']
        if model =='Warp':
            filename = 'warp_model.pkl'
        elif model == 'K OS Warp':
            filename='koswarp_model.pkl'
        elif model == 'BPR':
            filename ='bpr_model.pkl'
        else:
            filename = 'logistic_model.pkl'

    loaded_model = pickle.load(open(filename, 'rb'))

    def sample_recommendations(loaded_model, data, user_ids):
        #number of users and movies in training data 
        n_users, n_items = data['train'].shape
    
        #generate recommendations for each user we input
        for user_id in user_ids:
        
            #movies they already like 
            known_positives_train = data['item_labels'][data['train'].tocsr()[user_id].indices]
            known_positives_test = data['item_labels'][data['test'].tocsr()[user_id].indices]
        
            #movies our model predicts they will like
            scores = loaded_model.predict(user_id, np.arange(n_items))
            #rank them in order of most liked to least liked 
            top_items =data['item_labels'][np.argsort(-scores)]
        
                
            #print out the results 
            #print("user %s" % user_id )
            #print("       Known Positives_train:")
        
            #for x in known_positives_train[:3]:
            #    print("       %s" % x)
            
            #print("      Recommended:")
        
            #for x in top_items[:3]:
            #    print("       %s" % x)
    
            return known_positives_train[:5],top_items[:5]

    if request.method == 'POST':
        comment = request.form['comment']
        user_list = [int(comment)]
        user_rated, my_reco = sample_recommendations(loaded_model, data, user_list)

        
    
    return render_template(
        'predict.html',
        umsg='Top 5 User Rated Movies',
        year=datetime.now().year,
        recomsg='Our Top 5 movie recommendations to user',
        urated = user_rated,
        mreco = my_reco 
            )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
