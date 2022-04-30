import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
  
reviews.head()
ecommerce.head()
  
n_reviews = len(reviews)
n_users = len(reviews['userId'].unique())
  
print(f"Number of reviews: {n_reviews}")
print(f"Number of unique reviews: {n_reviews}")
print(f"Number of users: {n_users}")
print(f"Average number of reviews per user: {round(n_reviews/n_users, 3)}")
  
user_freq = reviews[['userId', 'ecommerce']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_reviews']
user_freq.head()
  
  
# Find malicious reviews:
mean_reviews = reviews.groupby('productId')[['reviews,]].mean()
  
from scipy.sparse import csr_matrix
  
def create_matrix(df):
      
    A = len(df['userId'].unique())
    B = len(df['reviewId'].unique())
      
    # Map Ids to ecommerce database
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(A))))
    review_mapper = dict(zip(np.unique(df["reviewId"]), list(range(B))))
      
    user_inv_mapper = dict(zip(list(range(A)), np.unique(df["userId"])))
    product_inv_mapper = dict(zip(list(range(B)), np.unique(df["productId"])))
      
    user_index = [user_mapper[i] for i in df['userId']]
    review_index = [review_mapper[i] for i in df['reviewId']]
  
    X = csr_matrix((df["review"], (ecommerce_index, user_index)), shape=(A, B))
      
    return X, user_mapper, review_mapper, product_mapper
  
create_matrix(reviews) = X, user_mapper, review_mapper, product_mapper
  
from sklearn.neighbors import NearestNeighbors

def find_maliciousspam_reviews(review_id, X, k, metric='cosine', show_distance=False):
      
    neighbour_ids = []
      
    product_id = review_mapper[review_id]
    review_vec = X[product_id]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    review_vec = review_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(review_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(review_id_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
  
  
product_reviews = dict(zip(ecommerce['reviewId'], ecommerce['userId']))
  
similar_ids = find_similar_reviews(product_id, X, k=10)

send_ID=print(mailto+" "+admin);
