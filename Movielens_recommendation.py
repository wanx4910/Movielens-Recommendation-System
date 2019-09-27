from flask import Flask, render_template, request
from flask_table import Table, Col
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
import re
import time
from IPython.display import clear_output
import math
import json
from scipy.sparse import coo_matrix
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cosine

app=Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

class Results(Table):
    id = Col('Id', show=False)
    title = Col('Recommendations')

moviedata = pd.read_pickle('moviedata.pkl')
ratingdata = pd.read_pickle('ratingdata.pkl')
userdata = pd.read_pickle('userdata.pkl')
r1 = pd.read_pickle('r1_matrix.pkl')
userMovieRatingMatrix = pd.pivot_table(ratingdata, values='Rating', index=['User_id'], columns=['Item_id'])
userMovieRatingMatrix = userMovieRatingMatrix.fillna(0)
moviematrix = moviedata.drop(columns=['Movie_Title', 'Release_Date', 'Video_Release_Date', 'IMDB_URL'])
moviematrix = moviematrix.set_index('Movie_ID')

def alreadywatch (User):
    watch = ratingdata[ratingdata['User_id']==User].sort_values(by='Rating', ascending=False)
    item_list=list(watch.loc[:,'Item_id'])
    watchlist = moviedata[moviedata.index.isin(item_list)]
    return watchlist

def predictionforuser(User):
    r1[r1.index==User].T # extracts all the predicted ratings for user and transposes it
    filterlist = alreadywatch(User).index
    recommendation = r1[r1.index==User].T.sort_values(by=[User],ascending=False) # sorts the predicted ratings in descending order and saves it to a variable called recommendation
    recommendation_filtered = recommendation[~recommendation.index.isin(filterlist)] # produces the recommendation for user and excludes the movies he has already watch by title
    movie_recommendation = pd.DataFrame(columns=["Top 20 Recommended Movie Title"])
    i = 0
    for index in recommendation_filtered.index:
      movie_recommendation.loc[i] = [moviedata.at[index, "Movie_Title"]]
      i += 1
    return movie_recommendation.head(20)


def genknnmovie (User):
    watch = ratingdata[ratingdata['User_id']==User].sort_values(by='Rating', ascending=False)
    item_list=list(watch.loc[:,'Item_id'])
    watchlist = moviedata[moviedata.index.isin(item_list)].head(5)
    watchdf = watchlist['Movie_ID']
    k=0
    movie_recommendation_knn = pd.DataFrame(columns=['Top 20 Recommended Movie Title'])
    allMovie = pd.DataFrame(moviematrix.index)
    for i in list(watchdf):
      allMovie = allMovie[allMovie.Movie_ID!=i]
      allMovie['distance'] = allMovie['Movie_ID'].apply(lambda x: hamming(moviematrix.loc[i], moviematrix.loc[x]))
      KnearestMovies = allMovie.sort_values(['distance'], ascending = True)['Movie_ID'][:4]
      KnearestMovies = allMovie[~allMovie.index.isin(watch['Item_id'])]   
      for index in KnearestMovies.index:
        movie_recommendation_knn.loc[k] = [moviedata.at[index, "Movie_Title"]]
        k += 1
    return movie_recommendation_knn.head(20)

def cosinemovie (User):
    watch = ratingdata[ratingdata['User_id']==User].sort_values(by='Rating', ascending=False)
    item_list=list(watch.loc[:,'Item_id'])
    watchlist = moviedata[moviedata.index.isin(item_list)].head(5)
    watchdf = watchlist['Movie_ID']
    k=0
    movie_recommendation_cosine = pd.DataFrame(columns=['Top 20 Recommended Movie'])
    MovieItem = pd.DataFrame(userMovieRatingMatrix.T.index)
    for i in list(watchdf):
        MovieItem = MovieItem[MovieItem.Item_id!=i]
        MovieItem['similarity'] = MovieItem['Item_id'].apply(lambda x: cosine(userMovieRatingMatrix.T.loc[i], userMovieRatingMatrix.T.loc[x]))
        mostSimilarMovie = MovieItem[~MovieItem.index.isin(watch['Item_id'])]
        mostSimilarMovie = MovieItem.sort_values(['similarity'], ascending = False)['Item_id'][:4]   
        for index in mostSimilarMovie.index:
          movie_recommendation_cosine.loc[k] = [moviedata.at[index, "Movie_Title"]]
          k += 1
    return movie_recommendation_cosine

def hybridrecommendationmodel(User):
    hybridrec = pd.DataFrame(columns=['Recommendation'])
    cosinemov = cosinemovie(User)
    knnmov = genknnmovie(User)
    matfat = predictionforuser(User)
    c = 0
    d = 0
    e = 0
    for i in cosinemov.index:
        hybridrec.loc[i] = cosinemov['Top 20 Recommended Movie'].loc[i]
        c += 1
        if c == 3:
          break
    for j in knnmov.index:
        hybridrec = hybridrec.append({'Recommendation':knnmov['Top 20 Recommended Movie Title'].loc[j]},ignore_index=True)
        d += 1
        if d == 3:
          break
    for k in matfat.index:
        hybridrec = hybridrec.append({'Recommendation':matfat['Top 20 Recommended Movie Title'].loc[k]},ignore_index=True)
        e += 1
        if e == 3:
          break
    return hybridrec

@app.route('/result', methods = ['POST'])

def result():
    if request.method == 'POST':
        User = request.form.get('User')
        User = int(User)
        output = hybridrecommendationmodel(User)['Recommendation']
        table = Results(output)
        table.border = True
        return render_template("result.html", table=table)

if __name__ == "__main__":
    app.run()