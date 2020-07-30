import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gensim  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from groa_ds_api.models import *
from groa_ds_api.utils.database_util import DatabaseUtility
from groa_ds_api.utils.info_util import InfoUtility


class RecommendationUtility(object):
    """ Movie utility class that uses a W2V model to recommend movies
    based on the movies a user likes and dislikes """

    def __init__(self, model_path: str, db_tool: DatabaseUtility, info_tool: InfoUtility):
        """ Initialize model with name of .model file """
        self.model_path = model_path
        self.model = self.__load_model()
        self.db = db_tool
        self.info_tool = info_tool

    # ------- Start Private Methods -------
    def __load_model(self):
        """ Get the model object for this instance, loading it if it's not already loaded """
        w2v_model = gensim.models.Word2Vec.load(self.model_path)
        # Keep only the normalized vectors.
        # This saves memory but makes the model untrainable (read-only).
        w2v_model.init_sims(replace=True)
        self.model = w2v_model
        return self.model

    def __get_JSON(self, rec_df: pd.DataFrame):
        """ 
        Turn predictions into JSON
        Callers:
            - get_recommendations
            - get_similar_movies
            - get_movie_list
        """
        rec_json = []

        for i in range(rec_df.shape[0]):
            rec = dict(rec_df.iloc[i].to_dict())
            if 'score' in rec:
                rec['score'] = float(rec['score']) if not isinstance(
                    rec['score'], str) else 0.0
            rec['year'] = int(rec['year']) if not isinstance(
                rec['year'], str) else 0
            rec['genres'] = rec['genres'].split(',')
            rec['avg_rating'] = float(rec['avg_rating']) if not isinstance(
                rec['avg_rating'], str) else 0.0
            rec_json.append(rec)

        return rec_json

    def __prep_data(self, ratings_df: pd.DataFrame,
                    watched_df: pd.DataFrame = None,
                    watchlist_df: pd.DataFrame = None,
                    not_watchlist_df: pd.DataFrame = None,
                    good_threshold: int = 4,
                    bad_threshold: int = 3):
        """
        Converts dataframes to lists of movie_ids.
        Callers:
            - get_recommendations
        """
        try:
            # split according to user rating
            good_df = ratings_df[ratings_df['rating'] >= good_threshold]
            bad_df = ratings_df[ratings_df['rating'] <= bad_threshold]
            neutral_df = ratings_df[(ratings_df['rating'] > bad_threshold) & (
                ratings_df['rating'] < good_threshold)]
            # add not_watchlist to bad_df
            bad_df = pd.concat([bad_df, not_watchlist_df])
            # convert dataframes to lists
            good_list = good_df['movie_id'].to_list()
            bad_list = bad_df['movie_id'].to_list()
            neutral_list = neutral_df['movie_id'].to_list()

        except Exception:
            raise Exception("Error making good, bad and neutral list")

        ratings_dict = pd.Series(
            ratings_df['rating'].values, index=ratings_df['movie_id']).to_dict()

        if watched_df is not None:
            # Construct list of watched movies that aren't rated "good" or "bad"
            hist_list = ratings_df[~ratings_df['movie_id'].isin(
                good_list+bad_list)]['movie_id'].to_list()
        else:
            hist_list = neutral_list

        if watchlist_df is not None:
            # gets list of movies user wants to watch for validation
            val_list = watchlist_df['movie_id'].tolist()
        else:
            val_list = []

        return (good_list, bad_list, hist_list, val_list, ratings_dict)

    def __predict(self, input: List,
                  bad_movies: List = [],
                  hist_list: List = [],
                  val_list: List = [],
                  ratings_dict: Dict = {},
                  checked_list: List = [],
                  rejected_list: List = [],
                  n: int = 50,
                  harshness: int = 1):
        """
        Returns a list of recommendations, given a list of movies.
        Callers:
            - get_movie_list
            - get_recommendations
        Returns:
        A list of tuples
            (Movie ID, Similarity score)
        """

        clf = self.model
        # list for storing duplicates for scoring
        dupes = []

        def _aggregate_vectors(movies: List, feedback_list: List = []):
            """ Gets the vector average of a list of movies """
            movie_vec = []
            for i in movies:
                try:
                    # get the vector for each movie
                    m_vec = clf.wv.__getitem__(i)
                    if ratings_dict:
                        try:
                            # get user_rating for each movie
                            r = ratings_dict[i]
                            # Use a polynomial to weight the movie by rating.
                            # This equation is somewhat arbitrary. I just fit a polynomial
                            # to some weights that look good. The effect is to raise
                            # the importance of 1, 2, 9, and 10 star ratings to about 1.8.
                            w = ((r**3)*-0.00143) + ((r**2)*0.0533) + \
                                (r*-0.4695) + 2.1867
                            m_vec = m_vec * w
                        except:
                            continue
                    movie_vec.append(m_vec)
                except KeyError:
                    continue
            if feedback_list:
                for i in feedback_list:
                    try:
                        f_vec = clf.wv.__getitem__(i)
                        # weight feedback by changing multiplier here
                        movie_vec.append(f_vec*1.8)
                    except:
                        continue
            return np.mean(movie_vec, axis=0)

        def _similar_movies(v, n, bad_movies: List[str] = []):
            """ Aggregates movies and finds n vectors with highest cosine similarity """
            if bad_movies:
                v = _remove_dislikes(bad_movies, v, harshness=harshness)
            return clf.wv.similar_by_vector(v, topn=n+1)[1:]

        def _remove_dupes(recs, input: List[str], bad_movies: List[str], hist_list: List[str] = [], feedback_list: List[str] = []):
            """ Remove any recommended IDs that were in the input list """
            all_rated = input + bad_movies + hist_list + feedback_list
            nonlocal dupes
            dupes = [x for x in recs if x[0] in input]
            return [x for x in recs if x[0] not in all_rated]

        def _remove_dislikes(bad_movies: List[str], good_movies_vec: List[str], rejected_list: List[str] = [], harshness: int = 1):
            """ Takes a list of movies that the user dislikes.
            Their embeddings are averaged,
            and subtracted from the input. """
            bad_vec = _aggregate_vectors(bad_movies, rejected_list)
            bad_vec = bad_vec / harshness
            return good_movies_vec - bad_vec

        aggregated = _aggregate_vectors(input, checked_list)
        recs = _similar_movies(aggregated, n, bad_movies)
        recs = _remove_dupes(recs, input, bad_movies,
                             hist_list, checked_list + rejected_list)
        return recs

    def __get_list_preview(self, data: Tuple[int, str, bool]):
        """ 
        Turns list preview sql into an object.
        Callers:
            - get_user_lists
            - get_all_lists 
        """
        return {
            "list_id": data[0],
            "name": data[1],
            "private": data[2]
        }
    # ------- End Private Methods -------

    # ------- Start Public Methods -------
    def get_most_similar_title(self, movie_id: str, id_list: List[str]):
        """ Get the title of the most similar movie to movie_id from id_list """
        clf = self.model
        vocab = clf.wv.vocab
        if movie_id not in vocab:
            return ""
        # ensure all in vocab
        id_list = [movie_id for movie_id in id_list if movie_id in vocab]
        match = clf.wv.most_similar_to_given(movie_id, id_list)
        return match

    def get_similar_movies(self, payload: SimInput):
        """ Gets movies with highest cosine similarity """
        # request data
        movie_id = payload.movie_id
        n = payload.num_movies
        # get model
        clf = self.model
        try:
            m_vec = clf.wv.__getitem__(movie_id)
            movies_df = pd.DataFrame(clf.wv.similar_by_vector(
                m_vec, topn=n+1)[1:], columns=['movie_id', 'score'])
            result_df = self.info_tool.get_info(movies_df)
            result_df = result_df.fillna("None")
            return {
                "data": self.__get_JSON(result_df)
            }
        except:
            return {
                "data": []
            }

    def get_recommendations(self, payload: RecInput, background_tasker):
        """ Uses user's ratings to generate recommendations """
        # request data
        user_id = payload.user_id
        n = payload.num_recs
        good_threshold = payload.good_threshold
        bad_threshold = payload.bad_threshold
        harshness = payload.harshness

        # Check if user has ratings data
        query = "SELECT date, movie_id, rating FROM user_ratings WHERE user_id=%s;"
        success, ratings_sql = self.db.run_query(
                query, params=(user_id,), fetch="all")
        ratings = pd.DataFrame(ratings_sql, columns=[
                               'date', 'movie_id', 'rating'])
        if ratings.shape[0] == 0:
            return {"data": []}

        # Get user watchlist, willnotwatchlist, watched
        query = "SELECT date, movie_id FROM user_watchlist WHERE user_id=%s;"
        success, watchlist_sql = self.db.run_query(
                query, params=(user_id,), fetch="all")
        watchlist = pd.DataFrame(watchlist_sql, columns=['date', 'movie_id'])

        query = "SELECT date, movie_id FROM user_watched WHERE user_id=%s;"
        success, watched_sql = self.db.run_query(
                query, params=(user_id,), fetch="all")
        watched = pd.DataFrame(watched_sql, columns=['date', 'movie_id'])

        query = "SELECT date, movie_id FROM user_willnotwatchlist WHERE user_id=%s;"
        success, willnotwatch_sql = self.db.run_query(
                query, params=(user_id,), fetch="all")
        notwatchlist = pd.DataFrame(
            willnotwatch_sql, columns=['date', 'movie_id'])

        # Prepare data
        good_list, bad_list, hist_list, val_list, ratings_dict = self.__prep_data(
            ratings, watched, watchlist, notwatchlist, good_threshold=good_threshold, bad_threshold=bad_threshold
        )

        # Run prediction with parameters then wrangle output
        w2v_preds = self.__predict(
            good_list, bad_list, hist_list, val_list, ratings_dict, harshness=harshness, n=n)
        df_w2v = pd.DataFrame(w2v_preds, columns=['movie_id', 'score'])

        # get movie info using movie_id
        rec_data = self.info_tool.get_info(df_w2v)
        rec_data = rec_data.fillna("None")

        def _commit_to_database(model_recs, user_id, num_recs, good, bad, harsh):
            """ Commit recommendations to the database """
            date = datetime.now()
            model_type = "ratings"

            create_rec = """
            INSERT INTO recommendations 
            (user_id, date, model_type, num_recs, good_threshold, bad_threshold, harshness) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING recommendation_id;
            """
            success, rec_id = self.db.run_query(
                create_rec, params=(user_id, date, model_type, num_recs, good, bad, harsh))

            create_movie_rec = """
            INSERT INTO recommendations_movies
            (recommendation_id, movie_number, movie_id)
            VALUES (%s, %s, %s);
            """

            for num, movie in enumerate(model_recs):
                success, _ = self.db.run_query(
                create_movie_rec, params=(rec_id, num+1, movie['movie_id']), fetch="none")

        rec_json = self.__get_JSON(rec_data)

        # add background task to commit recs to DB
        background_tasker.add_task(
            _commit_to_database,
            rec_json, user_id, n, good_threshold, bad_threshold, harshness)

        return {
            "data": rec_json
        }
    
    def get_movie_list(self, list_id: int):
        """ Gets all movies in MovieList and the associated recs """
        query = """SELECT l.movie_id, m.primary_title, m.start_year, m.genres, m.poster_url 
        FROM list_movies AS l LEFT JOIN movies AS m ON l.movie_id = m.movie_id
        WHERE l.list_id = %s;"""
        success, list_sql = self.db.run_query(
            query, params=(list_id,), fetch="all")
        if not success:
            return "Failure"
        list_json = {
            "data": [],
            "recs": []
        }
        if len(list_sql) > 0:
            movie_ids = [movie[0] for movie in list_sql]
            data_df = self.info_tool.get_info(pd.DataFrame({"movie_id": movie_ids}))
            data_df = data_df.fillna("None")
            list_json["data"] = self.__get_JSON(data_df)
            w2v_preds = self.__predict(movie_ids)
            df_w2v = pd.DataFrame(w2v_preds, columns=['movie_id', 'score'])
            rec_data = self.info_tool.get_info(df_w2v)
            rec_data = rec_data.fillna("None")
            list_json["recs"] = self.__get_JSON(rec_data)
        return list_json
    # ------- End Public Methods -------