import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from elasticsearch import Elasticsearch, RequestsHttpConnection  # type: ignore
from requests_aws4auth import AWS4Auth  # type: ignore

from groa_ds_api.models import *
from groa_ds_api.utils.database_util import DatabaseUtility
from groa_ds_api.utils.info_util import InfoUtility


class DataUtility(object):
    """ Movie utility class that uses a W2V model to recommend movies
    based on the movies a user likes and dislikes """

    def __init__(self, db_tool: DatabaseUtility, info_tool: InfoUtility):
        """ Initialize model with name of .model file """
        self.db = DatabaseUtility()
        self.info_tool = info_tool
        self.es = self.__get_es()

    # ------- Start Private Methods -------
    def __get_es(self):
        return Elasticsearch(
            hosts=[{'host': os.getenv('ELASTIC'), 'port': 443}],
            http_auth=AWS4Auth(os.getenv('ACCESS_ID'), os.getenv(
                'ACCESS_SECRET'), 'us-east-1', 'es'),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

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
    def add_rating(self, payload: RatingInput):
        query = "SELECT COUNT(*) FROM user_ratings WHERE user_id = %s AND movie_id = %s;"
        params = (payload.user_id, payload.movie_id)
        success, rating = self.db.run_query(query, params=params, fetch="one")
        if not success:
            return "Failure"
        if rating > 0:
            query = """UPDATE user_ratings SET rating = %s, date = %s
            WHERE user_id = %s AND movie_id = %s"""
            params = (payload.rating, datetime.now(),
                      payload.user_id, payload.movie_id)
        else:
            query = """
            INSERT INTO user_ratings
            (user_id, movie_id, rating, date, source)
            VALUES (%s, %s, %s, %s, %s)
            """
            params = (payload.user_id, payload.movie_id,
                      payload.rating, datetime.now(), "groa")
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def remove_rating(self, user_id: str, movie_id: str):
        """
        Removes a single rating from the user_ratings table.
        """
        query = "DELETE FROM user_ratings WHERE user_id = %s AND movie_id = %s;"
        params = (user_id, movie_id)
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def add_to_watchlist(self, payload: UserAndMovieInput):
        query = "SELECT COUNT(*) FROM user_watchlist WHERE user_id = %s AND movie_id = %s;"
        params = (payload.user_id, payload.movie_id)
        success, watchlist = self.db.run_query(
            query, params=params, fetch="one")
        if not success:
            return "Failure"
        if watchlist == 0:
            query = """
            INSERT INTO user_watchlist
            (user_id, movie_id, date, source)
            VALUES (%s, %s, %s, %s);
            """
            params = (payload.user_id, payload.movie_id,
                      datetime.now(), "groa")
            success, _ = self.db.run_query(
                query, params=params, commit=True, fetch="none")
            if not success:
                return "Failure"
        return "Success"

    def remove_watchlist(self, user_id: str, movie_id: str):
        query = "DELETE FROM user_watchlist WHERE user_id = %s AND movie_id = %s;"
        params = (user_id, movie_id)
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def add_to_notwatchlist(self, payload: UserAndMovieInput):
        query = "SELECT COUNT(*) FROM user_willnotwatchlist WHERE user_id = %s AND movie_id = %s;"
        params = (payload.user_id, payload.movie_id)
        success, notwatchlist = self.db.run_query(
            query, params=params, fetch="one")
        if not success:
            return "Failure"
        if notwatchlist == 0:
            query = """
            INSERT INTO user_willnotwatchlist
            (user_id, movie_id, date)
            VALUES (%s, %s, %s);
            """
            params = (payload.user_id, payload.movie_id, datetime.now())
            success, _ = self.db.run_query(
                query, params=params, commit=True, fetch="none")
            if not success:
                return "Failure"
        return "Success"

    def remove_notwatchlist(self, user_id: str, movie_id: str):
        query = "DELETE FROM user_willnotwatchlist WHERE user_id = %s AND movie_id = %s;"
        success, _ = self.db.run_query(
            query,
            params=(user_id, movie_id),
            commit=True,
            fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def search_movies(self, query: str):
        result = self.es.search(index="groa", size=20, expand_wildcards="all", body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["description", "primary_title", "original_title", "genres"]
                }
            }
        })
        movie_ids = []
        for elem in result['hits']['hits']:
            movie_ids.append(elem['_id'])
        search_df = self.info_tool.get_info(pd.DataFrame({"movie_id": movie_ids}))
        search_df = search_df.fillna("None")
        search_json = self.__get_JSON(search_df)
        return {
            "data": search_json
        }

    def add_interaction(self, user_id: str, movie_id: str):
        query = """
        UPDATE recommendations_movies
        SET interaction = TRUE
        FROM recommendations
        WHERE recommendations.user_id = %s 
        AND recommendations_movies.movie_id = %s;
        """
        params = (user_id, movie_id)
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def create_movie_list(self, payload: CreateListInput):
        """ Creates a MovieList """
        query = """INSERT INTO movie_lists
        (user_id, name, private) VALUES (%s, %s, %s) RETURNING list_id;"""
        params = (payload.user_id, payload.name, payload.private)
        success, list_id = self.db.run_query(query, params=params, commit=True)
        if not success:
            return "Failure"
        return {
            "list_id": list_id,
            "name": payload.name,
            "private": payload.private
        }

    def get_user_lists(self, user_id: str):
        """ Get user's MovieLists """
        query = "SELECT list_id, name, private FROM movie_lists WHERE user_id = %s;"
        success, user_lists = self.db.run_query(
            query, params=(user_id,), fetch="all")
        if not success:
            return "Failure"
        user_lists_json = [self.__get_list_preview(
            elem) for elem in user_lists]
        return user_lists_json

    def get_all_lists(self):
        """ Get all MovieLists """
        query = "SELECT list_id, name, private FROM movie_lists WHERE private=FALSE;"
        success, lists = self.db.run_query(query, fetch="all")
        if not success:
            return "Failure"
        lists_json = [self.__get_list_preview(elem) for elem in lists]
        return lists_json

    def add_to_movie_list(self, list_id: int, movie_id: str):
        """ Add movie to a MovieList """
        query = """INSERT INTO list_movies
        (list_id, movie_id) VALUES (%s, %s);"""
        params = (list_id, movie_id)
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def remove_from_movie_list(self, list_id: int, movie_id: str):
        """ Remove movie from a MovieList """
        query = "DELETE FROM list_movies WHERE list_id = %s AND movie_id = %s;"
        params = (list_id, movie_id)
        success, _ = self.db.run_query(
            query, params=params, commit=True, fetch="none")
        if not success:
            return "Failure"
        return "Success"

    def delete_movie_list(self, list_id: int):
        """ Delete a MovieList """
        query = "DELETE FROM movie_lists WHERE list_id = %s RETURNING user_id, private;"
        success, result = self.db.run_query(
            query, params=(list_id,), commit=True, fetch="all")
        if not success:
            return "Failure"
        return result[0]

    def get_service_providers(self, movie_id: str):
        """ Get the service providers of a given movie_id """
        query = """
        SELECT m.provider_id, p.name, p.logo_url, m.provider_movie_url, 
        m.presentation_type, m.monetization_type
        FROM movie_providers AS m
        LEFT JOIN providers AS p ON m.provider_id = p.provider_id
        WHERE m.movie_id = %s; 
        """
        success, prov_sql = self.db.run_query(query, params=(movie_id,), fetch="all")
        prov_json = {
            "data": []
        }
        for provider in prov_sql:
            prov_json["data"].append({
                "provider_id": provider[0],
                "name": provider[1],
                "logo": str(provider[2]),
                "link": provider[3],
                "presentation_type": provider[4],
                "monetization_type": provider[5]
            })
        return prov_json

    def get_recent_recommendations(self):
        """ Grabs the movies of recent recommendations from our API """
        query = """
        SELECT m.movie_id
        FROM recommendations_movies AS m LEFT JOIN recommendations AS r
        ON m.recommendation_id = r.recommendation_id
        ORDER BY r.date DESC
        LIMIT 50;
        """
        success, recs = self.db.run_query(query, fetch="all")
        if not success:
            return "Failure"
        recs_df = pd.DataFrame(recs, columns=["movie_id"])
        rec_data = self.info_tool.get_info(recs_df)
        rec_data = rec_data.fillna("None")
        rec_json = self.__get_JSON(rec_data)
        return {
            "data": rec_json
        }
    # ------- End Public Methods -------