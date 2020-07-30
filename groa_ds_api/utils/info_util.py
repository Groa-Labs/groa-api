import pandas as pd  # type: ignore

from groa_ds_api.utils.database_util import DatabaseUtility


class InfoUtility(object):
    """ Movie utility class that uses a W2V model to recommend movies
    based on the movies a user likes and dislikes """

    def __init__(self, db_tool: DatabaseUtility):
        """ Initialize info utility """
        self.db = db_tool
        self.id_book = self.__get_id_book()

    # ------- Start Private Methods -------
    def __get_id_book(self):
        """ Gets movie data from database to merge with recommendations """
        query = """
        SELECT movie_id, primary_title, start_year, genres, 
        poster_url, trailer_url, description, average_rating 
        FROM movies;"""
        success, movie_sql = self.db.run_query(query, fetch="all")
        
        id_book = pd.DataFrame(movie_sql, columns=[
                               'movie_id', 'title', 'year', 'genres', 'poster_url',
                               'trailer_url', 'description', 'avg_rating'])
        return id_book

    def get_info(self, recs: pd.DataFrame):
        """ Merging recs with id_book to get movie info """
        return pd.merge(recs, self.id_book, how='left', on='movie_id')