import os
from typing import Any, List, Optional, Tuple

import psycopg2  # type: ignore


class DatabaseUtility(object):
    """ Database Utility to execute queries on PostgreSQL DB """

    def __init__(self):
        """ Initialize connection to DB """
        self.connection = self.__get_connection()

    def __get_connection(self):
        return psycopg2.connect(
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('HOST'),
            port=os.getenv('PORT')
        )

    def __get_cursor(self):
        """ Grabs cursor from self.connection """
        try:
            cursor = self.connection.cursor()
            return cursor
        except:
            self.connection = self.__get_connection()
            return self.connection.cursor()

    def run_query(self, query: str,
                    params: Tuple[Any, ...] = (),
                    commit: bool = False,
                    fetch: str = "one"):
        try:
            cursor_dog = self.__get_cursor()
            if len(params) == 0:
                cursor_dog.execute(query)
            else:
                cursor_dog.execute(query, params)
            result = None
            if fetch == "one":
                result = cursor_dog.fetchone()[0]
            elif fetch == "all":
                result = cursor_dog.fetchall()
            if commit:
                self.connection.commit()
            cursor_dog.close()
            return True, result
        except:
            # refresh connection to avoid FailedTransaction
            self.connection = self.__get_connection()
            return False, None