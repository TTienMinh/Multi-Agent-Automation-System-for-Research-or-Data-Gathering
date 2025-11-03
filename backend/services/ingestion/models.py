import psycopg2
import os
import dotenv

dotenv.load_dotenv()

class RawItems:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(
                dbname=os.environ.get("DB_NAME"),
                user=os.environ.get("DB_USER"),
                password=os.environ.get("DB_PASSWORD"),
                host=os.environ.get("DB_HOST"),
                port=os.environ.get("DB_PORT")
            )
            print("Database connection established.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.connection = None

    def close(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
            self.connection = None

    def execute_query(self, query: str, params=None):
        """
        Execute a SQL query and return results.
        """
        if not self.connection:
            print("No database connection.")
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return []

    def execute_non_query(self, query: str, params=None):
        """
        Execute a SQL command that does not return results (e.g., CREATE, INSERT).
        """
        if not self.connection:
            print("No database connection.")
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            cursor.close()
            return True
        except psycopg2.Error as e:
            print(f"Error executing non-query: {e}")
            return False


if __name__ == "__main__":
    db = RawItems()