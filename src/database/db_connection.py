import psycopg2

from psycopg2 import pool
import logging
from typing import Optional, Dict, Any
import yaml
import os

logger = logging.getLogger(__name__)
    
class DatabaseConnection:
    _instance = None
    _connection_pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Initialize the connection pool with settings from config"""
        try:
            config = self._load_db_config()
            self._connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=config['host'],
                database=config['database'],
                user=config['username'],
                password=config['password'],
                port=config['port']
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise

    def _load_db_config(self) -> Dict[str, Any]:
        """Load database configuration from settings.yaml"""
        config_path = os.path.join('config', 'settings.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['database']
        except Exception as e:
            logger.error(f"Failed to load database configuration: {str(e)}")
            raise

    def get_connection(self):
        """Get a connection from the pool"""
        try:
            if self._connection_pool is None:
                raise Exception("Database connection pool is not initialized")
            return self._connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get database connection: {str(e)}")
            raise

    def release_connection(self, conn):
        """Release a connection back to the pool"""
        try:
            if self._connection_pool is not None:
                self._connection_pool.putconn(conn)
            else:
                logger.error("Cannot release connection: connection pool is not initialized")
                raise Exception("Database connection pool is not initialized")
        except Exception as e:
            logger.error(f"Failed to release database connection: {str(e)}")
            raise

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[list]:
        """Execute a query and return results"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:  # If the query returns data
                    return cursor.fetchall()
                conn.commit()
                return None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to execute query: {str(e)}")
            raise
        finally:
            if conn:
                self.release_connection(conn)

    def close_all_connections(self):
        """Close all connections in the pool"""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("All database connections closed") 