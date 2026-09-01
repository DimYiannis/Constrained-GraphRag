"""
    Neo4h connection/session handling
"""

import os

from neo4j import Driver, GraphDatabase

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"

def get_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> Driver:
    """
        driver instance, to connect to the database

        calls .verify_connectivity() method to ensure 
        that a working connection can be established.
    """
    uri = uri or os.environ.get("NEO4J_URI", DEFAULT_URI)
    user = user or os.environ.get("NEO4J_USER", DEFAULT_USER)
    password = password or os.environ.get("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "no Neo4js password - set NEO4J_PASSWORD or pass password="
        )
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # verify connection
    driver.verify_connectivity()
    return driver

def run_query(
    driver: Driver,
    query: str,
    parameters: dict | None = None,
    database: str | None = None,
):
    """
        Run a Cypher query with the method Driver.execute_query()
    
        args:
            driver: driver instance
            query: cypher query string
            parameters: query params
            database: target db name
        
        return:
            EagerResult (records, summary, keys) -> The object
            Neo4j's Python driver gives after executing a query:
            EagerResult
            ├── records
            ├── summary
            └── keys
    """
    return driver.execute_query(
        query,
        parameters or {},
        database_=database)