import hashlib
import json
from cachetools import TTLCache

class SearchCache: # not used in current implementation
    """
    A class to handle caching of search results using TTL (Time-To-Live) cache.

    Attributes:
        cache (TTLCache): The TTL cache instance.
    """

    def __init__(self, maxsize=100000, ttl=3600):
        """
        Initializes the SearchCache with a specified maximum size and TTL.
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def generate_key(self, talent: dict, job: dict) -> str:
        """
        Generates a unique key based on the talent and job dictionaries.

        Args:
            talent (dict): The talent dictionary.
            job (dict): The job dictionary.

        Returns:
            str: A unique key generated using MD5 hash.
        """
        key_dict = {
            'talent': talent,
            'job': job
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str):
        """
        Retrieves a value from the cache based on the provided key.
        """
        return self.cache.get(key)

    def set(self, key: str, value):
        """
        Sets a value in the cache with the provided key.
        """
        self.cache[key] = value

    def clear(self):
        """
        Clears all entries in the cache.
        """
        self.cache.clear()