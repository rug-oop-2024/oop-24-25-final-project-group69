from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob

class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes NotFoundError with the specified path.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")

class Storage(ABC):
    """
    Abstract base class defining the interface for storage backends.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            List[str]: List of paths.
        """
        pass

class LocalStorage(Storage):
    """
    Local storage implementation for saving and loading files on disk.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a specified base path.

        Args:
            base_path (str): The base path for storing assets (default: "./assets").
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a file using a sanitized key as the filename.

        Args:
            data (bytes): Data to save.
            key (str): Key used to determine the file path.

        Raises:
            NotFoundError: If the path does not exist.
        """
        sanitized_key = key.replace(":", "_")
        path = self._join_path(sanitized_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a file using the specified key.

        Args:
            key (str): Key used to determine the file path.

        Returns:
            bytes: The loaded data.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str) -> None:
        """
        Delete a file at the specified key.

        Args:
            key (str): Key used to determine the file path.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files under a specified prefix.

        Args:
            prefix (str): Prefix path to list files (default: "/").

        Returns:
            List[str]: List of relative paths to files.

        Raises:
            NotFoundError: If the prefix path does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys 
                if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a path exists; raises NotFoundError if not.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a given path in an OS-agnostic way.

        Args:
            path (str): The path to join with the base path.

        Returns:
            str: The joined path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
