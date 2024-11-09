from typing import Dict, List
import base64

import os


class Artifact():
    """Class Artifact
    Attributes:
        "asset_path"
        "version"
        "data"
        "metadata"
        "type"
        "tags"
        "id"
    """

    _type: str
    _name: str
    _data: bytes
    _version: str
    _asset_path: str
    _metadata: Dict[str, str]
    _tags: List[str]

    def __init__(self,
                 name: str,
                 data: bytes,
                 type: str = "",
                 version: str = "",
                 asset_path: str = "",
                 metadata: Dict[str, str] = {},
                 tags: List[str] = []) -> None:
        """
        Initializes an Artifact instance with the specified attributes.

        Args:
            _type (str): The type of the artifact.
            _name (str): The name of the artifact.
            _data (bytes): The binary data associated with the artifact.
            _version (str): The version of the artifact.
            _asset_path (str): The file path where the artifact is stored.
            _metadata (Dict[str, str]): A dictionary of metadata for
            the artifact.
            _tags (List[str]): A list of tags associated with the artifact.
        """
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata
        self._tags = tags
        self._type = type

    @property
    def type(self) -> str:
        """
        Gets the type of the asset.

        Returns:
            str: The type of the asset.
        """
        return self._type

    @property
    def name(self) -> str:
        """
        Gets the name of the asset.

        Returns:
            str: The name of the asset.
        """
        return self._name

    @property
    def data(self) -> bytes:
        """
        Gets the binary data associated with the asset.

        Returns:
            bytes: The binary data of the asset.
        """
        return self._data

    @property
    def version(self) -> str:
        """
        Gets the version of the asset.

        Returns:
            str: The version of the asset.
        """
        return self._version

    @property
    def asset_path(self) -> str:
        """
        Gets the file path where the asset is stored.

        Returns:
            str: The file path of the asset.
        """
        return self._asset_path

    @property
    def metadata(self) -> Dict[str, str]:
        """
        Gets the metadata dictionary of the asset.

        Returns:
            Dict[str, str]: A dictionary containing the metadata of the asset.
        """
        return self._metadata

    @property
    def tags(self) -> List[str]:
        """
        Gets the tags associated with the asset.

        Returns:
            List[str]: A list of tags associated with the asset.
        """
        return self._tags

    @property
    def id(self) -> str:
        """Generate an ID based on base64 encoded asset_path and version
        Returns:
            str: an id
        """
        encoded_path = base64.b64encode(
            self.asset_path.encode()).decode('utf-8')
        return f"{encoded_path}_{self.version}"

    def read(self) -> bytes:
        """Method for reading artifact data
        Returns:
            bytes: read data
        """
        return self.data

    def save(self, data: bytes) -> None:
        """Method for saving artifact data
        Returns:
            bytes: saved data
        """
        self._data = data
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)
