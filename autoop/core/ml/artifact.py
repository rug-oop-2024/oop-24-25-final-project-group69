from pydantic import BaseModel, Field, PrivateAttr
from typing import Dict, List
import base64

import os


class Artifact(BaseModel):
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

    type: str = Field(default="")
    name: str = Field(default="")
    data: bytes = Field(default=b"")
    version: str = Field(default="")
    asset_path: str = Field(default="")
    metadata: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    """def __init__(self,
                 type: str,
                 name: str,
                 data: bytes,
                 version: str,
                 asset_path: str,
                 metadata: Dict[str, str],
                 tags: List[str]) -> None:
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.metadata = metadata
        self.tags = tags
        self.type = type"""

    @property
    def id(self) -> str:
        """Generate an ID based on base64 encoded asset_path and version
        Returns:
            str: an id
        """
        encoded_path = base64.b64encode(
            self.asset_path.encode()).decode('utf-8')
        return f"{encoded_path}:{self.version}"

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
        self.data = data
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        # Write data to file
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)
