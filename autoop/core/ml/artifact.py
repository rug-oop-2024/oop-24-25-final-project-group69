from pydantic import BaseModel, Field
from typing import Dict, List
import base64

class Artifact(BaseModel):
    """Class Artifact
    Attributes:
        "asset_path": "users/mo-assaf/models/yolov8.pth",
        "version": "1.0.2", 
        "data": b"binary_state_data",
        "metadata": {
            "experiment_id": "exp-123fbdiashdb",
            "run_id": "run-12378yufdh89afd",
        },
        "type": "model:torch",
        "tags": ["computer_vision", "object_detection"],
        "id": "{base64(asset_path)}:{version}"
    """
    
    type: str = Field(default="model:torch")
    name: str = Field(default="")
    data: bytes = Field(default=b"")
    version: str = Field(default="1.0.0")
    asset_path: str = Field(default="")
    metadata: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @property
    def id(self) -> str:
        """Generate an ID based on base64 encoded asset_path and version
        Returns:
            str: an id
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode('utf-8')
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """Method for reading artifact data.
        Returns:
            bytes: read data
        """
        return self.data

    def save(self, data:bytes) -> bytes:
        """Method for saving artifact data.
        Returns:
            bytes: saved data
        """
        self.data = data
        return self.data
