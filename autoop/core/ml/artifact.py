from pydantic import BaseModel, Field
from typing import Dict, List
import base64

class Artifact(BaseModel):
    """
    {
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
    }
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
        """Generate an ID based on base64 encoded asset_path and version."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode('utf-8')
        return f"{encoded_path}:{self.version}"
    
    def read(self) -> bytes:
        """Placeholder for reading artifact data."""
        return self.data
    
    def save(self, data:bytes) -> bytes:
        """Placeholder for saving artifact data."""
        self.data = data
        return self.data
