from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    Manages the registration, retrieval, listing, and deletion of artifacts, handling
    interactions with the database and storage layers.

    Attributes:
        _database (Database): The database instance for metadata management.
        _storage (Storage): The storage instance for handling artifact data storage.
    """
    def __init__(self, 
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes the ArtifactRegistry with a database and storage instance.
        
        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage system to handle artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """
        Registers a new artifact by storing its data and metadata.

        Args:
            artifact (Artifact): The artifact to be registered, containing data, metadata,
                                 and asset path.
        """
        # Save the artifact data to storage
        self._storage.save(artifact.data, artifact.asset_path)
        # Save the artifact metadata to the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts in the database, optionally filtering by type.

        Args:
            type (str, optional): Filter by artifact type. If None, lists all types.

        Returns:
            List[Artifact]: A list of Artifact objects matching the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves a specific artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            Artifact: The retrieved Artifact object.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes a specific artifact by its ID from both the storage and the database.

        Args:
            artifact_id (str): The unique identifier of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class representing the automated machine learning system,
    providing access to artifact registry, storage, and database instances.

    Attributes:
        _instance (AutoMLSystem): Singleton instance of AutoMLSystem.
        _storage (LocalStorage): Local storage instance for storing assets.
        _database (Database): Database instance for managing metadata.
        _registry (ArtifactRegistry): Registry for managing artifacts.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoMLSystem with storage and database instances.

        Args:
            storage (LocalStorage): The local storage instance for storing assets.
            database (Database): The database instance for storing artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Returns the singleton instance of the AutoMLSystem, initializing it if necessary.

        Returns:
            AutoMLSystem: The singleton instance of AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Provides access to the artifact registry for managing artifacts.

        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
