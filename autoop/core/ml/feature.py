
from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    Represents a feature within a dataset, including its name and data type.

    Attributes:
        name (str): The name of the feature, default is an empty string.
        type (Literal['numerical', 'categorical']): Specifies the type of
        the feature, either 'numerical' or 'categorical'. Defaults to
        'numerical'.
    """
    name: str = Field(default="")
    type: Literal['numerical', 'categorical'] = Field(default='numerical')

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        This method returns the name of the feature, which is typically used
        for displaying or printing the feature in a readable format.

        Returns:
            str: The name of the feature.
        """
        return f"{self.name}"
