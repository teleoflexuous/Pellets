from pydantic import BaseModel, Field, validator, UUID4, ConfigDict, SecretStr
from typing import List, Dict, Any

# Config needs to be json serializable, human readable
class Config(BaseModel):
    db_path: str = "pellets.db"
    prompt_arguments_schema: Dict[str, Any] = {
        "model": (str, Field(..., description="The model used to generate the response")),
        "prompt": (str, Field(..., description="The prompt used to generate the response")),
    }
    response_schema: Dict[str, Any] = {
        "response": (str, Field(..., description="The response generated by the model")),
    }

config = Config()