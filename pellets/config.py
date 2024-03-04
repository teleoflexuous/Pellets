from pydantic import BaseModel, Field, validator, UUID4, ConfigDict, SecretStr, Json
from typing import Dict, Any, List

# Config needs to be json serializable, human readable
class Config(BaseModel):
    db_path: str = "pellets.db"
    prompt_parameters: List[str] = ["format", "options", "system", "template", "context"]

config = Config()