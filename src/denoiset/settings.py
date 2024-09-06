from typing import List, Optional
from pydantic import BaseModel


class ProcessingSoftware(BaseModel):
    name: str
    version: str

    
class ProcessingInputPredict3d(BaseModel):
    model: str
    in_dir: str
    filenames: Optional[str]

    
class ProcessingOutput(BaseModel):
    out_dir: str


class ProcessingParametersPredict3d(BaseModel):
    pattern: str
    exclude_tags: List[str]
    length: int
    padding: int
    live: bool
    t_interval: float
    t_exit: float


class ProcessingConfigPredict3d(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputPredict3d
    output: ProcessingOutput
    parameters: ProcessingParametersPredict3d
