from typing import List, Optional
from pydantic import BaseModel


class ProcessingSoftware(BaseModel):
    name: str
    version: str


class ProcessingOutput(BaseModel):
    out_dir: str
    
    
class ProcessingInputPredict3d(BaseModel):
    model: str
    in_dir: str
    filenames: Optional[str]

    
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


class ProcessingInputTrain3d(BaseModel):
    in_path: str
    model: Optional[str]
    

class ProcessingParametersTrain3d(BaseModel):
    seed: Optional[int]
    optimizer: str
    learning_rate: float
    batch_size: int
    val_fraction: float
    pattern: str
    extension: str
    n_epochs: int
    n_denoise: int
    length: int
    live: bool
    t_interval: float
    t_exit: float


class ProcessingConfigTrain3d(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputTrain3d
    output: ProcessingOutput
    parameters: ProcessingParametersTrain3d
