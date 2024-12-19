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
    vol_path: Optional[str]
    

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
    tilt_axis: float
    thickness: float
    global_shift: float
    bad_patch_low: float
    bad_patch_all: float
    ctf_res: float
    ctf_score: float
    min_selected: int
    max_selected: int
    sort_by: str


class ProcessingConfigTrain3d(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputTrain3d
    output: ProcessingOutput
    parameters: ProcessingParametersTrain3d


class AttrDict(dict):
    """
    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Adapted from: https://stackoverflow.com/a/48806603
    """

    def __init__(self, mapping=None, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)
