from typing import List, Optional
from pydantic import BaseModel


class SettingsSoftware(BaseModel):
    name: str
    version: str


class SettingsOutput(BaseModel):
    out_dir: str
    
    
class SettingsInputPredict3d(BaseModel):
    model: str
    in_dir: str
    filenames: Optional[str]

    
class SettingsParametersPredict3d(BaseModel):
    pattern: str
    exclude_tags: List[str]
    length: int
    padding: int
    live: bool
    t_interval: float
    t_exit: float


class SettingsConfigPredict3d(BaseModel):
    software: SettingsSoftware
    input: SettingsInputPredict3d
    output: SettingsOutput
    parameters: SettingsParametersPredict3d


class SettingsInputTrain3d(BaseModel):
    in_path: str
    model: Optional[str]
    vol_path: Optional[str]
    

class SettingsParametersTrain3d(BaseModel):
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


class SettingsConfigTrain3d(BaseModel):
    software: SettingsSoftware
    input: SettingsInputTrain3d
    output: SettingsOutput
    parameters: SettingsParametersTrain3d


class SettingsInputPredict3d(BaseModel):
    input: str
    model: Optional[str]

    
class SettingsOutputPredict3d(BaseModel):
    output: str
    

class SettingsParametersPredict3d(BaseModel):
    pattern: str
    live: bool
    exclude_tags: list
    inf_length: int
    inf_padding: int
    t_interval: float
    t_exit: float


class SettingsConfigPredict3d(BaseModel):
    software: SettingsSoftware
    input: SettingsInputPredict3d
    output: SettingsOutputPredict3d
    parameters: SettingsParametersPredict3d

    
class SettingsInputDenoise3d(BaseModel):
    input: str
    metrics_file: Optional[str]
    model: Optional[str]

    
class SettingsOutputDenoise3d(BaseModel):
    output: str

    
class SettingsParametersDenoise3d(BaseModel):
    pattern: str
    odd_pattern: str
    odd_extension: str
    n_extract: int
    length: int
    seed: Optional[int]
    optimizer: str
    learning_rate: float
    batch_size: int
    val_fraction: float
    n_epochs: int
    n_denoise: int
    train_only: bool
    train_all_epochs: bool
    ch_threshold: float
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
    exclude_tags: list
    inf_length: int
    inf_padding: int

    
class SettingsConfigDenoise3d(BaseModel):
    software: SettingsSoftware
    input: SettingsInputDenoise3d
    output: SettingsOutputDenoise3d
    parameters: SettingsParametersDenoise3d


class SettingsInputCurate(BaseModel):
    metrics_file: str

    
class SettingsOutputCurate(BaseModel):
    output: str


class SettingsParametersCurate(BaseModel):
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
    

class SettingsConfigCurate(BaseModel):
    software: SettingsSoftware
    input: SettingsInputCurate
    output: SettingsOutputCurate
    parameters: SettingsParametersCurate
    
    
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
