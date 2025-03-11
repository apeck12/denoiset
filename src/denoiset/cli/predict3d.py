import os
import time
import denoiset.inference as inference
from denoiset.args import PredictArgs, AttrDict
from denoiset.settings import SettingsConfigPredict3d


def store_parameters(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    reconfig = {}
    reconfig["software"] = {"name": "denoiset", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in ["input", "model"]}
    reconfig["output"] = {k: d_config[k] for k in ["output"]}

    used_keys = [list(reconfig[key].keys()) for key in reconfig]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in d_config if key not in used_keys]
    reconfig["parameters"] = {k: d_config[k] for k in param_keys}

    reconfig = SettingsConfigPredict3d(**reconfig)

    os.makedirs(config.output, exist_ok=True)
    with open(os.path.join(config.output, "predict3d.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():

    args = PredictArgs()
    config = args.parse_args()
    config = AttrDict(vars(config))
    os.makedirs(config.output, exist_ok=True)
    store_parameters(config)

    if not config.live:
        config.t_interval = config.t_exit = 0
    
    n2n = inference.Denoiser3d(
        config.model,
        config.output,
        config.inf_length,
        config.inf_padding,
    )
    n2n.process(
        config.input,
        pattern=config.pattern,
        exclude_tags=config.exclude_tags,
        t_interval=config.t_interval,
        t_exit=config.t_exit,
    )
    

if __name__ == "__main__":
    main()
