import os
import denoiset.curation as curation
from denoiset.args import CurateArgs, AttrDict
from denoiset.settings import SettingsConfigCurate


def store_parameters(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    reconfig = {}
    reconfig["software"] = {"name": "denoiset", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in ["metrics_file"]}
    reconfig["output"] = {k: d_config[k] for k in ["output"]}

    used_keys = [list(reconfig[key].keys()) for key in reconfig]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in d_config if key not in used_keys]
    reconfig["parameters"] = {k: d_config[k] for k in param_keys}

    reconfig = SettingsConfigCurate(**reconfig)

    with open(config.output.replace("csv", "json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))

        
def main():

    args = CurateArgs()
    config = args.parse_args()
    config = AttrDict(vars(config))
    store_parameters(config)

    curator = curation.TomogramCurator(config.metrics_file)
    metrics = ['tilt_axis', 'thickness', 'global_shift',
               'bad_patch_low', 'bad_patch_all', 'ctf_res', 'ctf_score'] 
    for m in metrics:
        curator.reset_criterion(m.title().replace('Ctf', 'CTF'), config[m])

    curator.curate(
        out_file = None,
        vol_path = None,
        max_selected = config.max_selected,
        sort_by = config.sort_by,
    )
    curator.visualize_curated(out_file=config.output.replace("csv", "png"))

    # order selected list based on one metric and drop irrelevant columns
    curator.sort_selected(config.max_selected, config.sort_by)
    df_sel = curator.select_dataframe()
    df_sel.to_csv(config.output, index=False)    

    
if __name__ == "__main__":
    main()
