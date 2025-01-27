import os
import time
import denoiset.curation as curation
import denoiset.training as training
import denoiset.inference as inference
from denoiset.args import DenoiseArgs, AttrDict

def main():

    args = DenoiseArgs()
    config = args.parse_args()
    config = AttrDict(vars(config))
    os.makedirs(config.output, exist_ok=True)

    if not config.train_only:
        train_out = os.path.join(config.output, "training")
        os.makedirs(train_out, exist_ok=True)
    else:
        train_out = config.output

    if not config.live:
        config.t_interval = config.t_exit = 0

    # optionally curate based on tilt-series metrics
    if config.metrics_file:
        curator = curation.TomogramCurator(config.metrics_file)
        metrics = ['tilt_axis', 'thickness', 'global_shift',
                   'bad_patch_low', 'bad_patch_all', 'ctf_res', 'ctf_score']        
        for m in metrics:
            curator.reset_criterion(m.title().replace('Ctf', 'CTF'), config[m])

        train_list = os.path.join(train_out, "traininglist.txt")
        curator.curate_live(
            out_file=train_list,
            vol_path=config.input,
            max_selected=config.max_selected,
            min_selected=config.min_selected,
            sort_by=config.sort_by,
            t_interval=config.t_interval,
            t_exit=config.t_exit,
        )
        time.sleep(config.t_interval)
        train_in = train_list
        #config.__setitem__("input", train_list) # this needs fixing
    else:
        train_in = config.input

    # train model
    n2n_train = training.Trainer3d(
        train_in,
        train_out,
        fn_model=config.model,
        seed=config.seed,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        pattern=config.odd_pattern,
        extension=config.odd_extension,
        length=config.length,
        n_extract=config.n_extract,
    )
    n2n_train.train(
        n_epochs=config.n_epochs,
        n_denoise=config.n_denoise,
        ch_threshold=config.ch_threshold,
        train_all_epochs=config.train_all_epochs,
    )

    if not config.train_only:
        n2n_infer = inference.Denoiser3d(
            n2n_train.opt_model,
            config.output,
            config.inf_length,
            config.inf_padding,
        )
        n2n_infer.process(
            config.input,
            pattern=config.pattern,
            exclude_tags=config.exclude_tags,
            t_interval=config.t_interval,
            t_exit=config.t_exit,
        )

if __name__ == "__main__":
    main()
