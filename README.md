I have a directory called `dune` in which I have `ndlar_flow` <https://github.com/DUNE/ndlar_flow> and this repo:
```
cd dune
git clone https://github.com/jvmead/light_file_processing.git
cd light_file_processing
mkdir outputs
```

I prefer running in `screen` <https://hsf-training.github.io/analysis-essentials/shell-extras/screen.html> for safety of mind when running remotely.
I would strongly recommend using `remote-ssh` - config can be tricky for NERSC so see bottom of cheat sheet <https://surfdrive.surf.nl/index.php/s/jDkQV9ipn9l1ts5>.

Inside the screen (or not) using the ndlar_flow virtual environment in the adjacent directory for convenience:
```
source /global/homes/<u/user>/dune/ndlar_flow/ndlar_flow.venv/bin/activate
```

Running the script:
```
python to_dataframes.py --stage truth sum_tpc flashes match 
                        --nfiles 100
                        --indir /global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.4_1E19_RHC/MiniRun6.4_1E19_RHC.flow/FLOW/0000000/
                        --outdir outputs/
```
Different `stage`s can be run separately and then will be read in if they already exist when running the `match` stage to optionally save time.
If you wish to override this default, use the `--overwrite` argument.

One potential improvement might be to have `--outdir` generate a directory from the path if it doesn't exist and then handle the book-keeping implicitly.
Another might be to incorporate the arguments for thresholds already included (and truth match time window tolerance not yet included) into the output file/dir naming convention.
Alternatively, a timestamped directory could be generated upon running and a copy of the input arguments saved to a `config.json` in said directory.
