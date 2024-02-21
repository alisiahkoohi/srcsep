<h1 align="center">Unearthing InSights into Mars</h1>

Code to partially reproduce results in "Unearthing InSights into Mars: unsupervised source separation with limited data" available [here](https://proceedings.mlr.press/v202/siahkoohi23a.html).


## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/alisiahkoohi/insight_src_sep
cd insight_src_sep/
conda env create -f environment.yml
conda activate srcsep
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate srcsep`, the
following times.

## Scripts

Deglitching can be done for a toy example by running the following:

```bash
python scripts/toy_example.py
```

The default command line arguments are stored at `configs/toy_example.json`. Non-default arguments can be passed to the script by for example:

```bash
python scripts/toy_example.py
    --max_itr 1000 \
    --j 8,8 \
    --q 1,1 \
    --type exp_glitch
```

The generated data is stored in `data/checkpoints/` directory. To visualize the results, run:

```bash
python scripts/visualize_results.py
    --max_itr 1000 \
    --j 8,8 \
    --q 1,1 \
    --type exp_glitch
```

The figures will be stored in the `plots/` directory.

**Note regarding caching:** The scattering covariance computation caches the results in `srcsep/_cached_dir` and following runs with the same exact setup will simply load the results. Feel free to delete the cache when needed.

## Questions

Please contact alisk@rice.edu for questions.

## Author

Rudy Morel and Ali Siahkoohi


