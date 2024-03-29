# data_cleaning_with_latent_operators
Code repository for the LOP paper on data cleaning.

## Setup
We provided the REIN benchmark  data used for the paper as is, in **rein_data.zip**. Once the data is decompressed to the **DATASETS_REIN/** folder, one must create a python 3 environment and install the dependencies listed on **requirements.txt**. Next, a new LOP model can be trained on any dataset as long as its configuration is mapped to the **datasets.py** script.
## Reproducing results and Training
To reproduce the paper results on the REIN benchmark, a script was provided (**rein_benchmark.py**). For example:
```
python3 rein_benchmark.py --dataset adult  --K 12 --latent 120 --epochs 100 --batch_size 256 --eval_tuples 30000
```
## Ablation Studies
All our ablation studies are available on **ablation_studies.py**, and it is already configured to replicate only the published ones.
```
python3 ablation_studies.py --dataset soccer_PLAYER  --K 12 --latent 120 --epochs 40 --batch_size 256 --training_tuples 30000
```
## Plots
All the evaluation files are saved inside the **evaluation/** folder as .CSV and can be easily plotted. We provide a script to replicate our plots in **plotter.py**, the usage follows:
```
python3 plotter.py --experiment tuplewise  --dataset adult --y_title  --legend
```