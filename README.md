# Weather Forecasting with Residual Networks

Deep learning architecture which can forecast climatic variables like precipitation and temperature using univariate historical data of those variables. Model architecture is inspired from N-BEATS: [Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437) by Oreshkin et. al. (2019). 

## Data
Data for our experiments was obtained from [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/) maintained by the World Bank Group. 

## Usage instructions
To train the model, clone this repository locally and download the data from the above source. Make sure to name the data files `rain_wb.csv` and `temp_wb.csv`, or change the names of files to be loaded in `data_utils.py`. Then in the repo directory, run:

```
# For temperature 
python3 main.py --task 'temp' --config 'configs/temp.yaml' --root 'path/to/dir/containing/datafiles'

# For precipitation
python3 main.py --task 'rain' --config 'configs/rain.yaml' --root 'path/to/dir/containing/datafiles'
```

To load a trained model to perform any downstream tasks, use the `--load` CLI argument like so (model weight will be stored as `best_model.ckpt`):

```
python3 main.py --task 'temp' --config 'configs/temp.yaml' --root 'dir/with/data' --load 'dir/with/trained/model'
```
