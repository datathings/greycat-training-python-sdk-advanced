# Python SDK advanced

This repository holds a DEMO example of the Greycat-Python binding.

## Dataset
Create a folder named `data/`. Open the terminal inside and run:
```
curl -LO https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset/resolve/main/data.csv
```

## Set the Python environment environment
Using a Python environment of the version 3.11, install the required python packages:
```
pip install -r requirements.txt
```

## Foder structure

* gc: Greycat files to define the importer and the data model.
  - `edi/importer.gcl`: function to load the data from the CSV file to the database.
  - `model/model.gcl`: establishes the data model of the database and the normalize function.
* lib: Python library used by the script.
  - `architecture.py`: definition of the model class (in this case the Transformer).
  - `engine.py`: functions required to run the training loop.
  - `processing.py`: defines the data loader for training each optimization step.
  - `utils.py`: auxiliary functions.
* `main.py`: script to be executed in order to run the experiment.
* `project.gcl`: Greycat file containing the functions to get and insert data in the database.
* `config.yaml`: parameters of the execution.
* configs(*): historical of config files ever run. Datetime in the filename.
* histories(*): historical of train/test loss function curves. Datetime in the filename.
* trained-models(*): Folder holding all the models generated as a result of training. Datetime in the filename.

(*) These folders are not existing by default, but they are created when needed in case they don't exist.

## Execution
Before running the script, you should have an open Greycat session:
```
greycat serve --user=1
```
The first time you execute this command, the dataset will be imported and preprocessed in Greycat.

All the parameters of the execution are in the `config.yaml` file.

In order to run the script:
```
python -m main
```



## TODO

- [ ] Document data/ structure (\<input file\>.csv, models directory outputs, etc)
- [x] GreyCat:
  - [x] Load data
  - [x] Prepare data
  - [x] Endpoint to serve prepared data
  - [x] Endpoint to save model details
- [x] Python
  - [x] Pull prepared
  - [x] Train
  - [x] Push back model details (model path, config path, train/test loss history)