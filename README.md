# Python SDK advanced

This advanced training will illustrate how one can prepare and store data in GreyCat while leveraging Python dedicated libraries to train machine learning models.

## Setup

### Prerequisites

- Python >= 3.11
- pip

### Dataset

In this training we use the following dataset:

> Patrick Fleith. (2023). Controlled Anomalies Time Series (CATS) Dataset (Version 2) [Data set]. Solenix Engineering GmbH. https://doi.org/10.5281/zenodo.8338435

```bash
mkdir -p data
curl -L https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset/resolve/main/data.csv > data/dataset.csv
```

### Install Python requirements

```bash
python3 -m pip install -r requirements.txt
```

## Folder structure

```
.
├── configs               # History of config files ever run. Datetime in the filename.
├── config.yaml           # Parameters of the execution.
├── data
│   └── dataset.csv
├── gc                    # GreyCat source files implementing importer and data model.
│   ├── edi
│   │   └── importer.gcl  # Function to load the data from the CSV file to the database.
│   └── model
│       └── model.gcl     # Implementation of the data model of the database, which comes with a normalize method.
├── histories             # History of train/test loss function curves. Datetime in the filename.
├── lib                   # Python library used by the script.
│   ├── architecture.py   # Definition of the model class (in this case the Transformer).
│   ├── engine.py         # Functions required to run the training loop.
│   ├── processing.py     # Implementation of the data loader for training each optimization step.
│   └── utils.py          # Auxiliary functions.
├── LICENSE.txt
├── main.py               # Script to be executed in order to run the experiment.
├── project.gcl           # GreyCat file containing the functions to get and insert data in the database.
├── README.md
├── requirements.txt
└── trained-models        # Folder holding all the models generated as a result of training. Datetime in the filename.
```

`config`, `histories` and `trained-models` folders are not existing by default, but they are created when needed in case they don't exist.

### Run

- Start the GreyCat server:
  ```bash
  greycat serve --user=1
  ```
  The first time you execute this command, the dataset will be imported and preprocessed in GreyCat.

- All the parameters of the execution are in the `config.yaml` file; in order to run the script:
  ```bash
  python -m main
  ```

### Config parameters
```
dataset_id: (int)           # id of the dataset in the Greycat database.

dataloader:
  n_features: (int)         # number of features to use when training (must be multiple of TransformerAutoencoder.n_heads).
  n_rows: (int)             # number of rows to load for training.
  batch_size: (int)         # number of sequences to be forward and back-propagated in parallel in each training iteration.
  window_len: (int)         # number of time steps of each sequence.
  delay: (int)              # difference in time steps between the input and target output sequence.

TransformerAutoencoder:
  n_heads: (int)            # number of heads of the attention layers (must be divisor of dataloader.n_features).
  n_encoder_layers: (int)   # number of layers (Attention + Feed Forward) of the encoding part.
  n_decoder_layers: (int)   # number of layers (Attention + Feed Forward) of the decoding part.

model_selection:
  model_lib: (string)       # name of the python module where the model architecture is defined.
  model_name: (string)      # name of the model architecture class.

training:
  device: (string)          # choose between cpu or gpu.
  learning_rate: (float)    # learning rate of the optimizer.
  epochs: (int)             # number of times all the dataset will be sweeped.
  val_size: (float)         # between 0 and 1, the proportion of the dataset reserved for validate the results.
  save_model: (bool)        # whether you want to save results and trained model, and push the information back to GreyCat.
```
