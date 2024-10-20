# German-to-English Transformer Model with PyTorch

This project trains a transformer-based German-to-English translation model using PyTorch primitives. The code resides in the `scripts/modeling` folder, where `model.py` defines the model classes, `train.py` handles training, and `evaluate.py` evaluates the model on a validation set. The current evaluation metric used is the ROUGE score.

## Running the Code
To experiment with the code, install the required dependencies first. Navigate to the project directory and install dependencies by running:

```
cd german-to-english-transformer
python3 -m pip install -r requirements.txt
```

You can then run the code in the terminal using the code below.

```
python3 scripts/modeling/train.py  # to train
python3 scripts/modeling/evaluate.py  # to evaluate
```

To run the code in a Colab terminal (e.g., to leverage a Colab GPU), first add the project directory to the Python path, as shown below:

```
export PYTHONPATH="$/env/python:/content/german-to-english-transformer"
```

You can verify the path by running `echo $PYTHONPATH`. Note that `/env/python` represents the existing path before adding `/content/german-to-english-transformer`, so replace `/env/python` with your current path if it's different.

Once the path is set, proceed as described above. 

Note that `train.py` will save the trained model inside the `models` folder, and `evaluate.py` will save ROUGE scores and a sample of translation results in the `results` folder.

You can also run the code within `transformers.ipynb`, but this notebook is not very clean or well documented. It was mostly for experimentation.

## Configuration
The data paths and hyperparameters are configured in `scripts/modeling/config.yaml` and can be modified as necessary. After running `train.py`, the `model_name` in the config file will point to the latest trained model.

## Data
You can use any dataset in `.jsonl` format. Name your training data `train_data.jsonl` and your validation data `valid_data.jsonl`. Example datasets are available at [training dataset](https://disk.yandex.com/d/2V3YpeogygoBTA) and [validation dataset](https://disk.yandex.com/d/Q6Bm9NoG1VWcgA). Place the data files in the `data` folder.

---
