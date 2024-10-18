# German-to-English Transformer Model

Personal project to train a German-to-English translation model using transformers. 

To play with the code yourself, install the requirements first. To do so, `cd` into `german-to-english-transformer` and run

```
python3 -m pip install -r requirements.txt
```

You can then either use the notebook `transfomers.ipynb` or run the code in the terminal. To run in terminal, use the code below.

```
python3 -m scripts/modeling/train.py # to train
scripts/modeling/evaluate.py # to evaluate
```

To run in a colab terminal (for instance if you want to utilize a colab GPU), add the project directory to the Python first, like below.

```
export PYTHONPATH="$/env/python:/content/german-to-english-transformer"
```

You can run `echo $PYTHONPATH` to check that the directory has been added. Note that `/env/python` was the existing path before adding `content/german-to-english-transformer`. So you should replace `/env/python` with whatever yours was.

Once you've added the path, you can proceed as normal.