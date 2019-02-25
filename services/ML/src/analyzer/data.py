import pandas
import os

def load_data(name, index=None):
  return pandas.read_csv(
    os.path.join("/datasets/", name),
    index_col = index,
  )