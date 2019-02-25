import unittest
import pandas
import numpy

from io import StringIO
from src.analyzer.data import load_data

class TestLoadData(unittest.TestCase):
  def test(self):
    data = pandas.DataFrame(
      data = {
        'a': [1],
        'b': [2],
        'c': [3],
      },
    )
    filename = 'tmp.csv'
    data.to_csv('/datasets/' + filename, index=False)
    dataset = load_data(filename)
    numpy.testing.assert_array_equal(
      dataset.values,
      data.values,
    )