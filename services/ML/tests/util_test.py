import unittest
import pandas

from src.analyzer.util import drop_sparse_columns, drop_high_cardinality

class TestDropHighCardinalityObj(unittest.TestCase):
  def test(self):
    frame = pandas.DataFrame(
      data = {
        "name": [ str(x) for x in range(0, 100) ],
      },
    )
    frame = drop_high_cardinality(frame, max=10)
    self.assertEqual(0, frame.shape[1])

class TestDropHighCardinalityInt(unittest.TestCase):
  def test(self):
    """
      Should test to ensure that high cardinality integers are not being dropped.
      Low cadinality ints are generally categories (enums)
      High cardinality ints are generally continuous (temp, age, etc...)
    """
    frame = pandas.DataFrame(
      data = {
        "name": [ x for x in range(0, 100) ],
      },
    )
    frame = drop_high_cardinality(frame, max=10)
    self.assertEqual(1, frame.shape[1])

class TestDropSparseColumns(unittest.TestCase):
  def test(self):
    frame = pandas.DataFrame(
      data = {
        "name": [ None for x in range(0, 100) ],
      },
    )
    frame = drop_sparse_columns(frame)
    self.assertEqual(0, frame.shape[1])