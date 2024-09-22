# tests/test_record.py

import unittest
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_config import load_config
from utils.record import Record

class TestRecordGetMethod(unittest.TestCase):
    def setUp(self):
        self.record = Record(
            id=1,
            title="Sample Title",
            published_date="2024-09-22",
            categories=["Category1", "Category2", "Category3"],
            content="Sample content here."
        )
        config=load_config

    def test_get_existing_attribute(self):
        self.assertEqual(self.record.get('title'), "Sample Title")

    def test_get_nonexistent_attribute_with_default(self):
        self.assertEqual(self.record.get('author', default='Unknown'), 'Unknown')

    def test_get_existing_list_attribute_first_element(self):
        self.assertEqual(self.record.get('categories', 0), "Category1")

    def test_get_existing_list_attribute_last_element(self):
        self.assertEqual(self.record.get('categories', 2), "Category3")

    def test_get_list_attribute_out_of_range_with_default(self):
        self.assertEqual(self.record.get('categories', 5, default='Unknown'), 'Unknown')

    def test_get_non_list_attribute_with_index(self):
        self.assertEqual(self.record.get('title', 0, default='Invalid'), 'Invalid')

    def test_get_multiple_indices_on_list_attribute(self):
        # Assuming categories could be nested lists; in this case, it's not, so should return default
        self.assertEqual(self.record.get('categories', 0, 1, default='Invalid'), 'Invalid')

    def test_get_with_no_arguments(self):
        # Should return the attribute value without default
        self.assertEqual(self.record.get('id'), 1)

    def test_get_with_none_default(self):
        # If the attribute doesn't exist, return None
        self.assertIsNone(self.record.get('publisher'))

    def test_get_with_non_integer_index(self):
        # Non-integer indices should not work; should return default
        self.assertEqual(self.record.get('categories', 'a', default='Invalid'), 'Invalid')

    def test_get_with_negative_index(self):
        # Negative indices are not handled; should return default
        self.assertEqual(self.record.get('categories', -1, default='Invalid'), 'Invalid')

if __name__ == '__main__':
    unittest.main()