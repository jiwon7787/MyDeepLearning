import unittest
subtraction_function = lambda x, y: x - y

class TestSubtraction():
    def test_subtraction(self):
        self.assertEqual(subtraction_function(8,3),7)


unittest.main()
