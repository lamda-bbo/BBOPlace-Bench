import unittest

from src.placedb import get_node_to_net_dict


class NodeToNetDictTest(unittest.TestCase):
    def setUp(self):
        self.node_info = {"macro_a": {}, "macro_b": {}, "macro_c": {}}
        self.nets = [
            ("net_z", {"nodes": {"macro_a": {}, "macro_b": {}}}),
            ("net_a", {"nodes": {"macro_a": {}, "macro_c": {}}}),
            ("net_m", {"nodes": {"macro_a": {}, "macro_b": {}, "macro_c": {}}}),
        ]

    def test_net_order_is_sorted_and_immutable(self):
        mapping = get_node_to_net_dict(self.node_info, dict(self.nets))

        self.assertEqual(mapping["macro_a"], ("net_a", "net_m", "net_z"))
        self.assertIsInstance(mapping["macro_a"], tuple)

    def test_net_order_does_not_depend_on_input_insertion_order(self):
        forward = get_node_to_net_dict(self.node_info, dict(self.nets))
        reverse = get_node_to_net_dict(self.node_info, dict(reversed(self.nets)))

        self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()
