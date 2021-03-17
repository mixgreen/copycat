import unittest

import dax.clients.pygsti


def _mkl(size):
    """Make a list."""
    return [0] * size


class PygstTestCase(unittest.TestCase):
    def test_partition_circuit_list(self):
        data = [
            # Circuit list, max partition size, ref for num partitions
            ([_mkl(4), _mkl(4), _mkl(4)], 4, 3),
            ([_mkl(4), _mkl(4), _mkl(4)], 7, 3),
            ([_mkl(4), _mkl(4), _mkl(4)], 8, 2),
            ([_mkl(4), _mkl(4), _mkl(4)], 9, 2),
            ([_mkl(4), _mkl(4), _mkl(4)], 11, 2),
            ([_mkl(4), _mkl(4), _mkl(4)], 12, 1),
            ([_mkl(4), _mkl(4), _mkl(4), _mkl(4), _mkl(4)], 8, 3),
            ([_mkl(5), _mkl(5), _mkl(4), _mkl(4), _mkl(4)], 8, 4),
            ([_mkl(4), _mkl(4), _mkl(4), _mkl(5), _mkl(4)], 8, 4),
            ([_mkl(4), _mkl(4), _mkl(4), _mkl(4), _mkl(5)], 8, 3),
            ([_mkl(8), _mkl(8), _mkl(8), _mkl(8), _mkl(1)], 8, 5),
            ([_mkl(1), _mkl(1), _mkl(1), _mkl(1), _mkl(1)], 1, 5),
            ([_mkl(1), _mkl(1), _mkl(1), _mkl(1), _mkl(1)], 2, 3),
            ([_mkl(1), _mkl(1), _mkl(1), _mkl(1), _mkl(1)], 3, 2),
            ([_mkl(1), _mkl(1), _mkl(1), _mkl(1), _mkl(1)], 4, 2),
            ([_mkl(1), _mkl(1), _mkl(1), _mkl(1), _mkl(1)], 5, 1),
        ]

        for circuit_list, max_partition_size, num_partitions_ref in data:
            with self.subTest(circuit_list=circuit_list, max_partition_size=max_partition_size,
                              num_partitions_ref=num_partitions_ref):
                result = dax.clients.pygsti._partition_circuit_list(circuit_list, max_partition_size=max_partition_size)

                for partition in result:
                    # Test if partition size is not exceeded
                    self.assertLessEqual(sum(len(p) for p in partition), max_partition_size,
                                         'Partition exceeded max size')

                # Test number of partitions
                self.assertEqual(len(result), num_partitions_ref, 'Unexpected number of partitions')

    def test_partition_circuit_list_error(self):
        data = [
            # Circuit list, max partition size
            ([_mkl(4), _mkl(4), _mkl(5)], 4),
            ([_mkl(8), _mkl(4), _mkl(4)], 7),
            ([_mkl(9), _mkl(4), _mkl(4)], 8),
            ([_mkl(10), _mkl(4), _mkl(4)], 9),
            ([_mkl(4), _mkl(12), _mkl(4)], 11),
            ([_mkl(4), _mkl(4), _mkl(13)], 12),
            ([_mkl(9), _mkl(4), _mkl(4), _mkl(4), _mkl(4)], 8),
            ([_mkl(5), _mkl(9), _mkl(4), _mkl(4), _mkl(4)], 8),
            ([_mkl(4), _mkl(4), _mkl(9), _mkl(5), _mkl(4)], 8),
            ([_mkl(4), _mkl(4), _mkl(4), _mkl(9), _mkl(5)], 8),
            ([_mkl(8), _mkl(8), _mkl(8), _mkl(8), _mkl(9)], 8),
        ]

        for circuit_list, max_partition_size in data:
            with self.subTest(circuit_list=circuit_list, max_partition_size=max_partition_size):
                with self.assertRaises(ValueError, msg='Impossible partitioning did not raise'):
                    dax.clients.pygsti._partition_circuit_list(circuit_list, max_partition_size=max_partition_size)


if __name__ == '__main__':
    unittest.main()
