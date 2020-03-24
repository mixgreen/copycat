import unittest
import numpy as np

from dax.experiment import *
from dax.modules.indv_beam_mems import *

from dax.test.helpers.artiq import get_manager_or_parent


class ModuleWrapper(IndvBeamMemsModule):
    """Wrap class to initialize the module without devices"""

    def build(self, *args, **kwargs):
        try:
            # Build the module as normal, catch KeyError if a device can not be found
            # Necessary initialization was already done and we do not care about the devices
            super(ModuleWrapper, self).build(*args, **kwargs)
        except KeyError:
            # We stop building when we experience key errors
            pass


class TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, num_beams) -> None:
        super(TestSystem, self).build()

        # Only number of beams is relevant, rest are dummy values
        self.m = ModuleWrapper(self, 'module', num_beams=num_beams,
                               num_dpass_signals=1, dpass_aom='na', indv_aom='na', pid_sw='na')


class GetBeam2TestCase(unittest.TestCase):
    """
    This test class should cover all potential cases and outcomes of the _get_beam() function.
    """

    # Number of beams
    N = 2

    def setUp(self) -> None:
        assert self.N > 0, 'Number of beams N should be > 0'
        self.s = TestSystem(get_manager_or_parent(), num_beams=self.N)

    def test_num_beams_bounds(self):
        # Keep track of returned indices
        beam_index = set()

        # Allocate all beams
        for t in range(self.N):
            b = self.s.m._get_beam(t, True)
            self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
            self.assertNotEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Returned NO_BEAM while action was required')
            self.assertNotIn(b, beam_index, 'A beam index was returned twice')
            beam_index.add(b)  # Add beam index to keep track

        # Allocate with same targets, does not violate bounds
        for t in range(self.N):
            b = self.s.m._get_beam(t, True)
            self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
            self.assertEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Did not return NO_BEAM while no action was required')

        # Violate bounds by allocating one more beam
        with self.assertRaises(RuntimeError, msg='Number of beams bound violation did not raise'):
            self.s.m._get_beam(self.N, True)

    def test_free_beam(self):
        for _ in range(self.N * 2):
            # Keep track of returned indices
            beam_dict = dict()

            # Allocate all beams
            for t in range(self.N):
                b = self.s.m._get_beam(t, True)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotIn(b, beam_dict, 'A beam index was returned twice')
                beam_dict[b] = t  # Add beam index with target to keep track

            # Free all beams
            for t in range(self.N):
                b = self.s.m._get_beam(t, False)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Returned NO_BEAM while action was required')
                self.assertIn(b, beam_dict, 'Returned a non-assigned beam index')
                # Remove beam index to keep track and verify if beam index corresponds to assigned one
                self.assertEqual(t, beam_dict.pop(b), 'Returned beam index does not match earlier assigned index')

    def test_free_unassigned_beam(self):
        for t in range(self.N * 4):  # Test values in and outside the range of N
            # Free a beam that was never assigned
            b = self.s.m._get_beam(t, False)
            self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
            self.assertEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Did not return NO_BEAM while no action was required')

    def test_reassigned_beam(self):
        for i in range(self.N):
            with self.subTest(i=i):
                # Keep track of returned indices
                beam_dict = dict()

                # Allocate all beams
                for t in range(self.N):
                    b = self.s.m._get_beam(t, True)
                    self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                    self.assertNotIn(b, beam_dict, 'A beam index was returned twice')
                    beam_dict[b] = t  # Add beam index with target to keep track

                # Free one beam
                b = self.s.m._get_beam(i, False)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Returned NO_BEAM while action was required')
                self.assertIn(b, beam_dict, 'Returned a non-assigned beam index')
                # Remove beam index to keep track and verify if beam index corresponds to assigned one
                self.assertEqual(i, beam_dict.pop(b), 'Returned beam index does not match earlier assigned index')
                old_b = b

                # Reassign earlier freed beam
                b = self.s.m._get_beam(i, True)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotIn(b, beam_dict, 'A beam index was returned twice')
                self.assertEqual(old_b, b, 'Target was not reassigned to earlier beam')
                beam_dict[b] = i  # Add beam index with target to keep track

                # Free all beams
                for t in range(self.N):
                    b = self.s.m._get_beam(t, False)
                    self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                    self.assertNotEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Returned NO_BEAM while action was required')
                    self.assertIn(b, beam_dict, 'Returned a non-assigned beam index')
                    # Remove beam index to keep track and verify if beam index corresponds to assigned one
                    self.assertEqual(t, beam_dict.pop(b), 'Returned beam index does not match earlier assigned index')

                # Reassign earlier freed beam one more time
                b = self.s.m._get_beam(i, True)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotIn(b, beam_dict, 'A beam index was returned twice')
                self.assertEqual(old_b, b, 'Target was not reassigned to earlier beam')
                beam_dict[b] = i  # Add beam index with target to keep track

                # Free one beam one more time
                b = self.s.m._get_beam(i, False)
                self.assertIsInstance(b, np.int32, 'Returned beam index is not an Numpy int32')
                self.assertNotEqual(b, self.s.m._BeamConfig.NO_BEAM, 'Returned NO_BEAM while action was required')
                self.assertIn(b, beam_dict, 'Returned a non-assigned beam index')
                # Remove beam index to keep track and verify if beam index corresponds to assigned one
                self.assertEqual(i, beam_dict.pop(b), 'Returned beam index does not match earlier assigned index')


class GetBeam33TestCase(GetBeam2TestCase):
    # Test with 33 beams
    N = 33


if __name__ == '__main__':
    unittest.main()
