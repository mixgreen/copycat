import os
import string
import typing
import random

import numpy as np
from dax.util.output import temp_dir
from unittest.mock import patch
import pathlib

from dax.experiment import *
from dax.modules.trap_dc import ZotinoReader, TrapDcModule
from trap_dac_utils.reader import SpecialCharacter, BaseReader
from trap_dac_utils.types import LABEL_FIELD
import dax.sim.coredevice.ad53xx
import dax.sim.test_case
from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        with temp_dir():
            super(_TestSystem, self).build()
            f = open("test_map.csv", "w")
            self.trap_dc = TrapDcModule(self,
                                        'trap_dc',
                                        key='zotino0',
                                        solution_path='.',
                                        map_file=os.getcwd() + '/' + f.name,
                                        **kwargs)


class TrapDcTestCase(dax.sim.test_case.PeekTestCase):
    _NUM_CHANNELS = 32
    SEED = None

    _RNG = random.Random(SEED)
    _VREF = 5

    PATH_DATA = [{'A': -10., 'B': 0., 'C': 0., 'D': 0.,
                  'E': SpecialCharacter.X},
                 {'A': 1., 'B': 0., 'C': 0, 'D': 0,
                  'E': SpecialCharacter.X},
                 {'A': 1., 'B': 2., 'C': 0., 'D': 0.,
                  'E': SpecialCharacter.X},
                 {'A': 1., 'B': 2., 'C': 3., 'D': 4.,
                  'E': SpecialCharacter.X}]
    MAP_DATA = [{'label': 'A', 'channel': '2'},
                {'label': 'B', 'channel': '3'},
                {'label': 'C', 'channel': '4'},
                {'label': 'D', 'channel': '5'},
                {'label': 'E', 'channel': '6'}]

    @patch.object(BaseReader, '_read_channel_map')
    def setUp(self, _) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self._construct_env()

    _DEVICE_DB: typing.Dict[str, typing.Any] = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'core_cache': {
            'type': 'local',
            'module': 'artiq.coredevice.cache',
            'class': 'CoreCache'
        },
        'core_dma': {
            'type': 'local',
            'module': 'artiq.coredevice.dma',
            'class': 'CoreDMA'
        },
        "spi_zotino0": {
            "type": "local",
            "module": "artiq.coredevice.spi2",
            "class": "SPIMaster",
            "arguments": {"channel": 0x00001a}
        },
        "ttl_zotino0_ldac": {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x00001b}
        },
        "ttl_zotino0_clr": {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x00001c}
        },
        "zotino0": {
            "type": "local",
            "module": "artiq.coredevice.zotino",
            "class": "Zotino",
            "arguments": {
                "spi_device": "spi_zotino0",
                "ldac_device": "ttl_zotino0_ldac",
                "clr_device": "ttl_zotino0_clr"
            }
        }
    }

    def _construct_env(self, **kwargs):
        return self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def _test_uninitialized(self):

        self.expect(self.env.trap_dc._zotino, 'init', 'x')
        for i in range(self._NUM_CHANNELS):
            self.expect(self.env.trap_dc._zotino, f'v_out_{i}', 'x')
            self.expect(self.env.trap_dc._zotino, f'v_offset_{i}', 'x')

    @patch.object(BaseReader, '_read_channel_map')
    @patch.object(TrapDcModule, 'get_system_key')
    def test_dma(self, mock_get_system_key, _):
        num_path_rows = 10
        mock_get_system_key.return_value = "ZotinoTest"
        with temp_dir():
            self._test_uninitialized()
            self.env.dax_init()
            post_delay = 1000000000
            for _ in range(_NUM_SAMPLES):
                num_datas = []
                voltages = []
                path = []
                for _ in range(num_path_rows):
                    num_data, v, v_mu, c = self._generate_random_compressed_line()
                    path.append((v_mu, c))
                    num_datas.append(num_data)
                    voltages.append(v)

                name = self.env.trap_dc.record_dma_mu(
                    "TestName", path, post_delay)
                handle = self.env.trap_dc.get_dma_handle(name)
                with parallel:
                    self.env.trap_dc.shuttle_dma_handle(handle)
                    self.assertEqual(
                        self.env.core_dma._dma_play_name.pull(), "ZotinoTest")

    @patch.object(BaseReader, '_read_channel_map')
    def test_set_line(self, _):
        with temp_dir():
            self._test_uninitialized()
            self.env.dax_init()
            for _ in range(_NUM_SAMPLES):
                num_data, v, v_mu, c = self._generate_random_compressed_line()
                with self.subTest(v=v):
                    # Call functions
                    # self.env.trap_dc._zotino.write_offset_dacs_mu(o)
                    self.env.trap_dc.set_line((v_mu, c))
                    # Test
                    for i in range(num_data):
                        self.expect_close(self.env.trap_dc._zotino,
                                          f'v_out_{c[i]}',
                                          v[i],
                                          places=3)

    def _generate_random_compressed_line(self):
        num_data = self.rng.randrange(1, self._NUM_CHANNELS)
        c = self.rng.sample(range(self._NUM_CHANNELS), num_data)
        # o = self.rng.sample(range(2 ** 14), num_data)
        voltages = [self.rng.uniform(0 * V, self.env.trap_dc._zotino.vref * 3.9) - 2 * self.env.trap_dc._zotino.vref
                    for _ in range(num_data)]
        # Adjust voltage to make sure it is in range
        v_mu = [self.env.trap_dc._zotino.voltage_to_mu(voltage=v) for v in voltages]
        return num_data, voltages, v_mu, c

    @patch.object(BaseReader, '_read_channel_map')
    def test_shuttle_min_line_delay(self, mock_read_channel_map):
        mock_read_channel_map.return_value = [{'label': 'A', 'channel': '0'},
                                              {'label': 'B', 'channel': '1'},
                                              {'label': 'C', 'channel': '2'},
                                              {'label': 'D', 'channel': '3'}]
        s = self._construct_env()
        s.trap_dc.init()
        try:
            s.trap_dc.shuttle_mu([], 1)
            assert False
        except ValueError as e:
            assert str(e) == f"Line Delay must be greater than {s.trap_dc._min_line_delay_mu}"

    @patch.object(BaseReader, '_read_channel_map')
    def test_record_dma_min_line_delay(self, mock_read_channel_map):
        mock_read_channel_map.return_value = [{'label': 'A', 'channel': '0'},
                                              {'label': 'B', 'channel': '1'},
                                              {'label': 'C', 'channel': '2'},
                                              {'label': 'D', 'channel': '3'}]
        s = self._construct_env()
        s.trap_dc.init()
        try:
            s.trap_dc.record_dma_mu("", [], 1)
            assert False
        except ValueError as e:
            assert str(e) == f"Line Delay must be greater than {s.trap_dc._min_line_delay_mu}"

    @patch.object(BaseReader, '_read_channel_map')
    def test_shuttle(self, _):
        with temp_dir():
            shuttle_solution_mu = [([32768, 32768, 0, 32768, 32768,
                                     32768, 32768, 32768, 32768, 32768,
                                     32768, 32768, 32768, 32768, 32768,
                                     32768, 32768, 32768, 32768,
                                     32768, 32768, 32768, 32768,
                                     32768, 32768, 32768, 32768,
                                     32768, 32768, 32768, 32768, 32768],
                                    [0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15,
                                     16, 17, 18, 19, 20, 21, 22, 23,
                                     24, 25, 26, 27, 28, 29, 30, 31]),
                                   ([36045], [2]),
                                   ([39322], [3]),
                                   ([42598, 45875], [4, 5])]

            s = self._construct_env()
            s.trap_dc.init()

            line_delay_mu = 100000
            line_delay = s.core.mu_to_seconds(line_delay_mu)

            with parallel:
                s.trap_dc.shuttle(shuttle_solution_mu, line_delay)
                with sequential:

                    # create an offset for testing purposes
                    delay_mu(200)
                    for i in range(32):
                        if i != 2:
                            self.expect_close(s.trap_dc._zotino,
                                              f'v_out_{i}', 0, places=3)
                        else:
                            self.expect_close(s.trap_dc._zotino,
                                              'v_out_2', -10, places=3)
                    delay_mu(line_delay_mu)
                    for i in range(32):
                        if i != 2:
                            self.expect(s.trap_dc._zotino, f'v_out_{i}', 0.0)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_2', 1, places=3)
                    delay_mu(line_delay_mu)
                    for i in range(32):
                        if i != 2 and i != 3:
                            self.expect(s.trap_dc._zotino, f'v_out_{i}', 0.0)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_2', 1, places=3)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_3', 2, places=3)
                    delay_mu(line_delay_mu)
                    for i in range(32):
                        if i not in [2, 3, 4, 5]:
                            self.expect(s.trap_dc._zotino, f'v_out_{i}', 0.0)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_2', 1, places=3)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_3', 2, places=3)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_4', 3, places=3)
                    self.expect_close(s.trap_dc._zotino,
                                      'v_out_5', 4, places=3)
                    delay_mu(line_delay_mu)

    @patch.object(BaseReader, '_read_channel_map')
    def test_mu_conversion(self, _):
        with temp_dir():
            vref = 5.
            s = self._construct_env()
            s.trap_dc.init()
            open('test.csv', 'w')
            reader = ZotinoReader(pathlib.Path('.'),
                                  pathlib.Path('test.csv'))
            reader.init(self.env.trap_dc._zotino)
            for _ in range(_NUM_SAMPLES):
                v = [self._RNG.uniform(-2.0 * vref, 1.99
                                       * vref)  # v < 2*v_ref
                     for _ in range(self._NUM_CHANNELS)]
                with self.subTest(v_ref=vref, v_in=v):
                    mu_array = reader.convert_to_mu(v)
                    for i, mu in enumerate(mu_array):
                        o = dax.sim.coredevice.ad53xx._mu_to_voltage(
                            mu, vref=vref, offset_dacs=0x2000)
                        self.assertAlmostEqual(
                            v[i], o, places=3, msg='Input voltage does not match converted output voltage')

    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_process_solution_random(self,
                                     mock_read_channel_map,
                                     mock_read_solution):
        with temp_dir():
            self.env.trap_dc.init()
            for _ in range(_NUM_SAMPLES):
                headers = self.generate_headers()
                mock_read_solution.return_value = self.generate_path_data(
                    headers)
                headers = mock_read_solution.return_value[0]
                mock_read_channel_map.return_value = self.generate_map_data(
                    headers)
                open('test.csv', 'w')
                reader = ZotinoReader(pathlib.Path('.'),
                                      pathlib.Path('test.csv'))
                reader.init(self.env.trap_dc._zotino)

                read_solution = reader.read_solution("sequential.csv")
                result_zotino_path = reader.process_solution(read_solution)
                map_data = mock_read_channel_map.return_value
                expected_solution = mock_read_solution.return_value
                for i, t in enumerate(result_zotino_path[1:]):
                    print(expected_solution[i + 1])
                    print(t)
                    for j, channel in enumerate(t[1]):
                        label = self.channel_to_label(
                            channel, map_data, reader)
                        self.assertAlmostEqual(
                            expected_solution[i + 1][label], t[0][j], places=3)

    def generate_headers(self):
        return [self.rand_str() for _ in range(self._NUM_CHANNELS)]

    def generate_path_data(self, headers):
        headers = [self.rand_str() for _ in range(self._NUM_CHANNELS)]
        path_data = []
        special = [SpecialCharacter.X]
        for _ in range(self._RNG.randint(1, 50)):
            pool = [
                *special, self._RNG.uniform(-1.95 * self._VREF * V,
                                            1.95 * self._VREF * V)]
            line_map = {header: self._RNG.choice(pool)
                        for header in headers}
            path_data.append(line_map)

        return path_data

    def generate_map_data(self, labels):
        channels = self._RNG.sample(
            range(self._NUM_CHANNELS), self._NUM_CHANNELS)
        return [{LABEL_FIELD: label,
                 ZotinoReader._CHANNEL: str(channels[i])}
                for i, label in enumerate(labels)]

    def rand_str(self):
        return ''.join(self._RNG.choice(string.ascii_letters + string.digits)
                       for _ in range(self._RNG.randint(8, 15)))

    def channel_to_label(self,
                         channel,
                         map_data,
                         reader):
        for d in map_data:
            if d[reader._CHANNEL] == str(channel):
                return d[LABEL_FIELD]
        raise ValueError("Mapped to channel that isn't in channel map")

    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_process_solution(self,
                              mock_read_channel_map,
                              mock_read_solution):
        with temp_dir():
            self.env.trap_dc.init()
            mock_read_channel_map.return_value = self.MAP_DATA
            mock_read_solution.return_value = self.PATH_DATA
            open('test.csv', 'w')
            reader = ZotinoReader(pathlib.Path('.'),
                                  pathlib.Path('test.csv'))
            reader.init(self.env.trap_dc._zotino)
            # below is an example of one possible expected payload
            # channels not required to be ordered, only paired with correct voltages
            expected_zotino_path = [([-10., 0., 0., 0.], [2, 3, 4, 5]),
                                    ([1., 0., 0., 0.], [2, 3, 4, 5]),
                                    ([1., 2., 0., 0.], [2, 3, 4, 5]),
                                    ([1., 2., 3., 4.], [2, 3, 4, 5])]

            read_solution = reader.read_solution("sequential.csv")
            result_zotino_path = reader.process_solution(read_solution)

            self.assertIsInstance(result_zotino_path, list)
            self.assertEqual(len(result_zotino_path),
                             len(expected_zotino_path))

            for i, t in enumerate(result_zotino_path):
                self.assertIsInstance(t, tuple)
                self.assertEqual(len(t), 2)
                self.assertListEqual(t[0], expected_zotino_path[i][0])
                self.assertListEqual(t[1], expected_zotino_path[i][1])

    @patch.object(ZotinoReader, 'process_solution')
    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_get_path(self, _, _mock_read_solution, mock_process_solution):
        with temp_dir():
            self.env.trap_dc.init()
            mock_process_solution.return_value = [([-5., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 3., 4.], [2, 3, 4, 5])]
            expected_prepared_path = [([-10., 0., 0., 0.], [2, 3, 4, 5]),
                                      ([2.], [2]),
                                      ([4.], [3]),
                                      ([6., 8.], [4, 5])]
            prepared_path_result = self.env.trap_dc._read_solution(
                "name_of_file.csv", multiplier=2)
            self.assertListEqual(prepared_path_result, expected_prepared_path)

    @patch.object(ZotinoReader, 'process_solution')
    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_get_path_reverse(self, _, _mock_read_solution, mock_process_solution):
        with temp_dir():
            self.env.trap_dc.init()
            mock_process_solution.return_value = [([-10., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 3., 4.], [2, 3, 4, 5])]
            expected_prepared_path = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                                      ([0., 0.], [4, 5]),
                                      ([0.], [3]),
                                      ([-10.], [2])]
            prepared_path_result = self.env.trap_dc._read_solution(
                "name_of_file.csv", reverse=True)
            self.assertListEqual(prepared_path_result, expected_prepared_path)

    @patch.object(ZotinoReader, 'process_solution')
    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_get_path_segment(self, _, _mock_read_solution, mock_process_solution):
        with temp_dir():
            self.env.trap_dc.init()
            mock_process_solution.return_value = [([-10., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 3., 4.], [2, 3, 4, 5])]
            expected_prepared_path = [([1., 0., 0., 0.], [2, 3, 4, 5]),
                                      ([2.], [3])]
            prepared_path_result = self.env.trap_dc._read_solution(
                'name_of_file.csv', 1, 2)
            self.assertListEqual(prepared_path_result, expected_prepared_path)

    @patch.object(ZotinoReader, 'process_solution')
    @patch.object(BaseReader, 'read_solution')
    @patch.object(BaseReader, '_read_channel_map')
    def test_get_line(self, _, _mock_read_solution, mock_process_solution):
        with temp_dir():
            self.env.trap_dc.init()
            mock_process_solution.return_value = [([-10., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 0., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 0., 0.], [2, 3, 4, 5]),
                                                  ([1., 2., 3., 4.], [2, 3, 4, 5])]
            expected_prepared_line = ([3.5, 7., 0., 0.], [2, 3, 4, 5])
            prepared_line_result = self.env.trap_dc._read_line(
                'name_of_file.csv', 2, 3.5)
            self.assertTupleEqual(prepared_line_result, expected_prepared_line)

    @patch.object(BaseReader, '_read_channel_map')
    def test_reader_zotino_uninitialized(self, _):
        reader = ZotinoReader(pathlib.Path('.'),
                              pathlib.Path('test.csv'))
        try:
            reader.convert_to_mu([1.0, 2.0, 3.0])
        except RuntimeError as e:
            assert str(e) == "Must initialize reader using init method to use function convert_to_mu"

    @patch.object(BaseReader, '_read_channel_map')
    def test_calculate_low_slack(self, _):
        self.env.trap_dc.init()
        test_solution = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                         ([0., 0.], [4, 5]),
                         ([0.], [3]),
                         ([-10.], [2])]
        slack = self.env.trap_dc.calculate_slack(test_solution, .0002)
        l0 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[0][0])))
        assert slack > l0
        assert slack < l0 + self.env.trap_dc._min_line_delay_mu

    @patch.object(BaseReader, '_read_channel_map')
    def test_calculate_high_slack(self, _):
        line_delay = .0000016
        self.env.trap_dc.init()
        test_solution = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                         ([0., 0.], [4, 5]),
                         ([0.], [3]),
                         ([-10.], [2])]
        slack = self.env.trap_dc.calculate_slack(test_solution, line_delay)
        l0 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[0][0])))
        l1 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[1][0])))
        l2 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[2][0])))
        l3 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[3][0])))
        assert slack > l0 + l1 + l2 + l3 - 3 * line_delay
        assert slack < l0 + l1 + l2 + l3 - 3 * line_delay + self.env.trap_dc._min_line_delay_mu

    @patch.object(BaseReader, '_read_channel_map')
    def test_calculate_dma_low_slack(self, _):
        line_delay = .00003
        self.env.trap_dc.init()
        test_solution = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                         ([0., 0.], [4, 5]),
                         ([0.], [3]),
                         ([-10.], [2])]
        slack = self.env.trap_dc.calculate_dma_slack(test_solution, line_delay)
        l0 = self.env.core.mu_to_seconds(
            self.env.trap_dc._calculator._calculate_line_comm_delay_mu(len(test_solution[0][0]), True))
        assert slack > l0
        assert slack < l0 + self.env.trap_dc._min_line_delay_mu + self.env.trap_dc._calculator._dma_startup_time_mu

    @patch.object(BaseReader, '_read_channel_map')
    def test_calculate_slack_too_low(self, _):
        line_delay = .000000001
        self.env.trap_dc.init()
        test_solution = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                         ([0., 0.], [4, 5]),
                         ([0.], [3]),
                         ([-10.], [2])]
        try:
            self.env.trap_dc.calculate_slack(test_solution, line_delay)
            assert False
        except ValueError as e:
            assert str(e) == f"Line Delay must be greater than {self.env.trap_dc._min_line_delay_mu}"

    @patch.object(BaseReader, '_read_channel_map')
    def test_calculate_dma_slack_too_low(self, _):
        line_delay = .0000001
        self.env.trap_dc.init()
        test_solution = [([1., 2., 3., 4.], [2, 3, 4, 5]),
                         ([0., 0.], [4, 5]),
                         ([0.], [3]),
                         ([-10.], [2])]
        try:
            self.env.trap_dc.calculate_dma_slack(test_solution, line_delay)
            assert False
        except ValueError as e:
            assert str(e) == f"Line Delay must be greater than {self.env.trap_dc._min_line_delay_mu}"

    @patch.object(BaseReader, '_read_channel_map')
    def test_configure_calculator(self, _):
        dma_startup_mu = 1210
        dma_startup_time = self.env.core.mu_to_seconds(dma_startup_mu)
        self.env.trap_dc.init()
        self.env.trap_dc.configure_calculator(dma_startup_time=dma_startup_time,
                                              comm_delay_intercept_mu=np.int64(2),
                                              comm_delay_slope_mu=np.int64(3),
                                              dma_comm_delay_intercept_mu=np.int64(4),
                                              dma_comm_delay_slope_mu=np.int64(5))

        self.assertAlmostEqual(self.env.trap_dc._calculator._dma_startup_time_mu, dma_startup_mu, delta=2.0)
        assert self.env.trap_dc._calculator._comm_delay_intercept_mu == np.int64(2)
        assert self.env.trap_dc._calculator._comm_delay_slope_mu == np.int64(3)
        assert self.env.trap_dc._calculator._dma_comm_delay_intercept_mu == np.int64(4)
        assert self.env.trap_dc._calculator._dma_comm_delay_slope_mu == np.int64(5)

        self.env.trap_dc.configure_calculator()

        self.assertAlmostEqual(self.env.trap_dc._calculator._dma_startup_time_mu, dma_startup_mu, delta=2.0)
        assert self.env.trap_dc._calculator._comm_delay_intercept_mu == np.int64(2)
        assert self.env.trap_dc._calculator._comm_delay_slope_mu == np.int64(3)
        assert self.env.trap_dc._calculator._dma_comm_delay_intercept_mu == np.int64(4)
        assert self.env.trap_dc._calculator._dma_comm_delay_slope_mu == np.int64(5)
