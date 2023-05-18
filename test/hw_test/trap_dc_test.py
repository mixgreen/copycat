import os
from dax.util.output import temp_dir
from unittest.mock import patch

from dax.experiment import *
from dax.modules.trap_dc import SmartDma, TrapDcModule
from trap_dac_utils.reader import BaseReader

import test.hw_test

_CONFIG_PATH = 'config'


class DmaTrapDcModule(TrapDcModule):
    def build(self, *, solution_path, map_file):
        super(DmaTrapDcModule, self).build(key='zotino0',
                                           solution_path=solution_path,
                                           map_file=map_file)
        self.dma_cfg = self.create_smart_dma('dma.yml', cls=SmartDma)

    def init(self):
        super().init()

    def post_init(self):
        super().post_init()


class _DmaTestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        with temp_dir():
            super(_DmaTestSystem, self).build()
            f = open("test_map.csv", "w")
            os.makedirs(_CONFIG_PATH)
            open(_CONFIG_PATH + "/dx.csv", "w+")
            self.trap_dc = DmaTrapDcModule(self,
                                           'trap_dc',
                                           solution_path='.',
                                           map_file=os.getcwd() + '/' + f.name,
                                           **kwargs)


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        with temp_dir():
            super(_TestSystem, self).build()
            f = open("test_map.csv", "w")
            os.makedirs(_CONFIG_PATH)
            open(_CONFIG_PATH + "/dx.csv", "w+")
            self.trap_dc = TrapDcModule(self,
                                        'trap_dc',
                                        key='zotino0',
                                        solution_path='.',
                                        map_file=os.getcwd() + '/' + f.name,
                                        **kwargs)


class TrapDcHwTestCase(test.hw_test.HardwareTestCase):

    @patch.object(BaseReader, '_read_channel_map')
    def setUp(self, _) -> None:
        super().setUp()
        self.env = self._construct_env()

    @patch.object(BaseReader, 'read_config')
    @patch.object(BaseReader, '_read_channel_map')
    def _construct_dma_env(self, cfg, _, mock_read_config, *args, **kwargs):
        mock_read_config.return_value = cfg
        return self.construct_env(_DmaTestSystem, *args, **kwargs)

    @patch.object(BaseReader, '_read_channel_map')
    def _construct_env(self, _, *args, **kwargs):
        return self.construct_env(_TestSystem, *args, **kwargs)

    @patch.object(SmartDma, 'line_delays')
    @patch.object(SmartDma, 'solution_mus')
    @patch.object(SmartDma, 'names')
    @patch.object(SmartDma, 'solution_dict')
    @patch.object(BaseReader, 'read_config')
    @patch.object(BaseReader, '_read_channel_map')
    def test_update_dma(self, _, _mock_read_config, mock_solution_dict, mock_names,
                        mock_solution_mus, mock_line_delays):
        dma_dict = {'system.trap_dc.s1': ('abc123', 1.0),
                    'system.trap_dc.s2': ('abc456', 2.0),
                    'system.trap_dc.s3': ('abc789', 3.0),
                    'system.trap_dc.s4': ('abc101112', 4.0)}
        mock_solution_dict.return_value = dma_dict

        mock_names.return_value = ['s1', 's2', 's3', 's4']
        mock_solution_mus.return_value = [[([1, 2, 3]), ([3])],
                                          [([0, 0, 0]), ([1])],
                                          [([4, 4, 4]), ([5])],
                                          [([5, 6, 5]), ([0])]]
        mock_line_delays.return_value = [1.0, 2.0, 3.0, 4.0]

        cfg = self.env.trap_dc.create_smart_dma("config.json", cls=SmartDma)
        assert isinstance(cfg, SmartDma)

        # The below two lines effectively mock compare_dma_dict
        cfg._erase_names = ['system.trap_dc.s4', 'system.trap_dc.s3', 'system.trap_dc.s5']
        cfg._record_names = ['system.trap_dc.s1', 'system.trap_dc.s3', 'system.trap_dc.s4']
        cfg._keys = ["", "", "", "system.trap_dc.s2"]
        cfg.core_cache.put(cfg._powercycle, [1])

        self.env.trap_dc.init()
        keys = cfg._keys
        self.assertTrue('system.trap_dc.s1' in keys)
        self.assertTrue('system.trap_dc.s2' in keys)
        self.assertTrue('system.trap_dc.s3' in keys)
        self.assertTrue('system.trap_dc.s4' in keys)

        cfg.post_init()
        self.assertDictEqual(cfg.get_dataset_sys("dma_dict"), dma_dict)
        self.assertTrue(len(cfg._handles) == 4)

    @patch.object(SmartDma, 'line_delays')
    @patch.object(SmartDma, 'solution_mus')
    @patch.object(SmartDma, 'names')
    @patch.object(SmartDma, 'solution_dict')
    @patch.object(BaseReader, 'read_config')
    @patch.object(BaseReader, '_read_channel_map')
    def test_update_dma_powercycle(self, _, _mock_read_config, _mock_solution_dict, mock_names,
                                   mock_solution_mus, mock_line_delays):
        mock_names.return_value = ['s1', 's2', 's3', 's4']
        mock_solution_mus.return_value = [[([1, 2, 3]), ([3])],
                                          [([0, 0, 0]), ([1])],
                                          [([4, 4, 4]), ([5])],
                                          [([5, 6, 5]), ([0])]]
        mock_line_delays.return_value = [1.0, 2.0, 3.0, 4.0]

        cfg = self.env.trap_dc.create_smart_dma("config.json", cls=SmartDma)
        assert isinstance(cfg, SmartDma)
        # The below two lines effectively mock compare_dma_dict
        cfg._erase_names = ['system.trap_dc.s4', 'system.trap_dc.s3', 'system.trap_dc.s5']
        cfg._record_names = ['system.trap_dc.s1', 'system.trap_dc.s3', 'system.trap_dc.s4']
        cfg._keys = ["", "", "", "system.trap_dc.s2"]
        cfg.core_cache.put(cfg._powercycle, [])

        self.env.trap_dc.init()
        keys = cfg._keys
        self.assertTrue('system.trap_dc.s1' in keys)
        self.assertTrue('system.trap_dc.s2' in keys)
        self.assertTrue('system.trap_dc.s3' in keys)
        self.assertTrue('system.trap_dc.s4' in keys)
