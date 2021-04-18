import unittest
import logging

from dax.util.artiq import is_rpc
import dax.util.logging


def is_rpc_logger(logger):
    return all(is_rpc(getattr(logger, fn), flags={'async'})
               for fn in ['debug', 'info', 'warning', 'error', 'critical', 'exception', 'log'])


class LoggingTestCase(unittest.TestCase):

    def test_get_logger_class(self):
        dax.util.logging.decorate_logger_class(logging.getLoggerClass())
        self.assertTrue(is_rpc_logger(logging.getLoggerClass()))


if __name__ == '__main__':
    unittest.main()
