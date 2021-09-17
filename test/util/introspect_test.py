import unittest
import graphviz

import dax.base.system
import dax.util.introspect
from dax.util.artiq import get_managers
from dax.util.output import temp_dir


class _TestSystem(dax.base.system.DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self):
        super(_TestSystem, self).build()
        _TestModule(self, 'm1')
        _TestModuleChild(self, 'm2')
        _TestService(self)


class _TestModule(dax.base.system.DaxModule):
    """Testing module."""

    def init(self):
        pass

    def post_init(self):
        pass


class _TestModuleChild(_TestModule):
    @property
    def foo(self):
        # This property is not available pre-initialization, which is ok
        raise AttributeError


class _TestService(dax.base.system.DaxService):
    SERVICE_NAME = 'test_service'

    def init(self):
        pass

    def post_init(self):
        pass


class IntrospectTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.sys = _TestSystem(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_component_graphviz(self):
        with temp_dir():
            # We can not really test the contents of the graph at this moment
            g = dax.util.introspect.ComponentGraphviz(self.sys)
            self.assertIsInstance(g, graphviz.Digraph)

    def test_relation_graphviz(self):
        with temp_dir():
            # We can not really test the contents of the graph at this moment
            g = dax.util.introspect.RelationGraphviz(self.sys)
            self.assertIsInstance(g, graphviz.Digraph)
