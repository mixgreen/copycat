from dax.base import *


class LoadIonService(DaxService):
    SERVICE_NAME = 'load_ion'

    def build(self):
        pass

    def load(self):
        self.setattr_dataset_sys('foo', 1)

    def init(self):
        pass

    def config(self):
        pass
