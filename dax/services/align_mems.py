from dax.base import *
from dax.modules.indv_beam_mems import *
from dax.modules.interfaces.global_beam_if import *
from dax.modules.interfaces.detection_if import *
from dax.modules.interfaces.trap_if import *


class AlignMemsService(DaxService):
    SERVICE_NAME = 'align_mems'

    def build(self):
        # Obtain required modules
        self.ibeam = self.registry.search_module(IndvBeamMemsModule)  # Specifically request IndvBeamMemsModule
        self.gbeam = self.registry.search_module(GlobalBeamInterface)
        self.detect = self.registry.search_module_dict(DetectionInterface)
        self.trap = self.registry.search_module_dict(TrapInterface)

    def load(self):
        pass

    def init(self):
        pass

    def config(self):
        pass

    def align_mems(self):
        pass
