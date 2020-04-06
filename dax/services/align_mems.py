from dax.experiment import *
from dax.modules.indv_beam_mems import *
from dax.modules.interfaces.global_beam_if import *
from dax.modules.interfaces.detection_if import *
from dax.modules.interfaces.trap_if import *


class AlignMemsService(DaxService):
    SERVICE_NAME = 'align_mems'

    def build(self):
        # Obtain required modules
        self.ibeam = self.registry.find_module(IndvBeamMemsModule)  # Specifically request IndvBeamMemsModule
        self.gbeam = self.registry.find_module(GlobalBeamInterface)
        self.detect = self.registry.find_module(DetectionInterface)
        self.trap = self.registry.find_module(TrapInterface)

    def init(self):
        pass

    def post_init(self):
        pass

    def align_mems(self):
        pass
