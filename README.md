# Duke ARTIQ Extensions (DAX)

DAX is a library that extends the capabilities of [ARTIQ](https://github.com/m-labs/artiq)
while maintaining a vanilla ARTIQ experience. This project was initially created as a framework to develop modular
control software for ARTIQ-based quantum control systems. As the project evolved, additional components and utilities
were added to the repository. Users can implement modular control software for their ARTIQ projects using the DAX
framework or use other components and utilities provided by DAX in existing projects.

Currently, DAX consists of the following main components:

| Component                 | Development |
|---------------------------|-------------|
| `DAX.experiment` (system) | Stable      |
| `DAX.sim`                 | Stable      |
| `DAX.scan`                | Stable      |
| `DAX.servo`               | Beta        |
| `DAX.scheduler`           | Alpha       |
| `DAX.program`             | Alpha       |

Projects related to DAX:

- [DAX-comtools](https://gitlab.com/duke-artiq/dax-comtools), lightweight communication tools for DAX and ARTIQ
- [DAX-applets](https://gitlab.com/duke-artiq/dax-applets), applets for DAX and the ARTIQ dashboard
- [flake8-artiq](https://gitlab.com/duke-artiq/flake8-artiq), flake8 plugin for ARTIQ code
- [h5-artiq](https://gitlab.com/duke-artiq/h5-artiq), utilities for ARTIQ HDF5 archive files
- [trap-dac-utils](https://gitlab.com/duke-artiq/trap-dac-utils), generic utilities for trap DAC controllers and drivers

**DAX.experiment (system)**

The DAX.experiment (system) module is a *framework to develop modular control software*. Using this framework, control
software can be organized into modules and services with the help of a central searchable registry. Additionally, code
portability can be achieved using interfaces and clients.

**DAX.sim**

The DAX.sim module is a *functional simulator* for ARTIQ hardware that allows users to run experiment code without the
need for a physical core device. DAX.sim can help users test, debug, and verify the functionality of their code using
existing testing environments. DAX.sim does not depend on other components of DAX and can be used by any ARTIQ project.

**DAX.scan**

The DAX.scan module contains *lightweight scanning tools* that can be used for n-dimensional scanning type experiments.
The scanning tool provides an experiment template for yielding a single point of data and automates the process of
scanning over one or multiple parameters. DAX.scan is not dependent on other components of DAX and can be used by any
ARTIQ project.

**DAX.servo**

The DAX.servo module contains tools for *servo control flow* that can be used for closed-loop experiments with feedback.
The servo class serves as a template for experiments in which a single iteration of the experiment has to be
described. The servo class automates the process of looping, data handling, and exit routines. DAX.servo is not
dependent on other components of DAX and can be used by any ARTIQ project.

**DAX.scheduler**

The DAX.scheduler module contains a toolkit for *automatic scheduling of experiments*. The scheduling toolkit includes
classes to define jobs that can submit experiments. The provided scheduler takes a job set and schedules jobs
accordingly based on their interval, dependencies, and the chosen scheduling policy. DAX.scheduler is not dependent on
other components of DAX and can be used by any ARTIQ project.

**DAX.program**

The DAX.program module contains base classes that provide an *operation-level API for DAX systems*. Using these base
classes, users can write operation-level programs with an API independent of the underlying system. The programs
themselves should be targeted towards a specific DAX system to work within the constraints of the system and yield the
best run-time performance. A DAX program is designed like a regular ARTIQ experiment and works with the same timing and
execution principles. The execution model of DAX.program follows the accelerator model.

**DAX.util**

A collection of utilities that might be handy for any ARTIQ project.

## Resources

- [Wiki](https://gitlab.com/duke-artiq/dax/-/wikis/home)
- [API reference](https://duke-artiq.gitlab.io/dax/)
- [DAX example project](https://gitlab.com/duke-artiq/dax-example)
- [DAX zoo](https://gitlab.com/duke-artiq/dax-zoo)

## Installation

- [External users](https://gitlab.com/duke-artiq/dax/-/wikis/DAX/Installation)
- [Duke users](https://gitlab.com/duke-artiq/dax/-/wikis/DAX/Installation%20Duke%20users)

## Usage

Users can import the DAX system components and the ARTIQ experiment environment at once using the following import
statement:

```python
from dax.experiment import *
```

Users that would like to use DAX.sim can do so by annotating their device DB:

```python
from dax.sim import enable_dax_sim

device_db = enable_dax_sim(enable=True, ddb={
    # Your regular device db...
})
```

The scanning tools in DAX.scan can be imported using the following import statement:

```python
from dax.scan import *
```

The servo tools in DAX.servo can be imported using the following import statement:

```python
from dax.servo import *
```

The scheduling tools in DAX.scheduler can be imported using the following import statement:

```python
from dax.scheduler import *
```

The base classes and utilities for DAX.program can be imported using the following import statement:

```python
from dax.program import *
```

## Versioning

The major version number of DAX matches the version of the targeted ARTIQ release.

## Testing

Use pytest (installed separately) to run the DAX unit tests.

```shell
$ pytest
```

## Main contributors

- Leon Riesebos (Duke University)
- Brad Bondurant (Duke University)

## Publications

- DAX system: [Modular Software for Real-Time Quantum Control Systems (2022)](https://doi.org/10.1109/QCE53715.2022.00077)
- DAX.sim: [Functional Simulation of Real-Time Quantum Control Software (2022)](https://doi.org/10.1109/QCE53715.2022.00076)
- DAX.scheduler: [Universal Graph-Based Scheduling for Quantum Systems (2021)](https://doi.org/10.1109/MM.2021.3094968)

## Acknowledgements

The development of DAX is primarily funded by EPiQC, an NSF Expeditions in Computing (1832377), and the NSF STAQ
project (1818914). The work is also partially funded by the IARPA LogiQ program (W911NF-16-1-0082) and the DOE ASCR
Testbed QSCOUT.

More information about these projects can be found at:

- EPiQC: https://www.epiqc.cs.uchicago.edu/
- STAQ: https://staq.pratt.duke.edu/
- LogiQ: https://www.iarpa.gov/index.php/research-programs/logiq
- QSCOUT: https://www.sandia.gov/quantum/Projects/QSCOUT.html
