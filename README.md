# DAX

Duke ARTIQ Extensions (DAX).

DAX is a library that extends the capabilities of [ARTIQ](https://github.com/m-labs/artiq)
while maintaining a vanilla ARTIQ experience. The goal of the project is to provide a framework that enables the
development of modular and transparently layered software for current state-of-the-art and future-generation quantum
control systems. Users can implement software for their system using the DAX framework and combine it with other
components and utilities provided by DAX.

Projects related to DAX:

- [DAX comtools](https://gitlab.com/duke-artiq/dax-comtools), lightweight communication tools for DAX and ARTIQ
- [DAX applets](https://gitlab.com/duke-artiq/dax-applets), applets for DAX and the ARTIQ dashboard
- [flake8-artiq](https://gitlab.com/duke-artiq/flake8-artiq), flake8 plugin for ARTIQ code

More information about ARTIQ can be found in the [ARTIQ manual](https://m-labs.hk/artiq/manual/).

**DAX.sim**

The DAX.sim sub-module is a *functional simulator* for ARTIQ hardware that allows users to run experiment code without
the need for a physical core device. DAX.sim can help users to test, debug, and verify the functionality of their code.
DAX.sim does not depend on other components of DAX and can be used by any ARTIQ project.

**DAX.scan**

The DAX.scan sub-module contains *lightweight scanning tools* that can be used for n-dimensional scanning type
experiments. The scanning tool provides an experiment template for yielding a single point of data and automates the
process of scanning over one or multiple parameters. DAX.scan is not dependent on other components of DAX and can be
used by any ARTIQ project.

**DAX.scheduler**

The DAX.scheduler sub-module contains a toolkit for *automatic scheduling of experiments*. The scheduling toolkit
includes classes to define jobs that can submit experiments. The provided scheduler takes a job set and schedules the
jobs accordingly based on their interval, dependencies, and the chosen scheduling policy. DAX.scheduler is not dependent
on other components of DAX and can be used by any ARTIQ project.

**DAX.program**

The DAX.program sub-module contains base classes that provide an *operation-level API to DAX systems*. Using these base
classes, users can write operation-level programs with an API that is independent of the underlying system. The programs
themselves should be targeted towards a specific DAX system to work within the restrictions of the system and yield the
best run-time performance. A DAX program is designed like a regular ARTIQ experiment and works based on the same timing
and execution principles. The execution model of DAX.program follows the accelerator model.

## Resources

- [DAX wiki](https://gitlab.com/duke-artiq/dax/-/wikis/home)
- [DAX documentation (autogenerated content only)](https://duke-artiq.gitlab.io/dax/)
- [DAX example project](https://gitlab.com/duke-artiq/dax-example)

## Installation

- [External users](https://gitlab.com/duke-artiq/dax/-/wikis/DAX/Installation)
- [Duke users](https://gitlab.com/duke-artiq/dax/-/wikis/DAX/Installation%20Duke%20users)

## Usage

Users can import DAX and the ARTIQ experiment environment at once using the following import statement:

```python
from dax.experiment import *
```

Users that would like to use DAX.sim can do so by annotating their device DB in the following way:

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

The scheduling tools in DAX.scheduler can be imported using the following import statement:

```python
from dax.scheduler import *
```

The base classes and utilities of DAX.program can be imported using the following import statement:

```python
from dax.program import *
```

## Versioning

The major version number of DAX matches the version of the targeted ARTIQ release.

## Development

Below you will find the development stage of each component in the DAX library.

| Component                 | Development stage |
|---------------------------|-------------------|
| `DAX.experiment` (system) | Stable            |
| `DAX.sim`                 | Beta              |
| `DAX.scan`                | Stable            |
| `DAX.scheduler`           | Alpha             |
| `DAX.program`             | Alpha             |

## Testing

Use pytest or Python unittest to run the DAX unit tests. We recommend using pytest.

```shell
$ pytest
```

## Main contributors

- Leon Riesebos (Duke University)
- Brad Bondurant (Duke University)

## Publications

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
