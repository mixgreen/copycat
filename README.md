# DAX

Duke [ARTIQ](https://github.com/m-labs/artiq) Extensions (DAX).

DAX is a library that extends the capabilities of ARTIQ while maintaining a vanilla ARTIQ experience.
More information about ARTIQ can be found in the [ARTIQ manual](https://m-labs.hk/artiq/manual/).

## Usage

Users can import DAX and the ARTIQ experiment environment at once using the following import statement:

```python
from dax.experiment import *
```

## Testing

To run the DAX unit tests, execute the following command in the root directory of DAX:

```bash
python3 -m unittest
```
