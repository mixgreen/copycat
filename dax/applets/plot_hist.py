#!/usr/bin/env python3

import numpy as np
import PyQt5  # type: ignore
import pyqtgraph  # type: ignore

import artiq.applets.simple  # type: ignore


class HistogramPlot(pyqtgraph.PlotWidget):
    """Plot histogram applet.

    This applet is not compatible with ARTIQ's plot histogram applet and reads a custom data structure.

    The expected data structure is organized as `[counts_0, counts_1, ...]`.
    Each counts structure is a list of frequencies starting from count 0.

    Documentation about plotting functions can be found at:
    http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html#pyqtgraph.PlotItem.plot
    """

    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args

    def data_changed(self, data, mods, title):
        # Obtain data
        try:
            y = data[self.args.y][1]
            if not y:
                return  # Skip empty data
        except KeyError:
            return

        if self.args.index is not None:
            # Index was provided, just plot one element
            y = [y[self.args.index]]

        # Generate X values based on the length of the data (HDF5 dataset size is always homogeneous)
        x = list(range(len(y[0]) + 1))

        # Plot
        self.clear()
        for counts in y:
            # Convert dict to plot values and plot
            self.plot(x, counts, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

        # Set title and labels
        self.setTitle(title)
        self.setLabel('bottom', self.args.x_label)
        self.setLabel('left', self.args.y_label)


def main():
    # Create an applet object
    applet = artiq.applets.simple.TitleApplet(HistogramPlot, default_update_delay=0.1)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None, type=str)
    applet.argparser.add_argument("--y-label", default=None, type=str)
    applet.argparser.add_argument("--index", default=None, type=int)

    # Add datasets
    applet.add_dataset("y", "dataset with histograms")
    applet.run()


if __name__ == "__main__":
    main()
