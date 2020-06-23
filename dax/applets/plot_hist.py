#!/usr/bin/env python3

import numpy as np
import pyqtgraph  # type: ignore

import artiq.applets.simple  # type: ignore

from dax.applets.widget import PlotWidget


class HistogramPlot(PlotWidget):
    """Plot histogram applet.

    This applet is not compatible with ARTIQ's plot histogram applet and reads a custom data structure.
    """

    def update_applet(self, args):
        # Obtain data
        y = self.get_dataset(args.y)

        # Verify input data
        if not len(y):
            return

        if args.index is not None:
            # Index was provided, just plot one element
            y = [y[args.index]]

        # Generate X values based on the length of the data (HDF5 dataset size is always homogeneous)
        x = np.arange(len(y[0]) + 1, dtype=np.float) - 0.5  # View shift of -0.5 to align with x-axis labels

        # Enable legend (has to be done before plotting)
        self.addLegend()

        # Plot
        self.clear()
        for i, counts in enumerate(y):
            # Name of the plot
            name = f'{args.plot_names:s} {i:d}'
            # Convert dict to plot values and plot
            color = pyqtgraph.intColor(i)
            self.plot(x, counts, stepMode=True, fillLevel=0, brush=color, name=name)

        # Set title and labels
        self.setLabel('bottom', args.x_label)
        self.setLabel('left', args.y_label)


def main():
    # Create an applet object
    applet = artiq.applets.simple.TitleApplet(HistogramPlot, default_update_delay=0.1)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None, type=str)
    applet.argparser.add_argument("--y-label", default=None, type=str)
    applet.argparser.add_argument("--index", default=None, type=int)
    applet.argparser.add_argument("--plot-names", default='Plot', type=str)

    # Add datasets
    applet.add_dataset("y", "Dataset with histograms")
    applet.run()


if __name__ == "__main__":
    main()
