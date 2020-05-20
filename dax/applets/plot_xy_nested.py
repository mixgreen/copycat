#!/usr/bin/env python3

import numpy as np
import PyQt5  # type: ignore
import pyqtgraph  # type: ignore

import artiq.applets.simple  # type: ignore


class NestedXYPlot(pyqtgraph.PlotWidget):
    """Nested plot XY applet.

    Expects Y-data in a format like `[ [y_0, z_0], [y_1, z_1], ... ]`.

    Documentation about plotting functions can be found at:
    http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html#pyqtgraph.PlotItem.plot
    """

    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args

    def data_changed(self, data, mods, title):
        # Obtain input data
        try:
            y = data[self.args.y][1]
        except KeyError:
            return
        x = data.get(self.args.x, (False, None))[1]
        if x is None:
            x = np.arange(len(y))

        # Verify input data
        if not len(y) or len(y) != len(x):
            return

        # Handle sliding window
        if self.args.sliding_window is not None:
            # Get window size
            window_size = self.args.sliding_window

            if window_size != 0:
                # Truncate input data based on the window size
                y = y[-window_size:]
                x = x[-window_size:]

        # Plot
        self.clear()
        for y_values in zip(*y):
            # Transform to a list of Y-values and plot
            self.plot(x, y_values, pen=None, symbol="x")

        # Set title and labels
        self.setTitle(title)
        self.setLabel('bottom', self.args.x_label)
        self.setLabel('left', self.args.y_label)


def main():
    # Create applet object
    applet = artiq.applets.simple.TitleApplet(NestedXYPlot, default_update_delay=0.1)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None, type=str)
    applet.argparser.add_argument("--y-label", default=None, type=str)
    applet.argparser.add_argument("--sliding-window", default=None, type=int)

    # Add datasets
    applet.add_dataset("y", "Y data (nested)")
    applet.add_dataset("x", "X values", required=False)
    applet.run()


if __name__ == "__main__":
    main()
