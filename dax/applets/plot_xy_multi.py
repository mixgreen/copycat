#!/usr/bin/env python3

import numpy as np
import pyqtgraph  # type: ignore

import artiq.applets.simple

from dax.applets.widget import PlotWidget


class MultiXYPlot(PlotWidget):
    """Plot XY applet for multiple graphs."""

    def update_applet(self, args):
        # Obtain input data
        y = np.asarray(self.get_dataset(args.y))
        x = np.asarray(self.get_dataset(args.x, np.arange(len(y))))
        h_lines = self.get_dataset(args.h_lines, [])
        v_lines = self.get_dataset(args.v_lines, [])

        # Verify input data
        if not len(y) or len(y) > len(x):
            return

        if len(x) > len(y):
            # Trim x data
            x = x[:len(y)]

        # Sort based on x data
        idx = x.argsort()
        x = x[idx]
        y = y[idx]

        # Handle sliding window
        if args.sliding_window > 0:
            # Get window size
            window_size = args.sliding_window

            # Truncate input data based on the window size
            v_lines = [v for v in v_lines if len(y) - window_size <= v < len(y)]
            y = y[-window_size:]
            x = x[-window_size:]

        # Enable legend (has to be done before plotting)
        self.addLegend()

        # Plot
        self.clear()
        for i, y_values in enumerate(zip(*y)):
            # Name of the plot
            name = f'{args.plot_names:s} {i:d}'
            # Transform to a list of Y-values and plot
            color = pyqtgraph.intColor(i)
            self.plot(x, y_values, pen=color, symbol='o', symbolBrush=color, name=name)

        # Plot horizontal and vertical lines
        for h in h_lines:
            self.addLine(y=h)
        for v in v_lines:
            self.addLine(x=v)

        # Set labels
        self.setLabel('bottom', args.x_label)
        self.setLabel('left', args.y_label)


def main():
    # Create applet object
    applet = artiq.applets.simple.TitleApplet(MultiXYPlot, default_update_delay=0.1)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None, type=str)
    applet.argparser.add_argument("--y-label", default=None, type=str)
    applet.argparser.add_argument("--sliding-window", default=0, type=int)
    applet.argparser.add_argument("--plot-names", default='Plot', type=str)

    # Add datasets
    applet.add_dataset("y", "Y data (multiple graphs)")
    applet.add_dataset("x", "X values", required=False)
    applet.add_dataset("v-lines", "Vertical lines", required=False)
    applet.add_dataset("h-lines", "Horizontal lines", required=False)
    applet.run()


if __name__ == "__main__":
    main()
