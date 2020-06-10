#!/usr/bin/env python3

import numpy as np
import pyqtgraph  # type: ignore

import artiq.applets.simple  # type: ignore

from dax.applets.widget import PlotWidget


class XYPlot(PlotWidget):
    """Extended and backwards compatible version of the standard ARTIQ plot XY applet."""

    def update_applet(self, args):  # noqa: C901
        # Obtain input data
        y = self.get_dataset(args.y)
        x = self.get_dataset(args.x, np.arange(len(y)))
        error = self.get_dataset(args.error, None)
        fit = self.get_dataset(args.fit, None)
        h_lines = self.get_dataset(args.h_lines, [])
        v_lines = self.get_dataset(args.v_lines, [])

        # Verify input data
        if not len(y) or len(y) > len(x):
            return
        if error is not None and hasattr(error, "__len__"):
            if not len(error):
                error = None
            elif len(error) != len(y):
                return
            if not isinstance(error, np.ndarray):
                # See https://github.com/pyqtgraph/pyqtgraph/issues/211
                error = np.array(error)
        if fit is not None and hasattr(error, "__len__"):
            if not len(fit):
                fit = None
            elif len(fit) != len(y):
                return

        if len(x) > len(y):
            # Trim x data
            x = x[:len(y)]

        # Handle sliding window
        if args.sliding_window > 0:
            # Get window size
            window_size = args.sliding_window

            # Truncate input data based on the window size
            v_lines = [v for v in v_lines if len(y) - window_size <= v < len(y)]
            y = y[-window_size:]
            x = x[-window_size:]
            if error is not None:
                error = error[-window_size:]
            if fit is not None:
                fit = fit[-window_size:]

        # Plot
        self.clear()
        self.plot(x=x, y=y, pen=None, symbol='o')
        if error is not None:
            # Plot error bars
            self.addItem(pyqtgraph.ErrorBarItem(x=np.array(x), y=np.array(y), height=error))
        if fit is not None:
            # Plot fit
            self.plot(x, fit)

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
    applet = artiq.applets.simple.TitleApplet(XYPlot, default_update_delay=0.1)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None, type=str)
    applet.argparser.add_argument("--y-label", default=None, type=str)
    applet.argparser.add_argument("--sliding-window", default=0, type=int)

    # Add datasets
    applet.add_dataset("y", "Y values")
    applet.add_dataset("x", "X values", required=False)
    applet.add_dataset("error", "Error bars for each X value", required=False)
    applet.add_dataset("fit", "Fit values for each X value", required=False)
    applet.add_dataset("v-lines", "Vertical lines", required=False)
    applet.add_dataset("h-lines", "Horizontal lines", required=False)
    applet.run()


if __name__ == "__main__":
    main()
