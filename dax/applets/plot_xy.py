#!/usr/bin/env python3

import numpy as np

import artiq.applets.plot_xy  # type: ignore


class XYPlot(artiq.applets.plot_xy.XYPlot):
    """Extended and backwards compatible version of the standard ARTIQ plot XY applet.

    Documentation about plotting functions can be found at:
    http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html#pyqtgraph.PlotItem.plot
    """

    def data_changed(self, data, mods, title):
        try:
            y = data[self.args.y][1]
        except KeyError:
            return
        x = data.get(self.args.x, (False, None))[1]
        if x is None:
            x = np.arange(len(y))
        error = data.get(self.args.error, (False, None))[1]
        fit = data.get(self.args.fit, (False, None))[1]

        if not len(y) or len(y) != len(x):
            return
        if error is not None and hasattr(error, "__len__"):
            if not len(error):
                error = None
            elif len(error) != len(y):
                return
        if fit is not None:
            if not len(fit):
                fit = None
            elif len(fit) != len(y):
                return

        if self.args.sliding_window is not None:
            # Get window size
            window_size = int(self.args.sliding_window)

            if window_size != 0:
                # Truncate input data based on the window size
                y = y[:window_size]
                x = x[:window_size]
                if error is not None:
                    error = error[:window_size]
                if fit is not None:
                    fit = fit[:window_size]

        self.clear()
        self.plot(x, y, pen=None, symbol="x")
        self.setTitle(title)
        if error is not None:
            # See https://github.com/pyqtgraph/pyqtgraph/issues/211
            if hasattr(error, "__len__") and not isinstance(error, np.ndarray):
                error = np.array(error)
            errbars = pyqtgraph.ErrorBarItem(
                x=np.array(x), y=np.array(y), height=error)
            self.addItem(errbars)
        if fit is not None:
            xi = np.argsort(x)
            self.plot(x[xi], fit[xi])

        # Set labels
        if self.args.x_label is not None:
            self.setLabel('bottom', self.args.x_label)
        if self.args.y_label is not None:
            self.setLabel('left', self.args.y_label)


def main():
    # Create applet object
    applet = artiq.applets.plot_xy.TitleApplet(XYPlot)

    # Add custom arguments
    applet.argparser.add_argument("--x-label", default=None)
    applet.argparser.add_argument("--y-label", default=None)
    applet.argparser.add_argument("--sliding-window", default=None)

    # Taken from ARTIQ code
    applet.add_dataset("y", "Y values")
    applet.add_dataset("x", "X values", required=False)
    applet.add_dataset("error", "Error bars for each X value", required=False)
    applet.add_dataset("fit", "Fit values for each X value", required=False)
    applet.run()


if __name__ == "__main__":
    main()
