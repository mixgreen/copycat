#!/usr/bin/env python3

import numpy as np
import PyQt5  # make sure pyqtgraph imports Qt5
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette
import pyqtgraph
import itertools
from artiq.applets.simple import TitleApplet


class MainWidget(QWidget):
    def __init__(self, args):
        QWidget.__init__(self)
        self.args = args
        self.plot = LivePlot(args)
        self.current_value = QLabel("", self)
        self.title = QLabel("", self)

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("", self))
        top_layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        top_layout.addWidget(self.current_value, alignment=Qt.AlignRight)
        top_layout.setAlignment(Qt.AlignBottom)
        layout.addLayout(top_layout)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # make it pretty
        layout.setContentsMargins(0, 10, 0, 0)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(pal)
        # self.current_value.setFont(QFont("Arial", pointSize=20))
        # self.title.setFont(QFont("Arial", pointSize=20, weight=QFont.UltraCondensed))

    def data_changed(self, data, mods, title):
        self.plot.data_changed(data, mods)

        try:
            y = data[self.args.y][1]
        except KeyError:
            return

        if not len(y):
            return

        self.current_value.setText("<font style='color:white; font-size:40px;'>{}\t</font>".format(str(y[-1])))

        if not title:
            title = self.args.y

        self.title.setText("<font style='color:darkgray; font-size:25px'>{}</font>".format(title))

class LivePlot(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.x_range = args.points
        self.reset()

    def reset(self, size=0):
        self.clear()
        if not size:
            self.counter = itertools.count()
            self.x = []
            self.symbols = []
            self.sizes = []
        else:
            self.counter = itertools.count(size)
            self.x = list(np.arange(size).tolist())
            self.symbols = ['x']*min(size-1, self.x_range)
            self.sizes = [12]*min(size-1, self.x_range)


    def data_changed(self, data, mods):
        try:
            if type(mods[0]["value"]) is tuple:
                self.reset()
        except KeyError:
            pass

        try:
            y = data[self.args.y][1]
        except KeyError:
            return

        ylen = len(y)

        if not ylen:
            return

        self.x.append(next(self.counter))
        if len(self.x) != ylen:
            self.reset(size=ylen)

        self.clear()
        if ylen <= self.x_range:
            self.symbols.append('x')
            self.sizes.append(12)
            self.plot(self.x, y, pen=None, symbol=self.symbols, symbolBrush='r', symbolSize=self.sizes)
        else:
            self.plot(self.x[-self.x_range:], y[-self.x_range:], pen=None, symbol=self.symbols, symbolBrush='r', symbolSize=self.sizes)

        # self.current_val = self.addItem(pyqtgraph.TextItem(text="hi"))

def main():
    applet = TitleApplet(MainWidget)
    applet.add_dataset("y", "Y values")
    applet.argparser.add_argument("--points", type=int, default=100, help="number of points to show on graph")
    applet.run()

if __name__ == "__main__":
    main()
