#!/usr/bin/env python3
# flake8: noqa

import numpy as np
# make sure pyqtgraph imports Qt5
import PyQt5  # type: ignore
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout  # type: ignore
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtGui import QFont, QPalette  # type: ignore
import pyqtgraph  # type: ignore
import itertools
from artiq.applets.simple import TitleApplet
from pyqtgraph import InfiniteLine


class MainWidget(QWidget):
    def __init__(self, args):
        QWidget.__init__(self)
        self.args = args
        current_value = QLabel("", self)
        self.title = QLabel("", self)

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("", self))
        top_layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        top_layout.addWidget(current_value, alignment=Qt.AlignRight)
        top_layout.setAlignment(Qt.AlignBottom)
        layout.addLayout(top_layout)
        self.plot = LivePlot(current_value, args)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # make it pretty
        layout.setContentsMargins(0, 10, 0, 0)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(pal)

    def enterEvent(self, evt):
        super().enterEvent(evt)
        self.plot.set_hover_mode(True)

    def leaveEvent(self, evt):
        super().leaveEvent(evt)
        self.plot.set_hover_mode(False)

    def data_changed(self, data, mods, title):
        self.plot.data_changed(data, mods)

        if not title:
            title = self.args.y

        self.title.setText("<font style='color:darkgray; font-size:40px'>{}</font>".format(title))


class LivePlot(pyqtgraph.PlotWidget):
    def __init__(self, current_value_label, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.x_range = args.points
        self.setMouseTracking(True)
        self.current_value_label = current_value_label
        self.hover_mode = False
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
            self.symbols = ['x'] * min(size - 1, self.x_range)
            self.sizes = [12] * min(size - 1, self.x_range)

    def set_hover_mode(self, hover_mode):
        self.hover_mode = hover_mode
        if not hover_mode:
            self.current_value_label.setText(
                "<font style='color:white; font-size:40px;'>{}\t</font>".format(str(self.y[-1])))
            if len(self.getPlotItem().items) == 3:  # make sure crosshair objects still exist
                self.ch_hline.hide()
                self.ch_vline.hide()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        plot_item = self.getPlotItem()
        pos = plot_item.getViewBox().mapSceneToView(ev.pos())
        idx = int(np.round(pos.x()))
        ylen = len(self.y)
        if idx < ylen - self.x_range:
            idx = ylen - self.x_range
        elif idx >= ylen:
            idx = ylen - 1
        self.current_value_label.setText(
            "<font style='color:white; font-size:40px;'>{}\t</font>".format(str(self.y[idx])))

        # crosshair
        if len(self.plotItem.items) == 1:  # plotItem only contains the data item
            self.ch_hline = InfiniteLine(angle=0, movable=False)
            self.ch_vline = InfiniteLine(angle=90, movable=False)
            plot_item.addItem(self.ch_hline)
            plot_item.addItem(self.ch_vline)
        self.ch_hline.setPos(self.y[idx])
        self.ch_vline.setPos(idx)
        self.ch_hline.show()
        self.ch_vline.show()

    def data_changed(self, data, mods):
        try:
            if type(mods[0]["value"]) is tuple:
                self.reset()
        except KeyError:
            pass

        try:
            self.y = data[self.args.y][1]
        except KeyError:
            return

        ylen = len(self.y)

        if not ylen:
            return

        self.x.append(next(self.counter))
        if len(self.x) != ylen:
            self.reset(size=ylen)

        self.clear()
        if ylen <= self.x_range:
            self.symbols.append('x')
            self.sizes.append(12)
            self.plot(self.x, self.y, pen=None, symbol=self.symbols, symbolBrush='r', symbolSize=self.sizes)
        else:
            self.plot(self.x[-self.x_range:], self.y[-self.x_range:], pen=None, symbol=self.symbols, symbolBrush='r',
                      symbolSize=self.sizes)

        if not self.hover_mode:
            self.current_value_label.setText(
                "<font style='color:white; font-size:40px;'>{}\t</font>".format(str(self.y[-1])))


def main():
    applet = TitleApplet(MainWidget)
    applet.add_dataset("y", "Y values")
    applet.argparser.add_argument("--points", type=int, default=100, help="number of points to show on graph")
    applet.run()


if __name__ == "__main__":
    main()
