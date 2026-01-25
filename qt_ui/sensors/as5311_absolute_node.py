import time

from PySide6.QtWidgets import QWidget,QFormLayout, QVBoxLayout, QGroupBox, QLabel

import pyqtgraph as pg
import numpy as np

from stim_math.sensors.as5311 import AS5311Data

from qt_ui.sensors.sensor_node_interface import SensorNodeInterface


class AS5311AbsoluteSensorNode(QWidget, SensorNodeInterface):
    TITLE = "absolute"
    DESCRIPTION = ("Adjust signal intensity based on absolute sensor value from AS5311\r\n"
                   "\r\n"
                   "example: increase signal strength when clenching")

    def __init__(self, /):
        super().__init__()

        # setup UI
        self.verticalLayout = QVBoxLayout(self)
        self.groupbox = QGroupBox(self)
        self.groupbox.setTitle("Settings")
        self.verticalLayout.addWidget(self.groupbox)
        self.formLayout = QFormLayout(self.groupbox)

        self.spinbox_low = pg.SpinBox(None, 0.0, compactHeight=False, suffix='m', siPrefix=True, dec=True, minStep=1e-6)
        self.spinbox_high = pg.SpinBox(None, 0.0, compactHeight=False, suffix='m', siPrefix=True, dec=True, minStep=1e-6)
        self.spinbox_volume = pg.SpinBox(None, 0.0, compactHeight=False, suffix='%', step=0.1)
        self.spinbox_decay = pg.SpinBox(None, 1, compactHeight=False, suffix='samples', int=True, bounds=(1, 10000))

        self.spinbox_low.valueChanged.connect(self.update_lines)
        self.spinbox_high.valueChanged.connect(self.update_lines)

        # Connect to save settings on change
        self.spinbox_low.valueChanged.connect(self.save_settings)
        self.spinbox_high.valueChanged.connect(self.save_settings)
        self.spinbox_volume.valueChanged.connect(self.save_settings)
        self.spinbox_decay.valueChanged.connect(self.save_settings)

        self.formLayout.addRow('high threshold', self.spinbox_high)
        self.formLayout.addRow('low threshold', self.spinbox_low)
        label = QLabel('volume change (?)')
        label.setToolTip(
            "Increase volume when near 'high threshold'\r\n"
            "Reduce volume when near 'low threshold'\r\n"
            "Negative value to reverse")
        self.formLayout.addRow(label, self.spinbox_volume)
        self.formLayout.addRow('decay', self.spinbox_decay)

        self.graph = pg.GraphicsLayoutWidget()
        self.verticalLayout.addWidget(self.graph)

        # setup plots
        self.p1 = self.graph.addPlot()

        self.p1.setLabels(left=('Position', 'm'))

        self.p1.addLegend(offset=(30, 5))

        self.position_plot_item = pg.PlotDataItem(name='position')
        self.position_plot_item.setPen(pg.mkPen({'color': "blue", 'width': 1}))
        self.p1.addItem(self.position_plot_item)
        self.decayed_plot_item = pg.PlotDataItem(name='decayed')
        self.decayed_plot_item.setPen(pg.mkPen({'color': "orange", 'width': 1}))
        self.p1.addItem(self.decayed_plot_item)

        self.low_marker = pg.InfiniteLine(1, 0, movable=False)
        self.p1.addItem(self.low_marker)

        self.high_marker = pg.InfiniteLine(10, 0, movable=False)
        self.p1.addItem(self.high_marker)

        self.p1.setXRange(-10, 0, padding=0.05)

        # setup volume output plot (smaller, at bottom)
        self.graph.nextRow()
        self.p_volume = self.graph.addPlot()
        self.p_volume.setXLink(self.p1)
        self.p_volume.setLabels(left=('Volume', ''))
        self.p_volume.enableAutoRange(axis='y')
        self.p_volume.addLegend(offset=(30, 5))

        self.volume_plot_item = pg.PlotDataItem(name='adjustment')
        self.volume_plot_item.setPen(pg.mkPen({'color': "green", 'width': 2}))
        self.p_volume.addItem(self.volume_plot_item)

        # Set row stretch factors: main plot 4, volume plot 1 (makes volume 1/5 of total)
        self.graph.ci.layout.setRowStretchFactor(0, 4)
        self.graph.ci.layout.setRowStretchFactor(1, 1)

        # setup algorithm variables
        self.position = 0
        self.decayed_position = 0
        self.current_adjustment = 1.0

        # setup plot variables
        self.x = []
        self.y = []
        self.y_decay = []
        self.y_volume = []

        self.update_lines()

        # Load saved settings
        self.load_settings()

    def _get_settings_dict(self):
        return {
            'high_threshold': self.spinbox_high.value(),
            'low_threshold': self.spinbox_low.value(),
            'volume_change': self.spinbox_volume.value(),
            'decay': self.spinbox_decay.value(),
        }

    def _apply_settings_dict(self, settings):
        if 'high_threshold' in settings:
            self.spinbox_high.setValue(float(settings['high_threshold']))
        if 'low_threshold' in settings:
            self.spinbox_low.setValue(float(settings['low_threshold']))
        if 'volume_change' in settings:
            self.spinbox_volume.setValue(float(settings['volume_change']))
        if 'decay' in settings:
            self.spinbox_decay.setValue(int(float(settings['decay'])))

    def new_as5311_sensor_data(self, data: AS5311Data):
        if not self.is_node_enabled():
            return

        self.position = data.x
        self.decayed_position = max(self.decayed_position, self.position)
        self.decayed_position = self.decayed_position - (self.decayed_position - self.position) / self.spinbox_decay.value()

        # Calculate current adjustment for plotting
        self._calculate_adjustment()

        self.x.append(time.time())
        self.y.append(data.x)
        self.y_decay.append(self.decayed_position)
        self.y_volume.append(self.current_adjustment)

        threshold = time.time() - 10
        while len(self.x) and self.x[0] < threshold:
            del self.x[0]
            del self.y[0]
            del self.y_decay[0]
            del self.y_volume[0]

        self.update_graph_data()

    def _calculate_adjustment(self):
        """Calculate volume adjustment based on current decayed position."""
        xp = [self.spinbox_low.value(), self.spinbox_high.value()]
        if self.spinbox_volume.value() >= 0:
            yp = [1 - self.spinbox_volume.value() / 100, 1]
        else:
            yp = [1, 1 + self.spinbox_volume.value() / 100]
        # check sorting
        if xp[1] < xp[0]:
            xp = xp[::-1]
            yp = yp[::-1]
        self.current_adjustment = np.clip(np.interp(self.decayed_position, xp, yp), 0, 1)

    def process(self, parameters):
        if 'volume' in parameters:
            parameters['volume'] *= self.current_adjustment

    def update_graph_data(self):
        x = np.array(self.x) - self.x[-1]
        y = np.array(self.y)
        self.position_plot_item.setData(x=x, y=y)

        y2 = np.array(self.y_decay)
        self.decayed_plot_item.setData(x=x, y=y2)

        y_vol = np.array(self.y_volume)
        self.volume_plot_item.setData(x=x, y=y_vol)

    def update_lines(self, *args, **kwargs):
        self.low_marker.setValue(self.spinbox_low.value())
        self.high_marker.setValue(self.spinbox_high.value())

