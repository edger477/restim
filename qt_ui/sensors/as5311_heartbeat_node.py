import time
import numpy as np
from collections import deque

from PySide6.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QGroupBox, QLabel
from PySide6.QtCore import QTimer

import pyqtgraph as pg

from qt_ui.sensors.sensor_node_interface import SensorNodeInterface
from stim_math.sensors.as5311 import AS5311Data


class HeartbeatDetector:
    """
    Real-time heartbeat detector for AS5311 sensor data.
    Detects heartbeat from periodic oscillations in position data.

    Features:
    - BPM validation (40-180 BPM range)
    - Missing beat extrapolation during movement
    - Outlier rejection for stable BPM
    - Requires 3-4 consecutive valid beats
    """
    def __init__(self, window_sec=5):
        # Buffer to store recent data for baseline calculation
        self.window_sec = window_sec
        self.data_buffer = deque(maxlen=500)  # ~5-10 seconds at typical rates

        # State tracking for peak detection
        self.is_peak = False
        self.beat_times = deque(maxlen=15)  # Store more beats for better averaging
        self.valid_beat_times = deque(maxlen=15)  # Only validated beats
        self.last_beat_time = 0

        # Sampling rate estimation
        self.sample_times = deque(maxlen=50)
        self.estimated_sample_rate = 50.0  # Default assumption

        # Configurable sensitivity
        self.sensitivity = 0.5  # Multiplier for std deviation threshold

        # BPM constraints
        self.MIN_BPM = 40
        self.MAX_BPM = 180
        self.MIN_INTERVAL = 60.0 / self.MAX_BPM  # 0.333 seconds
        self.MAX_INTERVAL = 60.0 / self.MIN_BPM  # 1.5 seconds

        # Expected interval tracking for extrapolation
        self.expected_interval = None
        self.expected_interval_std = None

        # Outlier detection threshold (% deviation from expected)
        self.outlier_threshold = 0.30  # 30% deviation

        # Minimum beats for stable BPM
        self.min_beats_for_bpm = 4

    def update_sample_rate(self, current_time):
        """Estimate sampling rate from data arrival intervals"""
        self.sample_times.append(current_time)
        if len(self.sample_times) > 1:
            intervals = np.diff(list(self.sample_times))
            avg_interval = np.mean(intervals)
            if avg_interval > 0:
                self.estimated_sample_rate = 1.0 / avg_interval

    def is_valid_interval(self, interval):
        """Check if beat interval is within valid BPM range (40-180 BPM)"""
        return self.MIN_INTERVAL <= interval <= self.MAX_INTERVAL

    def is_outlier(self, interval):
        """Check if interval is an outlier compared to expected interval"""
        if self.expected_interval is None:
            return False

        deviation = abs(interval - self.expected_interval) / self.expected_interval
        return deviation > self.outlier_threshold

    def extrapolate_missing_beats(self, interval, current_time):
        """
        Extrapolate missing beats when interval is too long.
        Detects when sensor movement masks beats.
        """
        if self.expected_interval is None:
            return []

        missing_beats = []

        # Check if this interval is approximately N times the expected interval
        ratio = interval / self.expected_interval
        num_missed = round(ratio) - 1

        if 1 <= num_missed <= 3:  # We likely missed 1-3 beats
            # Synthesize missing beat times
            for i in range(1, num_missed + 1):
                synthetic_time = current_time - (interval * (1 - i / (num_missed + 1)))
                missing_beats.append(synthetic_time)

        return missing_beats

    def update_expected_interval(self):
        """Update expected interval from valid beats using median"""
        if len(self.valid_beat_times) >= 3:
            intervals = np.diff(list(self.valid_beat_times))
            self.expected_interval = np.median(intervals)
            self.expected_interval_std = np.std(intervals)

    def calculate_stable_bpm(self):
        """
        Calculate BPM using only valid, non-outlier beats.
        Requires minimum number of consecutive valid beats.
        """
        if len(self.valid_beat_times) < self.min_beats_for_bpm:
            return 0.0, 0.0

        # Get intervals from valid beats
        intervals = np.diff(list(self.valid_beat_times))

        # Use median interval for robustness against outliers
        median_interval = np.median(intervals)
        bpm = 60.0 / median_interval

        # Calculate confidence based on:
        # 1. Number of valid beats (more = better)
        # 2. Interval consistency (lower std = better)
        # 3. Percentage of accepted vs rejected beats

        interval_std = np.std(intervals)
        consistency = 1.0 - min(1.0, interval_std / median_interval)

        # More beats = higher confidence (caps at 10 beats)
        beat_count_factor = min(1.0, len(self.valid_beat_times) / 10.0)

        # Overall confidence
        confidence = consistency * beat_count_factor

        return bpm, confidence

    def process_sample(self, val, current_time):
        """
        Process a single sensor sample and return current BPM estimate.

        Args:
            val: Position value in meters
            current_time: Timestamp in seconds

        Returns:
            tuple: (bpm, confidence) where bpm is beats per minute,
                   confidence is 0-1 indicating signal quality
        """
        self.update_sample_rate(current_time)
        self.data_buffer.append(val)

        # Need at least 1 second of data to start
        if len(self.data_buffer) < max(10, int(self.estimated_sample_rate)):
            return 0.0, 0.0

        # Calculate local baseline (moving average)
        local_avg = np.mean(self.data_buffer)
        std_dev = np.std(self.data_buffer)

        # Dynamic thresholding
        # Heart beats create oscillations of ~0.02mm (0.00002m)
        threshold = local_avg + (std_dev * self.sensitivity)

        # State machine for peak detection
        if val > threshold and not self.is_peak:
            # Potential beat detected
            interval = current_time - self.last_beat_time

            # Basic validation: not too fast (debouncing)
            if interval > self.MIN_INTERVAL:
                self.is_peak = True
                self.beat_times.append(current_time)

                # Validate beat against BPM limits
                if self.is_valid_interval(interval):
                    # Check if it's an outlier
                    if not self.is_outlier(interval):
                        # Valid beat - add to valid beats
                        self.valid_beat_times.append(current_time)
                        self.update_expected_interval()
                    else:
                        # Outlier - might be movement interference
                        # Try to extrapolate missing beats
                        missing = self.extrapolate_missing_beats(interval, current_time)
                        if missing:
                            # Add extrapolated beats
                            for beat_time in missing:
                                self.valid_beat_times.append(beat_time)
                            # Add current beat too
                            self.valid_beat_times.append(current_time)
                            self.update_expected_interval()
                        # else: Skip this beat as invalid

                self.last_beat_time = current_time

        elif val < local_avg:
            # Signal returned to baseline
            self.is_peak = False

        # Calculate stable BPM from valid beats
        bpm, confidence = self.calculate_stable_bpm()

        return bpm, confidence


class AS5311HeartbeatNode(QWidget, SensorNodeInterface):
    TITLE = "heartbeat"
    DESCRIPTION = ("Detect heartbeat from AS5311 sensor movements\r\n"
                   "\r\n"
                   "Adjusts volume based on detected heart rate (BPM).\r\n"
                   "Average heartbeats create ~0.02mm oscillations in sensor position.\r\n"
                   "\r\n"
                   "Note: Requires stable sensor placement. User movement may interfere.")

    def __init__(self):
        super().__init__()

        # Setup UI
        self.verticalLayout = QVBoxLayout(self)
        self.groupbox = QGroupBox(self)
        self.groupbox.setTitle("Settings")
        self.verticalLayout.addWidget(self.groupbox)
        self.formLayout = QFormLayout(self.groupbox)

        # BPM threshold controls
        self.spinbox_high_bpm = pg.SpinBox(
            value=90, bounds=[30, 200], step=1,
            suffix=' BPM', compactHeight=False
        )
        self.spinbox_low_bpm = pg.SpinBox(
            value=65, bounds=[30, 200], step=1,
            suffix=' BPM', compactHeight=False
        )
        self.spinbox_volume = pg.SpinBox(
            value=0.0, bounds=[-100, 100], step=1,
            suffix='%', compactHeight=False
        )
        self.spinbox_sensitivity = pg.SpinBox(
            value=0.5, bounds=[0.1, 2.0], step=0.1,
            compactHeight=False, dec=True
        )

        self.spinbox_low_bpm.valueChanged.connect(self.update_lines)
        self.spinbox_high_bpm.valueChanged.connect(self.update_lines)
        self.spinbox_sensitivity.valueChanged.connect(self.update_sensitivity)

        self.formLayout.addRow('high BPM threshold', self.spinbox_high_bpm)
        self.formLayout.addRow('low BPM threshold', self.spinbox_low_bpm)

        volume_label = QLabel('volume change (?)')
        volume_label.setToolTip(
            "Increase volume when near 'high BPM threshold'\r\n"
            "Reduce volume when near 'low BPM threshold'\r\n"
            "Negative value to reverse"
        )
        self.formLayout.addRow(volume_label, self.spinbox_volume)

        sensitivity_label = QLabel('sensitivity (?)')
        sensitivity_label.setToolTip(
            "Detection sensitivity (0.1-2.0)\r\n"
            "Lower = more sensitive but more false positives\r\n"
            "Higher = less sensitive but more stable\r\n"
            "Default: 0.5"
        )
        self.formLayout.addRow(sensitivity_label, self.spinbox_sensitivity)

        # Current BPM display
        self.label_current_bpm = QLabel('--- BPM')
        self.label_current_bpm.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.formLayout.addRow('Current BPM:', self.label_current_bpm)

        # Confidence display
        self.label_confidence = QLabel('---')
        self.formLayout.addRow('Signal Quality:', self.label_confidence)

        # Setup graphs
        self.graph = pg.GraphicsLayoutWidget()
        self.verticalLayout.addWidget(self.graph)

        # Top plot: Raw position
        self.p1 = self.graph.addPlot()
        self.p1.setLabels(left=('Position', 'm'), bottom='Time (s)')
        self.p1.addLegend(offset=(30, 5))
        self.p1.setTitle("Raw Sensor Position")

        # Bottom plot: BPM over time
        self.graph.nextRow()
        self.p2 = self.graph.addPlot()
        self.p2.setXLink(self.p1)
        self.p2.setLabels(left='BPM', bottom='Time (s)')
        self.p2.addLegend(offset=(30, 5))
        self.p2.setTitle("Detected Heart Rate")

        # Position plot
        self.position_plot_item = pg.PlotDataItem(name='position')
        self.position_plot_item.setPen(pg.mkPen({'color': "blue", 'width': 1}))
        self.p1.addItem(self.position_plot_item)

        # Heartbeat markers (vertical lines when beat detected)
        self.beat_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color='r', width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        self.beat_marker.setVisible(False)
        self.p1.addItem(self.beat_marker)

        # BPM plot
        self.bpm_plot_item = pg.PlotDataItem(name='BPM')
        self.bpm_plot_item.setPen(pg.mkPen({'color': "green", 'width': 2}))
        self.p2.addItem(self.bpm_plot_item)

        # BPM threshold lines
        self.low_bpm_marker = pg.InfiniteLine(
            50, angle=0, movable=False,
            pen=pg.mkPen(color='y', width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        self.high_bpm_marker = pg.InfiniteLine(
            100, angle=0, movable=False,
            pen=pg.mkPen(color='r', width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        self.p2.addItem(self.low_bpm_marker)
        self.p2.addItem(self.high_bpm_marker)

        self.p1.setXRange(-10, 0, padding=0.05)
        self.p2.setYRange(30, 150, padding=0.05)

        # Setup algorithm
        self.detector = HeartbeatDetector(window_sec=5)
        self.current_bpm = 0.0
        self.current_confidence = 0.0

        # Setup plot data
        self.x = []
        self.y_position = []
        self.y_bpm = []

        # Beat flash timer
        self.beat_flash_timer = QTimer()
        self.beat_flash_timer.timeout.connect(self.hide_beat_marker)
        self.last_beat_shown = 0

        self.update_lines()
        self.update_sensitivity()

    def new_as5311_sensor_data(self, data: AS5311Data):
        if not self.is_node_enabled():
            return

        current_time = time.time()

        # Process sample through heartbeat detector
        bpm, confidence = self.detector.process_sample(data.x, current_time)
        self.current_bpm = bpm
        self.current_confidence = confidence

        # Update data buffers
        self.x.append(current_time)
        self.y_position.append(data.x)
        self.y_bpm.append(bpm if bpm > 0 else np.nan)  # Use NaN for no detection

        # Show beat marker flash when new beat detected
        if len(self.detector.beat_times) > 0:
            latest_beat = self.detector.beat_times[-1]
            if latest_beat != self.last_beat_shown:
                self.last_beat_shown = latest_beat
                self.show_beat_marker()

        # Keep last 10 seconds of data
        threshold = current_time - 10
        while len(self.x) and self.x[0] < threshold:
            del self.x[0]
            del self.y_position[0]
            del self.y_bpm[0]

        self.update_graph_data()
        self.update_labels()

    def process(self, parameters):
        """Modify volume based on detected heart rate"""
        if 'volume' in parameters and self.current_bpm > 0:
            xp = [self.spinbox_low_bpm.value(), self.spinbox_high_bpm.value()]

            if self.spinbox_volume.value() >= 0:
                yp = [1 - self.spinbox_volume.value() / 100, 1]
            else:
                yp = [1, 1 + self.spinbox_volume.value() / 100]

            adjustment = np.clip(np.interp(self.current_bpm, xp, yp), 0, 1)

            # Weight by confidence - low confidence reduces effect
            adjustment = 1.0 + (adjustment - 1.0) * self.current_confidence

            parameters['volume'] *= adjustment

    def update_graph_data(self):
        if len(self.x) == 0:
            return

        # Convert to relative time (seconds ago)
        x = np.array(self.x) - self.x[-1]

        # Update position plot
        y_pos = np.array(self.y_position)
        self.position_plot_item.setData(x=x, y=y_pos)

        # Update BPM plot
        y_bpm = np.array(self.y_bpm)
        self.bpm_plot_item.setData(x=x, y=y_bpm)

    def update_lines(self, *args, **kwargs):
        """Update threshold lines on BPM plot"""
        self.low_bpm_marker.setValue(self.spinbox_low_bpm.value())
        self.high_bpm_marker.setValue(self.spinbox_high_bpm.value())

    def update_sensitivity(self, *args, **kwargs):
        """Update detector sensitivity"""
        self.detector.sensitivity = self.spinbox_sensitivity.value()

    def update_labels(self):
        """Update current BPM and confidence display"""
        if self.current_bpm > 0:
            self.label_current_bpm.setText(f'{self.current_bpm:.1f} BPM')

            # Color code by BPM range
            if self.current_bpm < 60:
                color = '#3498db'  # Blue - low
            elif self.current_bpm < 100:
                color = '#2ecc71'  # Green - normal
            else:
                color = '#e74c3c'  # Red - high
            self.label_current_bpm.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {color};"
            )
        else:
            self.label_current_bpm.setText('--- BPM')
            self.label_current_bpm.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: gray;"
            )

        # Confidence display
        confidence_pct = self.current_confidence * 100
        if confidence_pct > 70:
            quality = "Good"
            color = "green"
        elif confidence_pct > 40:
            quality = "Fair"
            color = "orange"
        else:
            quality = "Poor"
            color = "red"

        self.label_confidence.setText(f'{quality} ({confidence_pct:.0f}%)')
        self.label_confidence.setStyleSheet(f"color: {color};")

    def show_beat_marker(self):
        """Show visual indicator when heartbeat detected"""
        self.beat_marker.setVisible(True)
        self.beat_marker.setValue(0)  # Current time
        self.beat_flash_timer.start(100)  # Hide after 100ms

    def hide_beat_marker(self):
        """Hide beat marker after flash"""
        self.beat_marker.setVisible(False)
        self.beat_flash_timer.stop()
