import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow
)

from qt_ui.main_window_ui import Ui_MainWindow
import qt_ui.websocket_client
import qt_ui.motion_generation


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.motion_generator = qt_ui.motion_generation.MotionGenerator(self)
        self.ws_client = qt_ui.websocket_client.WebsocketClient(self)

        self.tab_calibration.calibrationSettingsChanged.connect(self.ws_client.updateCalibrationParameters)
        self.tab_transform_calibration.transformCalibrationSettingsChanged.connect(self.ws_client.updateTransformParameters)
        self.tab_calibration.calibrationSettingsChanged.connect(self.tab_details.updateCalibrationParameters)
        self.tab_carrier.modulationSettingsChanged.connect(self.ws_client.updateModulationParameters)
        self.motion_generator.positionChanged.connect(self.ws_client.updatePositionParameters)

        self.motion_generator.positionChanged.connect(self.graphicsView.updatePositionParameters)
        self.motion_generator.positionChanged.connect(self.tab_details.updatePositionParameters)
        self.graphicsView.mousePositionChanged.connect(self.motion_generator.updateMousePosition)

        self.comboBox.currentTextChanged.connect(self.motion_generator.patternChanged)
        self.motion_generator.patternChanged(self.comboBox.currentText())
        self.doubleSpinBox.valueChanged.connect(self.motion_generator.velocityChanged)
        self.motion_generator.velocityChanged(self.doubleSpinBox.value())

        # trigger updates
        self.tab_calibration.settings_changed()
        self.tab_carrier.settings_changed()


def run():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()