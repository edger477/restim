from qt_ui.settings import get_settings_instance


class SensorNodeInterface:
    TITLE = ""
    DESCRIPTION = ""

    def __init__(self):
        super().__init__()
        self._node_enabled = False

    # implement these functions in your subclass:
    # def new_as5311_sensor_data(self, data: AS5311Data):
    # def new_imu_sensor_data(self, data: IMUData):
    # def new_pressure_sensor_data(self, data: PressureData):

    def enable_node(self):
        self._node_enabled = True
        self.save_settings()

    def disable_node(self):
        self._node_enabled = False
        self.save_settings()

    def is_node_enabled(self):
        return self._node_enabled

    def process(self, parameters):
        """
        :param parameters: a dict such as {'volume': 0.5, 'alpha': -0.4}
        :return:

        Modify the parameter dict in-place.

        For safety reasons, it is strongly recommended only to decrease the volume,
        and never increase it in this function.
        """
        pass

    def _get_settings_dict(self):
        """
        Override in subclass to return dict of settings to persist.
        Example: {'high_threshold': 0.001, 'low_threshold': 0.0005, ...}
        """
        return {}

    def _apply_settings_dict(self, settings):
        """
        Override in subclass to apply loaded settings to widgets.
        :param settings: dict of setting key -> value
        """
        pass

    def save_settings(self):
        """Save node settings to persistent storage."""
        s = get_settings_instance()
        s.beginGroup(f'sensor/{self.__class__.__name__}')
        s.setValue('enabled', self._node_enabled)
        for key, value in self._get_settings_dict().items():
            s.setValue(key, value)
        s.endGroup()

    def load_settings(self):
        """Load node settings from persistent storage."""
        s = get_settings_instance()
        s.beginGroup(f'sensor/{self.__class__.__name__}')
        self._node_enabled = s.value('enabled', False, bool)
        settings = {}
        for key in s.childKeys():
            if key != 'enabled':
                settings[key] = s.value(key)
        s.endGroup()
        self._apply_settings_dict(settings)

