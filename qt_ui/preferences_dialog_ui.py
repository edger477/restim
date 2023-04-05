# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\designer\preferencesdialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PreferencesDialog(object):
    def setupUi(self, PreferencesDialog):
        PreferencesDialog.setObjectName("PreferencesDialog")
        PreferencesDialog.resize(466, 429)
        self.verticalLayout = QtWidgets.QVBoxLayout(PreferencesDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(PreferencesDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gb_websocket_server = QtWidgets.QGroupBox(self.tab)
        self.gb_websocket_server.setCheckable(True)
        self.gb_websocket_server.setObjectName("gb_websocket_server")
        self.formLayout_4 = QtWidgets.QFormLayout(self.gb_websocket_server)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label = QtWidgets.QLabel(self.gb_websocket_server)
        self.label.setObjectName("label")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_6 = QtWidgets.QLabel(self.gb_websocket_server)
        self.label_6.setObjectName("label_6")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.websocket_localhost_only = QtWidgets.QCheckBox(self.gb_websocket_server)
        self.websocket_localhost_only.setText("")
        self.websocket_localhost_only.setObjectName("websocket_localhost_only")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.websocket_localhost_only)
        self.websocket_port = QtWidgets.QSpinBox(self.gb_websocket_server)
        self.websocket_port.setMaximum(65535)
        self.websocket_port.setObjectName("websocket_port")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.websocket_port)
        self.verticalLayout_2.addWidget(self.gb_websocket_server)
        self.gb_tcp_server = QtWidgets.QGroupBox(self.tab)
        self.gb_tcp_server.setCheckable(True)
        self.gb_tcp_server.setObjectName("gb_tcp_server")
        self.formLayout_2 = QtWidgets.QFormLayout(self.gb_tcp_server)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_3 = QtWidgets.QLabel(self.gb_tcp_server)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_10 = QtWidgets.QLabel(self.gb_tcp_server)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.tcp_localhost_only = QtWidgets.QCheckBox(self.gb_tcp_server)
        self.tcp_localhost_only.setText("")
        self.tcp_localhost_only.setObjectName("tcp_localhost_only")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.tcp_localhost_only)
        self.tcp_port = QtWidgets.QSpinBox(self.gb_tcp_server)
        self.tcp_port.setMaximum(65535)
        self.tcp_port.setObjectName("tcp_port")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.tcp_port)
        self.verticalLayout_2.addWidget(self.gb_tcp_server)
        self.gb_udp_server = QtWidgets.QGroupBox(self.tab)
        self.gb_udp_server.setFlat(False)
        self.gb_udp_server.setCheckable(True)
        self.gb_udp_server.setObjectName("gb_udp_server")
        self.formLayout = QtWidgets.QFormLayout(self.gb_udp_server)
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(self.gb_udp_server)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_11 = QtWidgets.QLabel(self.gb_udp_server)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.udp_localhost_only = QtWidgets.QCheckBox(self.gb_udp_server)
        self.udp_localhost_only.setText("")
        self.udp_localhost_only.setObjectName("udp_localhost_only")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.udp_localhost_only)
        self.udp_port = QtWidgets.QSpinBox(self.gb_udp_server)
        self.udp_port.setMaximum(65535)
        self.udp_port.setObjectName("udp_port")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.udp_port)
        self.verticalLayout_2.addWidget(self.gb_udp_server)
        self.gb_serial = QtWidgets.QGroupBox(self.tab)
        self.gb_serial.setCheckable(True)
        self.gb_serial.setObjectName("gb_serial")
        self.formLayout_6 = QtWidgets.QFormLayout(self.gb_serial)
        self.formLayout_6.setObjectName("formLayout_6")
        self.label_23 = QtWidgets.QLabel(self.gb_serial)
        self.label_23.setObjectName("label_23")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.label_24 = QtWidgets.QLabel(self.gb_serial)
        self.label_24.setObjectName("label_24")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.serial_auto_expand = QtWidgets.QCheckBox(self.gb_serial)
        self.serial_auto_expand.setText("")
        self.serial_auto_expand.setObjectName("serial_auto_expand")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.serial_auto_expand)
        self.serial_port = QtWidgets.QLineEdit(self.gb_serial)
        self.serial_port.setObjectName("serial_port")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.serial_port)
        self.verticalLayout_2.addWidget(self.gb_serial)
        self.label_12 = QtWidgets.QLabel(self.tab)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox.setObjectName("groupBox")
        self.formLayout_3 = QtWidgets.QFormLayout(self.groupBox)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.audio_api = QtWidgets.QComboBox(self.groupBox)
        self.audio_api.setObjectName("audio_api")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.audio_api)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.audio_device = QtWidgets.QComboBox(self.groupBox)
        self.audio_device.setObjectName("audio_device")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.audio_device)
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.audio_latency = QtWidgets.QComboBox(self.groupBox)
        self.audio_latency.setObjectName("audio_latency")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.audio_latency.addItem("")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.audio_latency)
        self.verticalLayout_3.addWidget(self.groupBox)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_5.setEnabled(True)
        self.groupBox_5.setObjectName("groupBox_5")
        self.formLayout_5 = QtWidgets.QFormLayout(self.groupBox_5)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setObjectName("label_2")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setObjectName("label_5")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.display_fps = QtWidgets.QSpinBox(self.groupBox_5)
        self.display_fps.setMaximum(1000)
        self.display_fps.setObjectName("display_fps")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.display_fps)
        self.display_latency_ms = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.display_latency_ms.setSuffix("")
        self.display_latency_ms.setMaximum(1000.0)
        self.display_latency_ms.setObjectName("display_latency_ms")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.display_latency_ms)
        self.verticalLayout_4.addWidget(self.groupBox_5)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 3, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 0, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox_2)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 3, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 0, 2, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 4, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 0, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 1, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox_2)
        self.label_21.setObjectName("label_21")
        self.gridLayout_2.addWidget(self.label_21, 4, 0, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.groupBox_2)
        self.label_22.setObjectName("label_22")
        self.gridLayout_2.addWidget(self.label_22, 5, 0, 1, 1)
        self.threephase_alpha_axis = QtWidgets.QLineEdit(self.groupBox_2)
        self.threephase_alpha_axis.setObjectName("threephase_alpha_axis")
        self.gridLayout_2.addWidget(self.threephase_alpha_axis, 1, 1, 1, 1)
        self.threephase_beta_axis = QtWidgets.QLineEdit(self.groupBox_2)
        self.threephase_beta_axis.setObjectName("threephase_beta_axis")
        self.gridLayout_2.addWidget(self.threephase_beta_axis, 2, 1, 1, 1)
        self.threephase_volume_axis = QtWidgets.QLineEdit(self.groupBox_2)
        self.threephase_volume_axis.setObjectName("threephase_volume_axis")
        self.gridLayout_2.addWidget(self.threephase_volume_axis, 3, 1, 1, 1)
        self.threephase_carrier_axis = QtWidgets.QLineEdit(self.groupBox_2)
        self.threephase_carrier_axis.setObjectName("threephase_carrier_axis")
        self.gridLayout_2.addWidget(self.threephase_carrier_axis, 4, 1, 1, 1)
        self.threephase_modulation_frequency_axis = QtWidgets.QLineEdit(self.groupBox_2)
        self.threephase_modulation_frequency_axis.setObjectName("threephase_modulation_frequency_axis")
        self.gridLayout_2.addWidget(self.threephase_modulation_frequency_axis, 5, 1, 1, 1)
        self.threephase_alpha_min = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_alpha_min.setMinimum(-100.0)
        self.threephase_alpha_min.setMaximum(100.0)
        self.threephase_alpha_min.setObjectName("threephase_alpha_min")
        self.gridLayout_2.addWidget(self.threephase_alpha_min, 1, 2, 1, 1)
        self.threephase_alpha_max = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_alpha_max.setMinimum(-100.0)
        self.threephase_alpha_max.setMaximum(100.0)
        self.threephase_alpha_max.setObjectName("threephase_alpha_max")
        self.gridLayout_2.addWidget(self.threephase_alpha_max, 1, 3, 1, 1)
        self.threephase_alpha_enabled = QtWidgets.QCheckBox(self.groupBox_2)
        self.threephase_alpha_enabled.setText("")
        self.threephase_alpha_enabled.setObjectName("threephase_alpha_enabled")
        self.gridLayout_2.addWidget(self.threephase_alpha_enabled, 1, 4, 1, 1)
        self.threephase_beta_min = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_beta_min.setMinimum(-100.0)
        self.threephase_beta_min.setMaximum(100.0)
        self.threephase_beta_min.setObjectName("threephase_beta_min")
        self.gridLayout_2.addWidget(self.threephase_beta_min, 2, 2, 1, 1)
        self.threephase_beta_max = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_beta_max.setMinimum(-100.0)
        self.threephase_beta_max.setMaximum(100.0)
        self.threephase_beta_max.setObjectName("threephase_beta_max")
        self.gridLayout_2.addWidget(self.threephase_beta_max, 2, 3, 1, 1)
        self.threephase_beta_enabled = QtWidgets.QCheckBox(self.groupBox_2)
        self.threephase_beta_enabled.setText("")
        self.threephase_beta_enabled.setObjectName("threephase_beta_enabled")
        self.gridLayout_2.addWidget(self.threephase_beta_enabled, 2, 4, 1, 1)
        self.threephase_volume_min = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_volume_min.setMaximum(1.0)
        self.threephase_volume_min.setObjectName("threephase_volume_min")
        self.gridLayout_2.addWidget(self.threephase_volume_min, 3, 2, 1, 1)
        self.threephase_volume_max = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_volume_max.setMaximum(1.0)
        self.threephase_volume_max.setObjectName("threephase_volume_max")
        self.gridLayout_2.addWidget(self.threephase_volume_max, 3, 3, 1, 1)
        self.threephase_volume_enabled = QtWidgets.QCheckBox(self.groupBox_2)
        self.threephase_volume_enabled.setText("")
        self.threephase_volume_enabled.setObjectName("threephase_volume_enabled")
        self.gridLayout_2.addWidget(self.threephase_volume_enabled, 3, 4, 1, 1)
        self.threephase_carrier_min = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_carrier_min.setMinimum(300.0)
        self.threephase_carrier_min.setMaximum(1500.0)
        self.threephase_carrier_min.setObjectName("threephase_carrier_min")
        self.gridLayout_2.addWidget(self.threephase_carrier_min, 4, 2, 1, 1)
        self.threephase_carrier_max = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_carrier_max.setMinimum(300.0)
        self.threephase_carrier_max.setMaximum(1500.0)
        self.threephase_carrier_max.setObjectName("threephase_carrier_max")
        self.gridLayout_2.addWidget(self.threephase_carrier_max, 4, 3, 1, 1)
        self.threephase_carrier_enabled = QtWidgets.QCheckBox(self.groupBox_2)
        self.threephase_carrier_enabled.setText("")
        self.threephase_carrier_enabled.setObjectName("threephase_carrier_enabled")
        self.gridLayout_2.addWidget(self.threephase_carrier_enabled, 4, 4, 1, 1)
        self.threephase_modulation_frequency_min = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_modulation_frequency_min.setMaximum(100.0)
        self.threephase_modulation_frequency_min.setObjectName("threephase_modulation_frequency_min")
        self.gridLayout_2.addWidget(self.threephase_modulation_frequency_min, 5, 2, 1, 1)
        self.threephase_modulation_frequency_max = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.threephase_modulation_frequency_max.setMaximum(100.0)
        self.threephase_modulation_frequency_max.setObjectName("threephase_modulation_frequency_max")
        self.gridLayout_2.addWidget(self.threephase_modulation_frequency_max, 5, 3, 1, 1)
        self.threephase_modulation_frequency_enabled = QtWidgets.QCheckBox(self.groupBox_2)
        self.threephase_modulation_frequency_enabled.setText("")
        self.threephase_modulation_frequency_enabled.setObjectName("threephase_modulation_frequency_enabled")
        self.gridLayout_2.addWidget(self.threephase_modulation_frequency_enabled, 5, 4, 1, 1)
        self.verticalLayout_5.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_6.addWidget(self.textEdit)
        self.verticalLayout_5.addWidget(self.groupBox_3)
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout.addWidget(self.tabWidget)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.buttonBox = QtWidgets.QDialogButtonBox(PreferencesDialog)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply|QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(PreferencesDialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(PreferencesDialog)
        PreferencesDialog.setTabOrder(self.udp_localhost_only, self.audio_latency)
        PreferencesDialog.setTabOrder(self.audio_latency, self.display_fps)
        PreferencesDialog.setTabOrder(self.display_fps, self.display_latency_ms)
        PreferencesDialog.setTabOrder(self.display_latency_ms, self.audio_api)
        PreferencesDialog.setTabOrder(self.audio_api, self.audio_device)
        PreferencesDialog.setTabOrder(self.audio_device, self.gb_websocket_server)
        PreferencesDialog.setTabOrder(self.gb_websocket_server, self.websocket_localhost_only)
        PreferencesDialog.setTabOrder(self.websocket_localhost_only, self.gb_tcp_server)
        PreferencesDialog.setTabOrder(self.gb_tcp_server, self.tcp_localhost_only)
        PreferencesDialog.setTabOrder(self.tcp_localhost_only, self.gb_udp_server)

    def retranslateUi(self, PreferencesDialog):
        _translate = QtCore.QCoreApplication.translate
        PreferencesDialog.setWindowTitle(_translate("PreferencesDialog", "Preferences"))
        self.gb_websocket_server.setTitle(_translate("PreferencesDialog", "Websocket server"))
        self.label.setText(_translate("PreferencesDialog", "Port"))
        self.label_6.setText(_translate("PreferencesDialog", "Localhost only"))
        self.gb_tcp_server.setTitle(_translate("PreferencesDialog", "TCP server"))
        self.label_3.setText(_translate("PreferencesDialog", "Port"))
        self.label_10.setText(_translate("PreferencesDialog", "Localhost only"))
        self.gb_udp_server.setTitle(_translate("PreferencesDialog", "UDP server"))
        self.label_4.setText(_translate("PreferencesDialog", "Port"))
        self.label_11.setText(_translate("PreferencesDialog", "Localhost only"))
        self.gb_serial.setTitle(_translate("PreferencesDialog", "Serial port"))
        self.label_23.setText(_translate("PreferencesDialog", "COM port"))
        self.label_24.setText(_translate("PreferencesDialog", "Auto-expand L0"))
        self.label_12.setText(_translate("PreferencesDialog", "Changes require restart"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("PreferencesDialog", "Network"))
        self.groupBox.setTitle(_translate("PreferencesDialog", "Audio"))
        self.label_7.setText(_translate("PreferencesDialog", "Audio API"))
        self.label_8.setText(_translate("PreferencesDialog", "Device"))
        self.label_9.setText(_translate("PreferencesDialog", "Latency"))
        self.audio_latency.setItemText(0, _translate("PreferencesDialog", "high"))
        self.audio_latency.setItemText(1, _translate("PreferencesDialog", "low"))
        self.audio_latency.setItemText(2, _translate("PreferencesDialog", "0.00"))
        self.audio_latency.setItemText(3, _translate("PreferencesDialog", "0.02"))
        self.audio_latency.setItemText(4, _translate("PreferencesDialog", "0.04"))
        self.audio_latency.setItemText(5, _translate("PreferencesDialog", "0.06"))
        self.audio_latency.setItemText(6, _translate("PreferencesDialog", "0.08"))
        self.audio_latency.setItemText(7, _translate("PreferencesDialog", "0.10"))
        self.audio_latency.setItemText(8, _translate("PreferencesDialog", "0.12"))
        self.audio_latency.setItemText(9, _translate("PreferencesDialog", "0.14"))
        self.audio_latency.setItemText(10, _translate("PreferencesDialog", "0.16"))
        self.audio_latency.setItemText(11, _translate("PreferencesDialog", "0.18"))
        self.audio_latency.setItemText(12, _translate("PreferencesDialog", "0.20"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("PreferencesDialog", "Audio"))
        self.groupBox_5.setTitle(_translate("PreferencesDialog", "Phase"))
        self.label_2.setText(_translate("PreferencesDialog", "max fps"))
        self.label_5.setText(_translate("PreferencesDialog", "display latency [ms]"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("PreferencesDialog", "Display"))
        self.groupBox_2.setTitle(_translate("PreferencesDialog", "TCode mapping"))
        self.label_14.setText(_translate("PreferencesDialog", "Max"))
        self.label_17.setText(_translate("PreferencesDialog", "Target"))
        self.label_20.setText(_translate("PreferencesDialog", "volume"))
        self.label_15.setText(_translate("PreferencesDialog", "Min"))
        self.label_19.setText(_translate("PreferencesDialog", "beta"))
        self.label_13.setText(_translate("PreferencesDialog", "Remote control?"))
        self.label_16.setText(_translate("PreferencesDialog", "Axis"))
        self.label_18.setText(_translate("PreferencesDialog", "alpha"))
        self.label_21.setText(_translate("PreferencesDialog", "carrier frequency"))
        self.label_22.setText(_translate("PreferencesDialog", "modulation 1 frequency"))
        self.groupBox_3.setTitle(_translate("PreferencesDialog", "Notes"))
        self.textEdit.setHtml(_translate("PreferencesDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Safety limits:</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Volume: 0 - 1</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Carrier freq: 300 - 1500</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Modulation freq: 0 - 100</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("PreferencesDialog", "Threephase"))
