# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\designer\mediasettingswidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MediaSettingsWidget(object):
    def setupUi(self, MediaSettingsWidget):
        MediaSettingsWidget.setObjectName("MediaSettingsWidget")
        MediaSettingsWidget.resize(664, 477)
        self.verticalLayout = QtWidgets.QVBoxLayout(MediaSettingsWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(MediaSettingsWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(150, 0))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QFormLayout(self.frame)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.connection_status = QtWidgets.QLabel(self.widget)
        self.connection_status.setObjectName("connection_status")
        self.horizontalLayout.addWidget(self.connection_status)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.widget)
        self.widget_2 = QtWidgets.QWidget(self.frame)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.widget_2)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.widget_2)
        self.verticalLayout.addWidget(self.frame)
        self.stop_audio_automatically_checkbox = QtWidgets.QCheckBox(MediaSettingsWidget)
        self.stop_audio_automatically_checkbox.setChecked(True)
        self.stop_audio_automatically_checkbox.setObjectName("stop_audio_automatically_checkbox")
        self.verticalLayout.addWidget(self.stop_audio_automatically_checkbox)
        self.treeView = TreeViewWithComboBox(MediaSettingsWidget)
        self.treeView.setObjectName("treeView")
        self.verticalLayout.addWidget(self.treeView)
        self.widget_21 = QtWidgets.QWidget(MediaSettingsWidget)
        self.widget_21.setObjectName("widget_21")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_21)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.add_funscript_button = QtWidgets.QPushButton(self.widget_21)
        self.add_funscript_button.setObjectName("add_funscript_button")
        self.horizontalLayout_2.addWidget(self.add_funscript_button)
        self.bake_audio_button = QtWidgets.QPushButton(self.widget_21)
        self.bake_audio_button.setObjectName("bake_audio_button")
        self.horizontalLayout_2.addWidget(self.bake_audio_button)
        self.additional_search_paths_button = QtWidgets.QPushButton(self.widget_21)
        self.additional_search_paths_button.setObjectName("additional_search_paths_button")
        self.horizontalLayout_2.addWidget(self.additional_search_paths_button)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addWidget(self.widget_21)

        self.retranslateUi(MediaSettingsWidget)
        QtCore.QMetaObject.connectSlotsByName(MediaSettingsWidget)

    def retranslateUi(self, MediaSettingsWidget):
        _translate = QtCore.QCoreApplication.translate
        MediaSettingsWidget.setWindowTitle(_translate("MediaSettingsWidget", "Form"))
        self.label.setText(_translate("MediaSettingsWidget", "Media player"))
        self.connection_status.setText(_translate("MediaSettingsWidget", "TextLabel"))
        self.label_2.setText(_translate("MediaSettingsWidget", "File:"))
        self.stop_audio_automatically_checkbox.setText(_translate("MediaSettingsWidget", "Stop audio when file changes"))
        self.add_funscript_button.setText(_translate("MediaSettingsWidget", "Add funscript"))
        self.bake_audio_button.setText(_translate("MediaSettingsWidget", "Bake audio"))
        self.additional_search_paths_button.setText(_translate("MediaSettingsWidget", "Search paths"))
from qt_ui.widgets.table_view_with_combobox import TreeViewWithComboBox
import restim_rc
