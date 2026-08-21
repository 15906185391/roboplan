#!/usr/bin/env python3
"""PySide6 launcher for the RoboPlan Python examples."""

from __future__ import annotations

import json
import os
import signal
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
RUNNER = EXAMPLE_DIR / "example_runner.py"


def _cmeel_site_packages() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    return (
        purelib
        / "cmeel.prefix"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _available_models() -> tuple[str, ...]:
    try:
        if str(EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(EXAMPLE_DIR))
        cmeel_site = _cmeel_site_packages()
        if cmeel_site.exists() and str(cmeel_site) not in sys.path:
            sys.path.insert(0, str(cmeel_site))
        from common import get_model_data

        return tuple(get_model_data().keys())
    except Exception:
        return ("ur5", "franka", "dual", "kinova", "stretch", "so101")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    help_text: str = ""


@dataclass(frozen=True)
class ExampleSpec:
    module: str
    title: str
    category: str
    description: str
    interactive: bool
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)


MODEL_CHOICES = _available_models()
SUPPORTED_CONSTRAINED_MODELS = tuple(
    model for model in ("ur5", "franka", "kinova") if model in MODEL_CHOICES
) or ("ur5",)


COMMON_VISER_PARAMS = (
    ParameterSpec("host", "Viser host", "str", "localhost"),
    ParameterSpec("port", "Viser port", "str", "8000"),
)


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        "example_scene",
        "Scene basics",
        "Core",
        "Create a JointConfiguration, load a model scene, and print scene details.",
        False,
    ),
    ExampleSpec(
        "example_ik",
        "Simple IK",
        "IK",
        "Interactive inverse kinematics with draggable end-effector markers in Viser.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("max_iters", "Max iterations", "int", 100, 1, 10000, 10),
            ParameterSpec("step_size", "Step size", "float", 1.0, 0.001, 10.0, 0.05),
            ParameterSpec(
                "max_linear_error_norm",
                "Linear tolerance",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec(
                "max_angular_error_norm",
                "Angular tolerance",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec("check_collisions", "Check collisions", "bool", True),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_oink",
        "OInK tracking",
        "IK",
        "Optimal IK servo loop with draggable Viser targets and optional collision barriers.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("task_gain", "Task gain", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM damping", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec(
                "regularization", "Regularization", "float", 1e-3, 0.0, 1.0, 0.0001
            ),
            ParameterSpec("control_freq", "Control frequency", "float", 100.0, 1.0, 1000.0, 5.0),
            ParameterSpec(
                "reference_filter_tau",
                "Reference filter tau",
                "float",
                0.1,
                0.0,
                5.0,
                0.01,
            ),
            ParameterSpec("self_collision_num_pairs", "Self-collision pairs", "int", 0, 0, 200, 1),
            ParameterSpec("self_collision_d_min", "Self-collision d min", "float", 0.02, 0.0, 1.0, 0.005),
            ParameterSpec("self_collision_d_max", "Self-collision d max", "float", 0.25, 0.0, 2.0, 0.01),
            ParameterSpec("self_collision_gain", "Self-collision gain", "float", 1.0, 0.0, 100.0, 0.5),
            ParameterSpec("limit_acceleration", "Limit acceleration", "bool", False),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_oink_position_barriers",
        "OInK position barriers",
        "IK",
        "Optimal IK with Control Barrier Function position safety limits.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("task_gain", "Task gain", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM damping", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "Regularization", "float", 1e-3, 0.0, 1.0, 0.0001),
            ParameterSpec("control_freq", "Control frequency", "float", 400.0, 1.0, 1000.0, 10.0),
            ParameterSpec("barrier_gain", "Barrier gain", "float", 10.0, 0.0, 100.0, 0.5),
            ParameterSpec("barrier_size", "Barrier size", "float", 0.5, 0.05, 5.0, 0.05),
            ParameterSpec("safety_margin", "Safety margin", "float", 0.05, 0.0, 1.0, 0.005),
            ParameterSpec("max_position_error", "Max position error", "float", 0.15, 0.0, 2.0, 0.01),
            ParameterSpec("max_rotation_error", "Max rotation error", "float", 0.5, 0.0, 3.14, 0.05),
            ParameterSpec("reference_filter_tau", "Reference filter tau", "float", 0.1, 0.0, 5.0, 0.01),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_rrt",
        "RRT planner",
        "Planning",
        "Sample collision-free start/goal states, plan with RRT, time-parameterize, and animate.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("max_connection_distance", "Connection distance", "float", 3.0, 0.01, 20.0, 0.1),
            ParameterSpec("collision_check_step_size", "Collision step size", "float", 0.05, 0.001, 1.0, 0.005),
            ParameterSpec("collision_check_use_bisection", "Collision bisection", "bool", True),
            ParameterSpec("goal_biasing_probability", "Goal bias", "float", 0.15, 0.0, 1.0, 0.01),
            ParameterSpec("max_nodes", "Max nodes", "int", 1000, 1, 100000, 100),
            ParameterSpec("max_planning_time", "Planning time", "float", 2.0, 0.01, 120.0, 0.25),
            ParameterSpec("rrt_connect", "RRT-Connect", "bool", False),
            ParameterSpec("rrt_star", "RRT star", "bool", False),
            ParameterSpec("rewire_distance", "Rewire distance", "float", 5.0, 0.01, 20.0, 0.1),
            ParameterSpec("fast_return", "Fast return", "bool", True),
            ParameterSpec("include_shortcutting", "Shortcut path", "bool", False),
            ParameterSpec("max_shortcutting_iters", "Shortcut iterations", "int", 100, 1, 10000, 10),
            ParameterSpec(
                "toppra_mode",
                "TOPPRA mode",
                "choice",
                "Adaptive",
                choices=("Adaptive", "Hermite", "Cubic"),
            ),
            ParameterSpec("rng_seed", "Random seed", "optional_int", None, 0, 2**31 - 1, 1),
            ParameterSpec("include_obstacles", "Include obstacles", "bool", False),
            ParameterSpec("include_octrees", "Include octrees", "bool", False),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_toppra_joint_planning",
        "TOPPRA joint planning",
        "Planning",
        "Generate a smooth joint-space waypoint path and time-parameterize it with TOPPRA.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec(
                "toppra_mode",
                "TOPPRA mode",
                "choice",
                "Adaptive",
                choices=("Hermite", "Cubic", "Adaptive", "LinearBlend"),
            ),
            ParameterSpec("waypoint_count", "Waypoint count", "int", 6, 2, 50, 1),
            ParameterSpec("path_span", "Path span", "float", 0.45, 0.01, 5.0, 0.05),
            ParameterSpec("curvature_scale", "Curvature scale", "float", 0.25, 0.0, 1.0, 0.05),
            ParameterSpec("dt", "Sample dt", "float", 0.01, 0.001, 1.0, 0.001),
            ParameterSpec("velocity_scale", "Velocity scale", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("acceleration_scale", "Acceleration scale", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec(
                "max_adaptive_iterations",
                "Adaptive iterations",
                "int",
                10,
                1,
                100,
                1,
            ),
            ParameterSpec(
                "max_adaptive_step_size",
                "Adaptive step",
                "float",
                0.05,
                0.001,
                1.0,
                0.005,
            ),
            ParameterSpec(
                "max_blend_deviation",
                "Blend deviation",
                "float",
                0.01,
                0.0,
                1.0,
                0.001,
            ),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_constrained_rrt",
        "Constrained RRT",
        "Planning",
        "Plan paths that keep the gripper upright and inside a safe zone.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=SUPPORTED_CONSTRAINED_MODELS),
            ParameterSpec("max_tilt_degrees", "Max tilt degrees", "float", 5.0, 0.1, 45.0, 0.5),
            ParameterSpec("path_step_size", "Path step size", "float", 0.1, 0.001, 1.0, 0.01),
            ParameterSpec("max_connection_distance", "Connection distance", "float", 0.5, 0.01, 10.0, 0.05),
            ParameterSpec("max_nodes", "Max nodes", "int", 40000, 1, 200000, 1000),
            ParameterSpec("max_planning_time", "Planning time", "float", 1.0, 0.01, 120.0, 0.25),
            ParameterSpec("collision_check_step_size", "Collision step size", "float", 0.05, 0.001, 1.0, 0.005),
            ParameterSpec("rrt_star", "RRT star", "bool", True),
            ParameterSpec("rewire_distance", "Rewire distance", "float", 1.0, 0.01, 20.0, 0.1),
            ParameterSpec("rng_seed", "Random seed", "int", 1234, 0, 2**31 - 1, 1),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_cartesian_planning",
        "Cartesian planner",
        "Planning",
        "Plan and replay a lawnmower Cartesian path while plotting the generated joint trajectory.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec(
                "speed_mode",
                "Speed mode",
                "choice",
                "TimeOptimal",
                choices=("TimeOptimal", "Bounded"),
            ),
            ParameterSpec("max_linear_speed", "Max linear speed", "float", 0.1, 0.001, 5.0, 0.01),
            ParameterSpec("max_angular_speed", "Max angular speed", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("max_linear_acceleration", "Max linear acceleration", "float", 0.5, 0.001, 20.0, 0.05),
            ParameterSpec("max_angular_acceleration", "Max angular acceleration", "float", 2.5, 0.001, 50.0, 0.1),
            ParameterSpec("max_position_error", "Max position error", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("max_orientation_error", "Max orientation error", "float", 0.1, 0.0, 3.14, 0.01),
            ParameterSpec("velocity_scale", "Velocity scale", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("acceleration_scale", "Acceleration scale", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("dt", "Sample dt", "float", 0.01, 0.001, 1.0, 0.001),
            ParameterSpec("path_size", "Path size", "float", 0.15, 0.01, 2.0, 0.01),
            ParameterSpec("path_num_passes", "Path passes", "int", 5, 1, 100, 1),
            ParameterSpec("path_corner_radius", "Corner radius", "float", 0.0, 0.0, 1.0, 0.005),
            ParameterSpec("path_corner_arc_step_deg", "Corner arc step", "float", 1.0, 0.1, 45.0, 0.5),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_action_chunk_tracking",
        "Action chunk tracking",
        "Tracking",
        "Track mock learned-policy Cartesian or joint action chunks through OInK.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("action_space", "Action space", "choice", "cartesian", choices=("cartesian", "joint")),
            ParameterSpec("chunk_horizon", "Chunk horizon", "int", 6, 1, 100, 1),
            ParameterSpec("action_scale", "Action scale", "float", 1.0, 0.0, 10.0, 0.1),
            ParameterSpec("segment_time", "Segment time", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("control_freq", "Control frequency", "float", 100.0, 1.0, 1000.0, 5.0),
            ParameterSpec("task_gain", "Task gain", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM damping", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "Regularization", "float", 1e-6, 0.0, 1.0, 0.000001),
            ParameterSpec("limit_acceleration", "Limit acceleration", "bool", False),
            ParameterSpec("sleep", "Sleep while generating", "bool", False),
            ParameterSpec("playback_speed", "Playback speed", "float", 1.0, 0.01, 10.0, 0.1),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_teleop",
        "Keyboard teleop",
        "Tracking",
        "Teleoperate one or all end-effectors with keyboard commands in the terminal.",
        True,
        (
            ParameterSpec("model", "Robot model", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("device", "Device", "choice", "keyboard", choices=("keyboard",)),
            ParameterSpec("ee_mode", "EE mode", "choice", "first", choices=("first", "all")),
            ParameterSpec("control_freq", "Control frequency", "float", 50.0, 1.0, 1000.0, 5.0),
            ParameterSpec("linear_sensitivity", "Linear sensitivity", "float", 0.3, 0.001, 5.0, 0.05),
            ParameterSpec("angular_sensitivity", "Angular sensitivity", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("reference_filter_tau", "Reference filter tau", "float", 0.1, 0.0, 5.0, 0.01),
            ParameterSpec("task_gain", "Task gain", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM damping", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "Regularization", "float", 1e-6, 0.0, 1.0, 0.000001),
            ParameterSpec("config_task_gain", "Config task gain", "float", 1e-4, 0.0, 1.0, 0.0001),
            ParameterSpec("target_axes_length", "Target axes length", "float", 0.1, 0.001, 1.0, 0.01),
            ParameterSpec("target_axes_radius", "Target axes radius", "float", 0.005, 0.0001, 0.1, 0.001),
            *COMMON_VISER_PARAMS,
        ),
    ),
)


class ParameterEditor(QWidget):
    def __init__(self, spec: ExampleSpec, port: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.widgets: dict[str, QWidget] = {}
        self._build(port)

    def _build(self, port: int) -> None:
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for parameter in self.spec.parameters:
            widget = self._make_widget(parameter, port)
            self.widgets[parameter.name] = widget
            layout.addRow(parameter.label, widget)

    def _make_widget(self, parameter: ParameterSpec, port: int) -> QWidget:
        if parameter.kind == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(bool(parameter.default))
            return checkbox
        if parameter.kind == "choice":
            combo = QComboBox()
            for choice in parameter.choices:
                combo.addItem(str(choice), choice)
            default_index = combo.findData(parameter.default)
            if default_index >= 0:
                combo.setCurrentIndex(default_index)
            return combo
        if parameter.kind in {"int", "optional_int"}:
            spin = QSpinBox()
            minimum = -1 if parameter.kind == "optional_int" else int(parameter.minimum or 0)
            spin.setRange(minimum, int(parameter.maximum or 2**31 - 1))
            spin.setSingleStep(int(parameter.step or 1))
            spin.setSpecialValueText("None" if parameter.kind == "optional_int" else "")
            spin.setValue(spin.minimum() if parameter.default is None else int(parameter.default))
            return spin
        if parameter.kind == "float":
            spin = QDoubleSpinBox()
            spin.setRange(float(parameter.minimum if parameter.minimum is not None else -1e9), float(parameter.maximum if parameter.maximum is not None else 1e9))
            spin.setSingleStep(float(parameter.step or 0.1))
            spin.setDecimals(6)
            spin.setValue(float(parameter.default))
            return spin

        line = QLineEdit(str(port) if parameter.name == "port" else str(parameter.default))
        return line

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for parameter in self.spec.parameters:
            widget = self.widgets[parameter.name]
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentData()
            elif isinstance(widget, QSpinBox):
                if parameter.kind == "optional_int" and widget.value() == widget.minimum():
                    value = None
                else:
                    value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                continue
            values[parameter.name] = value
        return values

    def set_running(self, running: bool) -> None:
        for widget in self.widgets.values():
            widget.setEnabled(not running)


class ExamplePage(QWidget):
    def __init__(self, spec: ExampleSpec, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.process: QProcess | None = None
        self.editor = ParameterEditor(spec, 8000 + index)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(self.spec.title)
        title.setObjectName("PageTitle")
        description = QLabel(self.spec.description)
        description.setWordWrap(True)
        description.setObjectName("Description")

        layout.addWidget(title)
        layout.addWidget(description)

        controls = QFrame()
        controls.setObjectName("ParameterPanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.editor)
        layout.addWidget(controls)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_example)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_example)
        self.open_button = QPushButton("Open Viser")
        self.open_button.clicked.connect(self.open_viser)
        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(lambda: self.log.clear())
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.open_button)
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        self.status = QLabel("Ready.")
        self.status.setObjectName("Status")
        layout.addWidget(self.status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(10)
        self.log.setFont(fixed_font)
        layout.addWidget(self.log, 1)

    def start_example(self) -> None:
        if self.process is not None:
            return
        if not RUNNER.exists():
            QMessageBox.critical(self, "Missing runner", f"Could not find {RUNNER}")
            return

        params = self.editor.values()
        self.log.append(f"$ {sys.executable} {RUNNER.name} {self.spec.module}")
        self.log.append(f"params: {json.dumps(params, ensure_ascii=False)}\n")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [str(RUNNER), self.spec.module, "--params-json", json.dumps(params)]
        )
        process.setWorkingDirectory(str(EXAMPLE_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        current_pythonpath = env.value("PYTHONPATH")
        paths = [str(EXAMPLE_DIR)]
        cmeel_site = _cmeel_site_packages()
        if cmeel_site.exists():
            paths.append(str(cmeel_site))
        if current_pythonpath:
            paths.append(current_pythonpath)
        env.insert("PYTHONPATH", os.pathsep.join(paths))
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        self._set_running(True)
        process.start()

    def stop_example(self) -> None:
        if self.process is None:
            return
        self.status.setText("Stopping...")
        if sys.platform != "win32":
            pid = self.process.processId()
            if pid:
                os.kill(pid, signal.SIGINT)
        else:
            self.process.terminate()
        if not self.process.waitForFinished(2500):
            self.process.kill()

    def open_viser(self) -> None:
        params = self.editor.values()
        host = params.get("host", "localhost")
        port = params.get("port", "8000")
        QDesktopServices.openUrl(QUrl(f"http://{host}:{port}"))

    def close(self) -> bool:
        self.stop_example()
        return super().close()

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self._append_log(text)

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        self._append_log(text)

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "crashed" if exit_status == QProcess.CrashExit else "finished"
        self.log.append(f"\n[{self.spec.title} {status} with exit code {exit_code}]")
        self.process = None
        self._set_running(False)

    def _error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(f"Process error: {error.name}")

    def _set_running(self, running: bool) -> None:
        self.editor.set_running(running)
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status.setText("Running." if running else "Ready.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboPlan Example Workbench")
        self.resize(1180, 760)
        self.pages: list[ExamplePage] = []
        self._build()

    def _build(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        toolbar.addAction(quit_action)

        splitter = QSplitter()
        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(280)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().hide()

        for index, spec in enumerate(EXAMPLES):
            item = QListWidgetItem(f"{spec.category} / {spec.title}")
            item.setData(Qt.UserRole, index)
            self.list_widget.addItem(item)
            page = ExamplePage(spec, index)
            self.pages.append(page)
            self.tabs.addTab(page, spec.title)

        self.list_widget.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.list_widget.setCurrentRow(0)

        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def closeEvent(self, event) -> None:
        for page in self.pages:
            page.stop_example()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget {
            font-size: 13px;
        }
        QMainWindow, QListWidget, QTextEdit {
            background: #f7f8fa;
        }
        QListWidget {
            border: none;
            padding: 8px;
        }
        QListWidget::item {
            padding: 10px 8px;
            border-radius: 6px;
        }
        QListWidget::item:selected {
            background: #dce7f7;
            color: #1b3355;
        }
        #PageTitle {
            font-size: 22px;
            font-weight: 650;
        }
        #Description {
            color: #4a5361;
        }
        #Status {
            color: #596274;
        }
        #ParameterPanel {
            background: white;
            border: 1px solid #d9dee7;
            border-radius: 8px;
        }
        QPushButton {
            min-height: 30px;
            padding: 4px 12px;
        }
        QTextEdit {
            background: #101418;
            color: #d9e6f2;
            border: 1px solid #2d3640;
            border-radius: 8px;
        }
        """
    )
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
