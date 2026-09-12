"""Interactive PyQt6 viewer for comparing 3D masks before and after post-processing.

Run from the repository root:
    python inferences/visualize_postprocessing_pyqt6.py

Choose two directories containing matching ``<case_id>.nii.gz`` masks.  The
viewer shows orthogonal slice overlays and reconstructed 3D surfaces for both
volumes.  Optional packages: ``PyQt6 pyqtgraph PyOpenGL scikit-image``.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from plotly.subplots import make_subplots
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import marching_cubes

from utils.metrics import binary_slice_metrics, binary_volume_metrics


DEFAULT_BEFORE_DIR = ROOT_DIR / "output" / "transunet_2.5d-3_balanced_sampling" / "test" / "predictions"
DEFAULT_AFTER_DIR = ROOT_DIR / "output" / "transunet_2.5d-3_balanced_sampling" / "test" / "post_processed" / "predictions"
DEFAULT_GROUND_TRUTH_ROOT = ROOT_DIR / "data" / "nsclc-radiomics" / "nifti"
PLOTLY_OUTPUT_DIR = ROOT_DIR / "output" / "plotly_viewer"


def find_volumes(directory: Path) -> dict[str, Path]:
    """Map case IDs to NIfTI paths in one prediction directory."""
    if not directory.is_dir():
        return {}
    return {
        path.name.removesuffix(".nii.gz"): path
        for path in sorted(directory.glob("*.nii.gz"))
    }


def find_ground_truths(nifti_root: Path) -> dict[str, Path]:
    """Map case IDs to ``<nifti_root>/<case_id>/mask.nii.gz`` reference masks."""
    if not nifti_root.is_dir():
        return {}
    return {
        case_dir.name: case_dir / "mask.nii.gz"
        for case_dir in nifti_root.iterdir()
        if case_dir.is_dir() and (case_dir / "mask.nii.gz").is_file()
    }


def normalize_gray(image: np.ndarray) -> np.ndarray:
    """Convert a mask slice to an easy-to-read uint8 grayscale background."""
    """
    if values.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low, high = np.percentile(values, (1, 99))
    if high <= low:Tô
        return np.zeros_like(values, dtype=np.uint8)
    """


def ct_mask_overlay(image_slice: np.ndarray, mask_slice: np.ndarray, color: tuple[int, int, int], opacity: float) -> QImage:
    """Overlay one colored binary mask on a windowed CT slice."""
    # NIfTI images in this project are usually pre-normalized to 0-255; retain
    # that contrast.  Fall back to a CT HU display window for raw CT volumes.
    values = np.asarray(image_slice, dtype=np.float32)
    if 0.0 <= float(values.min()) and float(values.max()) <= 255.0:
        gray = np.clip(values, 0, 255).astype(np.uint8)
    else:
        gray = np.clip((values + 1000.0) * 255.0 / 1400.0, 0, 255).astype(np.uint8)
    foreground = np.asarray(mask_slice, dtype=bool)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    rgb[foreground] = (1.0 - opacity) * rgb[foreground] + opacity * np.asarray(color, dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    height, width = foreground.shape
    # copy() detaches QImage from the short-lived NumPy array.
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class MaskLabel(QLabel):
    """A resize-aware QLabel that displays one binary mask slice."""

    def __init__(self, title: str, color: tuple[int, int, int]) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._title, self._color = title, color
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 280)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setText(title)

    def set_overlay(self, image_slice: np.ndarray, mask_slice: np.ndarray, opacity: float) -> None:
        self._image = ct_mask_overlay(image_slice, mask_slice, self._color, opacity)
        self._refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)


def mask_mesh(mask_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a binary ZYX volume to physical XYZ vertices and triangle faces."""
    padded = np.pad(np.asarray(mask_zyx, dtype=np.uint8), 1)
    vertices_zyx, faces, _, _ = marching_cubes(padded, level=0.5)
    return (vertices_zyx - 1.0)[:, ::-1] * np.asarray(spacing_xyz), faces


def mesh_trace(mask_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], color: str, name: str) -> go.Mesh3d | None:
    if not np.any(mask_zyx):
        return None
    vertices, faces = mask_mesh(mask_zyx, spacing_xyz)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=0.35, name=name, flatshading=True,
        hoverinfo="skip", showlegend=True,
    )


def build_plotly_comparison(before: np.ndarray, after: np.ndarray, target: np.ndarray, spacing_xyz: tuple[float, float, float]) -> str:
    """Create self-contained interactive Plotly HTML matching the reference style."""
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Before prediction + Ground Truth", "After prediction + Ground Truth"),
        horizontal_spacing=0.03,
    )
    for column, prediction in ((1, before), (2, after)):
        # Draw GT first, then prediction.  Their overlap becomes purple while
        # non-overlap remains distinctly red or blue.
        for trace in (mesh_trace(target, spacing_xyz, "red", "Ground Truth"), mesh_trace(prediction, spacing_xyz, "blue", "Prediction")):
            if trace is not None:
                figure.add_trace(trace, row=1, col=column)
    combined = np.logical_or(np.logical_or(before, after), target)
    vertices, _ = mask_mesh(combined, spacing_xyz)
    padding = np.maximum(np.ptp(vertices, axis=0) * 0.08, 5.0)
    axes = {
        "xaxis": {"title": "X (mm)", "range": [float(vertices[:, 0].min() - padding[0]), float(vertices[:, 0].max() + padding[0])], "showbackground": True, "backgroundcolor": "rgb(232,238,248)", "gridcolor": "white"},
        "yaxis": {"title": "Y (mm)", "range": [float(vertices[:, 1].min() - padding[1]), float(vertices[:, 1].max() + padding[1])], "showbackground": True, "backgroundcolor": "rgb(232,238,248)", "gridcolor": "white"},
        "zaxis": {"title": "Z (mm)", "range": [float(vertices[:, 2].min() - padding[2]), float(vertices[:, 2].max() + padding[2])], "showbackground": True, "backgroundcolor": "rgb(232,238,248)", "gridcolor": "white"},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.05}},
    }
    figure.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        legend={"orientation": "h", "y": 1.0, "x": 0.5, "xanchor": "center"},
        scene=axes, scene2=axes,
    )
    html = figure.to_html(include_plotlyjs=True, full_html=True, config={"responsive": True, "scrollZoom": True, "displaylogo": False})
    # Plotly emits ``scene.camera`` / ``scene2.camera`` whenever the user
    # rotates, pans, or zooms a 3D scene. Mirror that update to the sibling.
    sync_script = """
<script>
window.addEventListener('load', () => {
  const graph = document.querySelector('.plotly-graph-div');
  let syncing = false;
  graph.on('plotly_relayout', event => {
    if (syncing) return;
    let target = null, camera = null;
    if (event['scene.camera']) { target = 'scene2.camera'; camera = event['scene.camera']; }
    if (event['scene2.camera']) { target = 'scene.camera'; camera = event['scene2.camera']; }
    if (!target || !camera) return;
    syncing = true;
    Plotly.relayout(graph, {[target]: camera}).finally(() => { syncing = false; });
  });
});
</script>
"""
    return html.replace("</body>", sync_script + "</body>")


class PostprocessingViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Post-processing Viewer - Before / After")
        self.resize(1500, 960)
        self.before_volumes: dict[str, Path] = {}
        self.after_volumes: dict[str, Path] = {}
        self.ground_truths: dict[str, Path] = {}
        self.before: np.ndarray | None = None
        self.after: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.ct_image: np.ndarray | None = None
        self.spacing_xyz = (1.0, 1.0, 1.0)
        self._build_ui()
        self.before_path.setText(str(DEFAULT_BEFORE_DIR))
        self.after_path.setText(str(DEFAULT_AFTER_DIR))
        self.ground_truth_path.setText(str(DEFAULT_GROUND_TRUTH_ROOT))
        self.refresh_cases()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        sources = QGroupBox("Prediction folders")
        source_layout = QGridLayout(sources)
        self.before_path, self.after_path = QLineEdit(), QLineEdit()
        self.ground_truth_path = QLineEdit()
        for row, (label, field) in enumerate((("Before post-processing", self.before_path), ("After post-processing", self.after_path), ("Ground-truth NIfTI root", self.ground_truth_path))):
            source_layout.addWidget(QLabel(label), row, 0)
            source_layout.addWidget(field, row, 1)
            browse = QPushButton("Choose folder…")
            browse.clicked.connect(lambda _, target=field: self.choose_folder(target))
            source_layout.addWidget(browse, row, 2)
        reload_button = QPushButton("Load matching cases")
        reload_button.clicked.connect(self.refresh_cases)
        source_layout.addWidget(reload_button, 3, 2)
        layout.addWidget(sources)

        case_bar = QHBoxLayout()
        case_bar.addWidget(QLabel("Case:"))
        self.case_selector = QComboBox()
        self.case_selector.currentTextChanged.connect(self.load_case)
        case_bar.addWidget(self.case_selector, 1)
        self.stats = QLabel("Choose two folders containing NIfTI prediction masks.")
        self.stats.setWordWrap(True)
        case_bar.addWidget(self.stats, 3)
        layout.addLayout(case_bar)

        overlay_bar = QHBoxLayout()
        overlay_bar.addWidget(QLabel("2D overlay opacity"))
        self.overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity.setRange(10, 100)
        self.overlay_opacity.setValue(52)
        self.overlay_opacity.valueChanged.connect(lambda _: self.update_all_slice_views())
        overlay_bar.addWidget(self.overlay_opacity, 1)
        layout.addLayout(overlay_bar)

        self.tabs = QTabWidget()
        self.slice_labels: dict[str, tuple[MaskLabel, MaskLabel, MaskLabel]] = {}
        for plane, axis in (("Axial (Z)", 0), ("Coronal (Y)", 1), ("Sagittal (X)", 2)):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            images = QHBoxLayout()
            before_label = MaskLabel("Before", (220, 40, 40))
            after_label = MaskLabel("After", (25, 130, 240))
            target_label = MaskLabel("Ground truth", (20, 175, 75))
            images.addWidget(before_label)
            images.addWidget(after_label)
            images.addWidget(target_label)
            tab_layout.addLayout(images, 1)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda _, selected_axis=axis: self.update_slice_views(selected_axis))
            slider.setEnabled(False)
            slider.setProperty("axis", axis)
            tab_layout.addWidget(slider)
            tab_layout.addWidget(QLabel("Red = before post-processing; blue = after post-processing; green = ground truth."))
            self.tabs.addTab(tab, plane)
            self.slice_labels[str(axis)] = (before_label, after_label, target_label)
            setattr(self, f"slider_{axis}", slider)

        mesh_tab = QWidget()
        mesh_layout = QVBoxLayout(mesh_tab)
        mesh_layout.addWidget(QLabel("Left: Before prediction (blue) + Ground Truth (red). Right: After prediction (blue) + Ground Truth (red)."))
        self.plotly_view = QWebEngineView()
        self.plotly_view.setMinimumHeight(460)
        mesh_layout.addWidget(self.plotly_view, 1)
        self.tabs.addTab(mesh_tab, "3D overlay")
        self.tabs.currentChanged.connect(lambda _: self.update_slice_metric_table())
        layout.addWidget(self.tabs, 1)

        metrics_layout = QHBoxLayout()
        self.metric_table = self.make_metric_table("3D metric")
        self.slice_metric_table = self.make_metric_table("2D metric (selected slice)")
        metrics_layout.addWidget(self.metric_table)
        metrics_layout.addWidget(self.slice_metric_table)
        layout.addLayout(metrics_layout)

    def choose_folder(self, target: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select prediction folder", target.text() or str(ROOT_DIR))
        if selected:
            target.setText(selected)
            self.refresh_cases()

    def refresh_cases(self) -> None:
        self.before_volumes = find_volumes(Path(self.before_path.text()))
        self.after_volumes = find_volumes(Path(self.after_path.text()))
        self.ground_truths = find_ground_truths(Path(self.ground_truth_path.text()))
        case_ids = sorted(self.before_volumes.keys() & self.after_volumes.keys() & self.ground_truths.keys())
        self.case_selector.blockSignals(True)
        self.case_selector.clear()
        self.case_selector.addItems(case_ids)
        self.case_selector.blockSignals(False)
        if case_ids:
            self.case_selector.setCurrentIndex(0)
            self.load_case(case_ids[0])
        else:
            self.stats.setText("No matching case IDs were found across before, after, and ground-truth folders.")

    def load_case(self, case_id: str) -> None:
        if not case_id:
            return
        try:
            before_image = sitk.ReadImage(str(self.before_volumes[case_id]))
            after_image = sitk.ReadImage(str(self.after_volumes[case_id]))
            target_image = sitk.ReadImage(str(self.ground_truths[case_id]))
            ct_path = self.ground_truths[case_id].parent / "image.nii.gz"
            if not ct_path.is_file():
                raise FileNotFoundError(f"missing original CT volume: {ct_path}")
            self.before = sitk.GetArrayFromImage(before_image) > 0
            self.after = sitk.GetArrayFromImage(after_image) > 0
            self.target = sitk.GetArrayFromImage(target_image) > 0
            self.ct_image = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
            if self.before.shape != self.after.shape or self.before.shape != self.target.shape or self.before.shape != self.ct_image.shape:
                raise ValueError(f"different shapes: before {self.before.shape}, after {self.after.shape}, ground truth {self.target.shape}, CT {self.ct_image.shape}")
            self.spacing_xyz = tuple(float(value) for value in target_image.GetSpacing())
        except Exception as error:
            QMessageBox.critical(self, "Unable to load case", f"{case_id}: {error}")
            return
        for axis in range(3):
            slider: QSlider = getattr(self, f"slider_{axis}")
            slider.blockSignals(True)
            slider.setRange(0, self.before.shape[axis] - 1)
            slider.setValue(self.before.shape[axis] // 2)
            slider.setEnabled(True)
            slider.blockSignals(False)
            self.update_slice_views(axis)
        # QWebEngineView.setHtml() converts the document into a data: URL,
        # which Chromium limits to roughly 2 MB.  Mesh-heavy Plotly documents
        # are larger, so load a local HTML file instead.
        PLOTLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        html_path = PLOTLY_OUTPUT_DIR / f"{case_id}_comparison.html"
        html_path.write_text(build_plotly_comparison(self.before, self.after, self.target, self.spacing_xyz), encoding="utf-8")
        self.plotly_view.load(QUrl.fromLocalFile(str(html_path.resolve())))
        self.update_metric_table()
        self.update_slice_metric_table()
        voxel_mm3 = float(np.prod(self.spacing_xyz))
        self.stats.setText(
            f"{case_id} | shape Z×Y×X: {self.before.shape} | spacing X×Y×Z: "
            f"{self.spacing_xyz[0]:.3f} × {self.spacing_xyz[1]:.3f} × {self.spacing_xyz[2]:.3f} mm | "
            f"before: {self.before.sum():,} voxels ({self.before.sum() * voxel_mm3:.1f} mm³) | "
            f"after: {self.after.sum():,} voxels ({self.after.sum() * voxel_mm3:.1f} mm³) | "
            f"ground truth: {self.target.sum():,} voxels ({self.target.sum() * voxel_mm3:.1f} mm³)"
        )

    def update_metric_table(self) -> None:
        if self.before is None or self.after is None or self.target is None:
            return
        before_metrics = binary_volume_metrics(self.before, self.target, self.spacing_xyz)
        after_metrics = binary_volume_metrics(self.after, self.target, self.spacing_xyz)
        displayed = (("Dice", "dice"), ("IoU", "iou"), ("Recall", "recall"), ("Precision", "precision"), ("HD95 (mm)", "hd95"), ("ASSD (mm)", "assd"))
        self.metric_table.setRowCount(len(displayed))
        for row, (label, key) in enumerate(displayed):
            before_value, after_value = float(before_metrics[key]), float(after_metrics[key])
            values = (label, self.format_metric(before_value), self.format_metric(after_value), self.format_metric(after_value - before_value))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.metric_table.setItem(row, column, item)
        self.metric_table.resizeColumnsToContents()

    @staticmethod
    def make_metric_table(first_header: str) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels([first_header, "Before", "After", "Δ After - Before"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(225)
        return table

    def update_slice_metric_table(self) -> None:
        if self.before is None or self.after is None or self.target is None:
            return
        axis = self.tabs.currentIndex()
        if axis not in (0, 1, 2):
            axis = 0
        index = getattr(self, f"slider_{axis}").value()
        before_metrics = binary_slice_metrics(np.take(self.before, index, axis=axis), np.take(self.target, index, axis=axis))
        after_metrics = binary_slice_metrics(np.take(self.after, index, axis=axis), np.take(self.target, index, axis=axis))
        displayed = (("Dice", "dice"), ("IoU", "iou"), ("Recall", "recall"), ("Precision", "precision"), ("FP pixels", "fp"), ("FN pixels", "fn"))
        self.slice_metric_table.setRowCount(len(displayed))
        for row, (label, key) in enumerate(displayed):
            before_value, after_value = float(before_metrics[key]), float(after_metrics[key])
            values = (label, self.format_metric(before_value), self.format_metric(after_value), self.format_metric(after_value - before_value))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.slice_metric_table.setItem(row, column, item)
        self.slice_metric_table.resizeColumnsToContents()


    @staticmethod
    def format_metric(value: float) -> str:
        return "N/A" if not np.isfinite(value) else f"{value:.4f}"

    def update_slice_views(self, axis: int) -> None:
        if self.before is None or self.after is None or self.target is None or self.ct_image is None:
            return
        index = getattr(self, f"slider_{axis}").value()
        before_slice = np.take(self.before, index, axis=axis)
        after_slice = np.take(self.after, index, axis=axis)
        target_slice = np.take(self.target, index, axis=axis)
        ct_slice = np.take(self.ct_image, index, axis=axis)
        # Coronal/sagittal views are transposed so the displayed vertical axis is Z.
        if axis in (1, 2):
            before_slice, after_slice, target_slice, ct_slice = before_slice.T, after_slice.T, target_slice.T, ct_slice.T
        labels = self.slice_labels[str(axis)]
        opacity = self.overlay_opacity.value() / 100.0
        labels[0].set_overlay(ct_slice, before_slice, opacity)
        labels[1].set_overlay(ct_slice, after_slice, opacity)
        labels[2].set_overlay(ct_slice, target_slice, opacity)
        if axis == self.tabs.currentIndex():
            self.update_slice_metric_table()

    def update_all_slice_views(self) -> None:
        for axis in range(3):
            self.update_slice_views(axis)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PostprocessingViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
