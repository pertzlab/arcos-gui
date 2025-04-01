"""Export image sequence from napari viewer."""

from napari_timestamper import render_as_rgb, save_image_stack

DEFAULT_EVENT_CMAP = "tab20"
DEFAULT_BIN_COLOR = "black"
DEFAULT_ALL_CELLS_COLOR = "gray"
# if TYPE_CHECKING:
# from arcos_gui.processing._data_storage import ArcosParameters, columnnames


class MovieExporter:
    """Export image sequence from napari viewer."""

    def __init__(self, viewer, outdir):
        self.viewer = viewer
        self.outdir = outdir

    def run(self, output_format, fps, scale_factor, output_name):
        """Run the exporter."""
        self._export_image_sequence(
            self.viewer, self.outdir, output_format, fps, scale_factor, output_name
        )

    def _export_image_sequence(
        self,
        viewer,
        outdir,
        output_format="png",
        fps=12,
        scale_factor=1,
        output_name="out",
    ):
        """Export a movie from a napari viewer.

        Parameters
        ----------
        viewer : napari.Viewer
            napari viewer object.
        outdir : str
            Path to output directory.
        fps : int
            Frames per second.
        scale_factor : float
            Scale factor for upscaling.
        output_format : str
            Output format for image sequence.
        """
        rgb = render_as_rgb(viewer, 0, upsample_factor=scale_factor)
        save_image_stack(
            image=rgb,
            directory=outdir,
            name=output_name,
            output_type=output_format,
            fps=fps,
        )
