
from tqdm import tqdm
import typing as tp

import h5py
import numpy as np
import pandas as pd

from neuralset import BaseExtractor
from neuralset.base import StrCast, Frequency, TimedArray
from neuralset.events import etypes, study


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

#: Default MNI152 template identifier used by fMRIPrep.
DEFAULT_SPACE = "MNI152NLin2009cAsym"
#: Default fMRI TR in seconds.
DEFAULT_TR = 1.49
#: Default resolution string used by fMRIPrep.
DEFAULT_RESOLUTION: str | None = "2"
#: Timeseries file name descriptor 
DEFAULT_TIMESERIES = "cneuromod2026"
TSERIES_DESCRIPT = {
    "cneuromod2026": "atlas-cneuromod26_desc-1134Parcels",
    "schaefer1000": "atlas-Schaefer18_desc-1000Parcels7Networks",
    "voxel_mni": "desc-voxelwise",
    "voxel_native": "desc-voxelwise",
}

# ---------------------------------------------------------------------------
# Support functions: TODO: import _utils functions here!
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class _CNeuroModStudy(study.Study):
    """Private base class for all CneuroMod studies."""


class _CNeuroModAudioStudy(_CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching and 
    audio-listening study fetchers, including Le Petit Prince, Narratives,
    Friends, Movie10 and OOD."""


class _CNeuroModMovieStudy(_CNeuroModAudioStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching study
    fetchers, including Friends, Movie10 and OOD."""

class _CNeuroModVideoGameStudy(_CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod videogame-playing study
    fetchers, including Shinobi, Mario, MarioStars and Mario3."""


# ---------------------------------------------------------------------------
# Timeseries event and extractor
# ---------------------------------------------------------------------------

class Timeseries(etypes.BaseSplittableEvent):
    """Pre-processed, masked, detrended and normalized functional MRI (fMRI) 
    recording event.

    Requires :code:`h5py` to be installed.

    Supports chunking via read() so chunks load only their own slice.

    Parameters
    ----------
    subject : str
        Subject identifier, e.g. ``"01"`` (required).
    filepath : Path or str
        Path to the .HDF5 file containing nested timeseries.     
    session : str
        Session identifier, e.g. ``"ses-001"`` (required). First-level key
        in .HDF5 file structure.
    run : str
        Run identifier, e.g. ``"ses-001_task-bourne01_timeseries"`` (required).
        Second-level key in .HDF5 file structure.
    frequency : float
        Sampling frequency in Hz (required).
    timeseries : str
        Timeseries format, e.g. ``"cneuromod2026"``, ``"schaefer1000"``,
        ``"voxel_mni"``, ``"voxel_native"``.
    space : str
        Coordinate space before timeseries extraction,
        e.g. ``"MNI152NLin2009cAsym"``, ``"T1w"``.
    """
    subject: StrCast
    session: str | None = None
    run: str | None = None
    timeseries: str = DEFAULT_TIMESERIES
    space: str = DEFAULT_SPACE

    def model_post_init(self, log__: tp.Any) -> None:
        if not self.frequency or pd.isna(self.frequency):
            raise ValueError(
                "Frequency must be provided for Timeseries event."
            )
        if not self.duration:
            raise ValueError(
                "Duration must be provided for Timeseries event."
            )
        if not self.session:
            raise ValueError(
                "Session must be provided for Timeseries event."
            )
        if not self.run:
            raise ValueError("Run must be provided for Timeseries event.")
        super().model_post_init(log__)

    def read(self) -> tp.Any:
        # If need be, crop based on specified offser and duration``.
        tseries = super().read()
        sr = Frequency(self.frequency)
        start_vol = sr.to_ind(self.offset)
        end_vol = start_vol + sr.to_ind(self.duration)
        if start_vol == 0 and end_vol >= tseries.shape[0]:
            return tseries
        return tseries[:, start_vol:end_vol]  # chunked

    def _read(self) -> tp.Any:
        with h5py.File(self.filepath, "r") as f:
            tseries = np.array(f[self.session][self.run]).T  # TimedArray last dim is time when freq > 0
        return tseries


class TimeseriesExtractor(BaseExtractor):
    """fMRI timeseries extraction (no caching).

    Input: an HDF5 file with fmri timeseries of shape [time, n_voxels/n_parcels] nested 
    per session and per run for each subject.

    Parameters
    ----------
    offset : float
        Seconds to shift TRs forward to align delayed BOLD response.
    frequency : ``"native"`` | float
        Target sampling frequency.
    """
    offset: float = 0.0
    event_types: tp.Literal["Timeseries"] = "Timeseries"
    aggregation: tp.Literal[
        "single",
        "sum",
        "mean",
        "first",
        "middle",
        "last",
        "cat",
        "stack",
        "trigger",
    ] = "single" 
    allow_missing: bool = False
    frequency: tp.Literal["native"] | float = "native"

    def _preprocess_event(self, event: Timeseries) -> TimedArray:
        """"""
        rec = event.read()
        header: dict[str, tp.Any] = {"timeseries": event.timeseries, "space": event.space}
        return TimedArray(
            data=rec.astype(np.float32),
            frequency=event.frequency,
            start=float("inf"),
            duration=event.duration,
            header=header,
        )
        
    def _get_data(self, events: list[Timeseries]) -> tp.Iterable[TimedArray]:
        """per-event computation (no caching)"""
        for event in tqdm(events, disable=len(events) < 2, desc="Processing timeseries data"):
            yield self._preprocess_event(event)

    def _get_timed_arrays(
        self, events: list[Timeseries], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        """return an iterable of :class:`~neuralset.base.TimedArray`, one per event"""
        for event, ta in zip(events, self._get_data(events)):
            out = ta.copy(start=event.start - self.offset)
            out = out.overlap(start, duration)
            yield out


