

import h5py
import logging
import hashlib
import numpy as np
import pandas as pd
import typing as tp

from tqdm import tqdm
from pathlib import Path

from neuralfetch import download
from neuralset import BaseExtractor
from neuralset.base import StrCast, Frequency, TimedArray
from neuralset.events import etypes, study

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

#: GitHub base URL for CNeuroMod datalad repositories.
_CNEUROMOD_GH_URL = "https://github.com/courtois-neuromod/{repo}.git"
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
    """Private base class for all CneuroMod studies.
    
    Handles the DataLad download of the BIDS and fMRIPrep repositories, the
    discovery of timelines from the fMRIPrep volumetric files, and the ``Fmri``
    event of each timeline.
    """

    #: Name of the raw BIDS DataLad repository (e.g. ``"movie10"``).
    BIDS_REPO: tp.ClassVar[str] = ""
    #: Name of the fMRIPrep derivatives DataLad repository (e.g. ``"movie10.fmriprep"``).
    FMRIPREP_REPO: tp.ClassVar[str] = ""
    # # Name of the timeseries DataLad repository (e.g. ``"movie10.timeseries"``).
    # TIMESERIES_REPO: tp.ClassVar[str] = ""

    url: tp.ClassVar[str] = "https://www.cneuromod.ca/"
    licence: tp.ClassVar[str] = (
        "CC0 (subjects 01, 02, 03, 05, 06) / Registered access — "
        "see https://www.cneuromod.ca/"
    )
    bibtex: tp.ClassVar[str] = """
    @article{boyle2020CCNposter,
        title={CNeuroMod, an open fMRI dataset with diverse naturalistic & 
        controlled tasks to build NeuroAI models},
        author={Boyle, Julie and Pinsard, Basile and St-Laurent, Marie 
        and Bellec, Lune},
        howpublished = {Poster presented at the OHBM 2026 Annual Meeting},
        address      = {Bordeaux, France},
        year={2026},
        month = {July},
    }
    @article{st2026cneuromod,
        title={CNeuroMod-THINGS, a densely-sampled fMRI dataset for visual neuroscience},
        author={St-Laurent, Marie and Pinsard, Basile and Contier, Oliver and DuPre, Elizabeth and Seeliger, Katja and Borghesani, Valentina and Boyle, Julie A and Bellec, Lune and Hebart, Martin N},
        journal={Scientific Data},
        volume={13},
        number={1},
        pages={141},
        year={2026},
        publisher={Nature Publishing Group UK London}
    }
    """
    description: tp.ClassVar[str] = (
        "Densely-sampled fMRI database collected on 6 subjects "
        "since 2018 across a broad range of cognitive tasks."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "pybids",
        "datalad",
        )


    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------
    
    def _repo_dir(self, repo: str) -> Path:
        """Local clone of a DataLad repository, as created by ``download.Datalad``."""
        return self.path / "download" / repo

    def _download_includes(self) -> dict[str, list[str]]:
        """Map each DataLad repository to the dataset-relative globs to fetch.

        Must be implemented by every study, for all the repositories it uses
        (BIDS, fMRIPrep, and e.g. stimuli / annotations): the layout differs
        between cneuromod datasets.
        Every repository named in a ``*_REPO`` class attribute must be a key, and
        every glob must match at least one file (both checked by :meth:`_download`).

        Note: the file names selected here are spelled out again by the loading
        code (:meth:`iter_timelines`, :meth:`_load_fmri_event`, the stimulus /
        transcript paths), so both must be kept in sync.
        """
        raise NotImplementedError

    def _clone_repo(self, repo: str, repo_url: str) -> Path:
        """Clone a DataLad repository into ``download/<repo>/``, unless already cloned.

        Only the git-annex pointers are cloned, not the file content.

        Parameters
        ----------
        repo : str
            Repository name, e.g. ``"movie10.fmriprep"``.
        repo_url : str
            Clone URL, e.g. ``"https://github.com/courtois-neuromod/movie10.fmriprep.git"``.

        Returns
        -------
        Path
            The local clone, ``{study path}/download/<repo>/``.
        """
        # This guards against the empty-selection bug raised in issue #282
        # that triggers whole-dataset download.
        # TODO: to be removed once upstream issue is fixed.
        import datalad.api as dlad

        sub_path = self._repo_dir(repo)
        if not sub_path.exists():
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            dlad.clone(source=repo_url, path=sub_path)

        return sub_path

    def _check_glob_matches(self, repo: str, repo_url: str, sub_path: Path, pattern: str) -> None:
        """Raise if an include glob matches no file of a cloned repository.

        Parameters
        ----------
        repo : str
            Repository name, e.g. ``"movie10.fmriprep"``.
        repo_url : str
            Clone URL of the repository.
        sub_path : Path
            Local clone of the repository (see :meth:`_resolve_subdir`).
        pattern : str
            Dataset-relative include glob, e.g. ``"task-*_events.tsv"``.

        Raises
        ------
        RuntimeError
            If ``pattern`` matches no file under ``sub_path``.
        """
        # This guards against the empty-selection bug raised in issue #282
        # that triggers whole-dataset download.
        # TODO: to be removed once upstream issue is fixed.

        # ``download.Datalad`` fetches the whole repository when its selection is
        # empty, so each glob is checked before the download starts. The check uses
        # the backend's own matcher (``Datalad._selected_paths``),
        # so it selects exactly the files the download would.
        
        matcher = download.Datalad(
                study=repo, dset_dir=self.path, repo_url=repo_url, include=[pattern]
            )
        if not matcher._selected_paths(sub_path):
            raise RuntimeError(
                f"No file of {sub_path} matches the include glob {pattern!r}; "
                "aborting rather than downloading the whole repository."
            )

    def _selection_hash(self, globs: list[str]) -> str:
        """Return a hash of the include globs."""
        
        # This guards against the failed wider selection bug raised in issue #282 
        # the success marker of download.Datalad is named after `study` and ignores `include`,
        # so a later, different selection would be silently skipped.
        # Name it after the selection instead; files already fetched are skipped by git-annex.
        # TODO: to be removed once upstream issue is fixed.
        return hashlib.sha1("\n".join(sorted(globs)).encode()).hexdigest()[:8]

    def _download(self, overwrite: bool = False) -> None:
        """Clone each DataLad repository and fetch only its selected files.

        Each repository is cloned on its own (not as a subdataset of
        ``cneuromod.all``), so the include globs of ``download.Datalad`` see all
        of its files.
        """

        includes = self._download_includes()
        repos = {
            getattr(self, name)
            for name in dir(type(self))
            if name.endswith("_REPO") and getattr(self, name)
        }
        missing = sorted(repos - includes.keys())
        if missing:
            raise ValueError(
                f"{type(self).__name__}._download_includes() does not cover the "
                f"repositories {missing}."
            )

        for repo, include in includes.items():
            # WORKAROUND 1: guard against the empty selection bug raised in issue #282
            repo_url = _CNEUROMOD_GH_URL.format(repo=repo)
            sub_path = self._clone_repo(repo, repo_url)

            for pattern in include:
                self._check_glob_matches(repo, repo_url, sub_path, pattern)
            
            # WORKAROUND 2: guard against failed wider selection raised in issue #282 
            selection = self._selection_hash(include)

            download.Datalad(
                study=f"{repo}-{selection}",
                dset_dir=self.path,
                repo_url=repo_url,
                include=include,
                threads=4,
            ).download(overwrite=overwrite)


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


