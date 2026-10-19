import json
import math
import logging
import hashlib
import pandas as pd
import typing as tp

from pathlib import Path

from neuralfetch import download
from neuralset.events import study
from neuralset.events import utils as event_utils

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

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class _CNeuroModStudy(study.Study):
    """Private base class for all CneuroMod studies.
    
    Handles the DataLad download of the BIDS and fMRIPrep repositories, the
    discovery of timelines from the fMRIPrep volumetric files, and the ``Fmri``
    event of each timeline. Subclasses set :attr:`BIDS_REPO` and
    :attr:`FMRIPREP_REPO`, implement :meth:`_download_includes`, and may
    override :meth:`_load_stimulus_events`.
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
            Local clone of the repository (see :meth:`_clone_repo`).
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

    # -----------------------------------------------------------------
    # Timelines
    # -----------------------------------------------------------------

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        fmriprep_dir = self._repo_dir(self.FMRIPREP_REPO)
        if not fmriprep_dir.exists():
            raise RuntimeError(
                f"fMRIPrep repository not found at {fmriprep_dir}. "
                "Run study.download() first."
            )
        
        pattern = f"sub-*/ses-*/func/*_space-{DEFAULT_SPACE}_desc-preproc_bold.nii.gz"
        for fp in sorted(fmriprep_dir.glob(pattern)):
            entities = dict(
                part.split("-", 1)
                for part in fp.name.split(f"_space-{DEFAULT_SPACE}")[0].split("_")
            )
            timeline = dict(
                subject=entities["sub"],
                session=entities["ses"],
                task=entities["task"],
            )
            
            if "run" in entities:
                timeline["run"] = entities["run"]
            yield timeline

    def _check_file_exists(self, filepath: Path) -> None:
        """Check if file datalad content is available locally."""
        if not filepath.resolve().exists():
                raise FileNotFoundError(
                    f"Content of {filepath} is not available locally. "
                    "Run study.download() or datalad get to fetch the content."
                )

    def _bids_stem(self, timeline: dict[str, tp.Any]) -> str:
        """BIDS file name stem of a timeline (e.g. ``sub-01_ses-001_task-bourne01``)."""
        tl = timeline
        stem = f"sub-{tl['subject']}_ses-{tl['session']}_task-{tl['task']}"
        if "run" in tl:
            stem += f"_run-{tl['run']}"
        return stem

    def _get_tr(self, filepath: Path) -> float:
        """Get the Repetition Time from a BOLD volume file."""
        with filepath.open() as f:
            tr = float(json.load(f)["RepetitionTime"])

        if not math.isclose(tr, DEFAULT_TR, rel_tol=1e-3):
            raise ValueError(
                f"Expected TR {DEFAULT_TR}s, got {tr}s for {filepath}"
            )
        return tr

    def _get_n_trs(self, filepath: Path) -> int:
        """Get the number of time points from a BOLD volume file."""
        import nibabel

        n_trs = nibabel.load(filepath).shape[-1]
        return n_trs
    
    def _load_fmri_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        
        func_dir = (
            self._repo_dir(self.FMRIPREP_REPO)
            / f"sub-{timeline['subject']}"
            / f"ses-{timeline['session']}"
            / "func"
        )
        
        stem = self._bids_stem(timeline)
        prefix = f"{stem}_space-{DEFAULT_SPACE}_desc"
        
        # import BOLD volume and mask
        bold_fp = func_dir / f"{prefix}-preproc_bold.nii.gz"
        mask_fp = func_dir / f"{prefix}-brain_mask.nii.gz"
        for fp in (bold_fp, mask_fp):
            self._check_file_exists(fp)
        
        n_trs = self._get_n_trs(bold_fp)
        tr = self._get_tr(func_dir / f"{prefix}-preproc_bold.json")
        
        events = event_utils.expand_bids_fmri(
            str(func_dir / f"{stem}_space-{DEFAULT_SPACE}_*"),
            preproc="fmriprep",
            start=0.0,
            frequency=1.0 / tr,
            duration=n_trs * tr,
        )
        if len(events) != 1 or "mask_filepath" not in events[0]:
            raise RuntimeError(
                f"Expected 1 {DEFAULT_SPACE} Fmri event with a brain mask for {stem}, "
                f"got {events}"
            )
        return events[0]

    def _load_stimulus_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load stimulus events of a timeline. Override in subclasses."""
        return pd.DataFrame()

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        fmri = pd.DataFrame([self._load_fmri_event(timeline)])
        stimuli = self._load_stimulus_events(timeline)
        if stimuli.empty:   
            return fmri
        return pd.concat([fmri, stimuli], ignore_index=True)


class _CNeuroModAudioStudy(_CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching and 
    audio-listening study fetchers, including Le Petit Prince, Narratives,
    Friends, Movie10 and OOD."""

    STIMULI_REPO: tp.ClassVar[str] = ""
    TRANSCRIPTS_REPO: tp.ClassVar[str] = ""

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """Stimulus file (e.g. audio ``.wav``, video ``.mkv``) presented during the timeline."""
        raise NotImplementedError

    def _get_stimulus_onset(self, timeline: dict[str, tp.Any]) -> float:
        """Onset of the stimulus, in seconds from the start of the run.

        The ``onset`` of the BIDS ``events.tsv``.
        """
        raise NotImplementedError

    def _get_transcript_path(self, timeline: dict[str, tp.Any]) -> Path:
        """Transcript (``.json``) of the stimulus presented during the timeline."""
        raise NotImplementedError

    def _get_transcript_language(self, timeline: dict[str, tp.Any]) -> str:
        """Language of the transcript (e.g. ``"english"``).

        A method rather than a constant, as it can change per run. 
        Override for non-English studies.
        """
        return "english"

    def _get_stimulus_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Stimulus event of the timeline: an ``Audio`` event (``.wav``).

        Duration and frequency are read from the file by the event.
        """
        return dict(
            type="Audio",
            start=self._get_stimulus_onset(timeline),
            filepath=str(self._get_stimulus_path(timeline)),
        )

    def _load_transcript_events(
        self, timeline: dict[str, tp.Any]
    ) -> list[dict[str, tp.Any]]:
        """``Word`` events and one ``Text`` event from the timeline's transcript.

        Transcript times are relative to the start of the stimulus, so they are
        shifted by the stimulus onset to be relative to the start of the run.
        """
        transcript_fp = self._get_transcript_path(timeline)
        if not transcript_fp.exists():
            logger.warning("No transcript found at %s", transcript_fp)
            return []
        with transcript_fp.open() as f:
            transcript = json.load(f)
        text_info = dict(
            language=self._get_transcript_language(timeline),
            modality="heard"
        )
        
        onset = self._get_stimulus_onset(timeline)
        words = transcript["words"]
        events: list[dict[str, tp.Any]] = [
            dict(
                type="Word",
                text=word["word"],
                start=onset + word["start"],
                duration=word["end"] - word["start"],
                **text_info,
            )
            for word in words
        ]
        if words:
            start, stop = words[0]["start"], words[-1]["end"]
            events.append(
                dict(
                    type="Text",
                    text=transcript["transcript"],
                    start=onset + start,
                    duration=stop - start,
                    **text_info,
                )
            )
        return events

    def _load_stimulus_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events = [self._get_stimulus_event(timeline)]
        events += self._load_transcript_events(timeline)
        return pd.DataFrame(events)

class _CNeuroModMovieStudy(_CNeuroModAudioStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching study
    fetchers, including Friends, Movie10 and OOD."""

    def _get_stimulus_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Stimulus event of the timeline: a ``Video`` event (``.mkv``).

        Duration and frequency are read from the file by the event.
        """
        return dict(
            type="Video",
            start=self._get_stimulus_onset(timeline),
            filepath=str(self._get_stimulus_path(timeline)),
        )

class _CNeuroModVideoGameStudy(_CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod videogame-playing study
    fetchers, including Shinobi, Mario, MarioStars and Mario3."""

