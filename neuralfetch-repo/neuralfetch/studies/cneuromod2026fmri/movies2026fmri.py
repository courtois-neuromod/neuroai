
from __future__ import annotations

import json
import typing as tp
from pathlib import Path

from .base import _CNeuroModMovieStudy


class Movie10(_CNeuroModMovieStudy):
    """Courtois NeuroMod — *Movie10* movie-watching fMRI dataset.

    Six subjects watched three Hollywood feature films ("The Bourne Supremacy" (2004), 
    "The Wolf of Wall Street" (2013), "Hidden Figures" (2016)) and one BBC nature documentary 
    ("Life : Challenges of life, reptiles and amphibian mammals" (2009)) totalling ~10 hours of 
    movie watching while undergoing 3T fMRI.

    Hidden Figure and Life were both visioned twice to support reproducibility analyses.
    Each movie was split into multiple BOLD runs of ~10-minute each.

    References
    ----------
    * CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/movie10.html
    * DataLad BIDS repo: https://github.com/courtois-neuromod/movie10
    * DataLad fMRIPrep repo: https://github.com/courtois-neuromod/movie10.fmriprep
    * DataLad timeseries repo: https://github.com/courtois-neuromod/movie10.timeseries
    * DataLad stimuli repo: https://github.com/courtois-neuromod/movie10.stimuli
    * DataLad transcripts repo: https://github.com/courtois-neuromod/movie10.annotations

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/movie10/bids``,
        ``{path}/movie10/fmriprep`` or ``{path}/movie10/timeseries``,
        ``{path}/movie10/stimuli`` and ``{path}/movie10/annotations``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    timeseries:
        Define to model pre-extracted, masked, denoised and normalized timeseries, 
        rather than the fMRIPrep BOLD derivatives. Select among ``"cneuromod2026"``, 
        ``"schaefer1000"`` (algonauts 2025 competition), ``"voxel_mni"`` or ``"voxel_native"``.
        (default ``None``) .
    subjects:
        Restrict data loading to a subset of subject labels
        (without ``sub-`` prefix).  ``None`` includes all available subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Example
    --------
    >>> study = Movie10(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "movie10"
    BIDS_REPO: tp.ClassVar[str] = "movie10"
    FMRIPREP_REPO: tp.ClassVar[str] = "movie10.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "movie10.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "movie10.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "movie10.annotations"
    MOVIES: list[str] = ["bourne", "figures", "life", "wolf"]

    dataset_name: tp.ClassVar[str] = "CNeuroMod Movie10"
    description: tp.ClassVar[str] = (
        "Six subjects watching 10 hours of Hollywood movies / BBC documentary during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = _CNeuroModMovieStudy.bibtex


    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for movie files (.mkv).

        Only movies segmented for individual runs are targeted:

        * e.g., ``bourne01.mkv``

        Returns
        -------
        list[str]
            Glob patterns relative to the stimuli repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for mvie in self.MOVIES:
            patterns.extend([
                # Movie stimuli MKVs shown for this movie-watching task
                f"{self._stimuli_dir}/{mvie}/{mvie}*.mkv",
            ])
        return patterns

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for movie transcripts (.json).

        Transcripts for movies segmented for individual runs are targeted:

        * e.g., ``movie10_bourne01_model-AA_transcript.json``

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for mvie in self.MOVIES:
            patterns.extend([
                # Movie dialogues transcribed with AssemblyAI speech-to-text
                f"{self._annotations_dir}/annotations/transcripts/"
                f"{mvie}/movie10_{mvie}*_model-AA_transcript.json",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the segmented movie file (.mkv) shown during a 
        given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        Path
            The path to the segmented movie file shown during a given run.

        Raises
        ------
        FileNotFoundError
            If the .mkv file does not exist (DataLad content not fetched).
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        mp = Path(
            f"{self._stimuli_dir}/{seg_name[:-2]}"
            f"/{seg_name}.mkv",
        )
        if not mp.exists():
            raise FileNotFoundError(
                f"Movie file not found: {mp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return mp


    def _load_transcript(self, timeline: dict[str, tp.Any]) -> dict:
        """
        Load the speech-to-text transcript of the segmented movie 
        shown during a given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        dict
            The transcript for the segmented movie shown during a given run.
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        tp = Path(
            f"{self._annotations_dir}/annotations/transcripts/"
            f"{seg_name[:-2]}/movie10_{seg_name}_model-AA_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }
        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript


class Friends(_CNeuroModMovieStudy):
    """Courtois NeuroMod — *Friends* TV-watching fMRI dataset.

    Six subjects (01, 02, 03, 04, 05, 06) watched several seasons of the 
    Friends sitcom across multiple scanning sessions while while undergoing
    3T fMRI. One subject (04) watched seasons 1-4, while the remaining subjects
    watched seasons 1-6 (totalling >50 hours of audio-visual stimulation). 

    Most episodes are split into two ~12-minute BOLD runs (labelled a and b), and
    double episodes are split into up to four BOLD runs (a, b, c and d).  Stimuli
    are provided as video clip files (.mkv) in the ``stimuli`` repository.
    Corresponding movie transcripts are provided in the annotations repository.

    References
    ----------
    * CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/friends.html
    * DataLad BIDS repo: https://github.com/courtois-neuromod/friends
    * DataLad fMRIPrep repo: https://github.com/courtois-neuromod/friends.fmriprep
    * DataLad timeseries repo: https://github.com/courtois-neuromod/friends.timeseries
    * DataLad stimuli repo: https://github.com/courtois-neuromod/friends.stimuli
    * DataLad annotations repo: https://github.com/courtois-neuromod/friends.annotations

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/friends/bids``,
        ``{path}/friends/fmriprep`` or ``{path}/friends/timeseries``,
        ``{path}/friends/stimuli`` and ``{path}/friends/annotations``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    timeseries:
        Define to model pre-extracted, masked, denoised and normalized timeseries, 
        rather than the fMRIPrep BOLD derivatives. Select among ``"cneuromod2026"``, 
        ``"schaefer1000"`` (algonauts 2025 competition), ``"voxel_mni"`` or ``"voxel_native"``.
        (default ``None``) .
    subjects:
        Restrict data loading to a subset of subject labels
        (without ``sub-`` prefix).  ``None`` includes all available subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    * Stimuli (mkv video clips) are stored in the ``stimuli/`` repository.
    * Movie transcripts are stored in the ``annotations/`` repository.

    Examples
    --------
    >>> study = Friends(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "friends"
    BIDS_REPO: tp.ClassVar[str] = "friends"
    FMRIPREP_REPO: tp.ClassVar[str] = "friends.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "friends.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "friends.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "friends.annotations"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Friends"
    description: tp.ClassVar[str] = (
        "Six subjects watching Friends TV show seasons 1-6 (≈50 h) "
        "during 3T fMRI acquisition."
    )
    bibtex: tp.ClassVar[str] = _CNeuroModMovieStudy.bibtex


    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for movie files (.mkv).

        Only episodes segmented for individual runs are targeted:

        * e.g., ``friends_s01e01a.mkv``

        Returns
        -------
        list[str]
            Glob patterns relative to the stimuli repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return [
            f"{self._stimuli_dir}/s*/friends_s0*e*[abcd].mkv",
        ]

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for episode transcripts (.json).

        Transcripts for movies segmented for individual runs are targeted:

        * e.g., ``friends_s01e01a_model-AA_desc-wUtter_transcript.json``

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return [
            f"{self._annotations_dir}/automated_transcription/s*/"
            "friends_s0*e*_model-AA_desc-wUtter_transcript.json",  # TODO: update desc-wSpeaker
        ]

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the segmented movie file (.mkv) shown during a 
        given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        Path
            The path to the segmented movie file shown during a given run.

        Raises
        ------
        FileNotFoundError
            If the .mkv file does not exist (DataLad content not fetched).
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        mp = Path(
            f"{self._stimuli_dir}/s{seg_name[2]}"
            f"/friends_{seg_name}.mkv",
        )
        if not mp.exists():
            raise FileNotFoundError(
                f"Movie file not found: {mp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return mp


    def _load_transcript(self, timeline: dict[str, tp.Any]) -> dict:
        """
        Load the speech-to-text transcript of the segmented movie 
        shown during a given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        dict
            The transcript for the segmented episode shown during a given run.
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        tp = Path(  # TODO: adjust based on friends.annotations structure
            f"{self._annotations_dir}/annotations/automated_transcription/"
            f"s{seg_name[2]}/friends_{seg_name}_model-AA_desc-wUtter_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }
        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript