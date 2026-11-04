from __future__ import annotations

import json
import typing as tp
import pandas as pd
from pathlib import Path

from .base import _CNeuroModMovieStudy, DEFAULT_SPACE


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
    * CNeuroMod documentation: https://docs.cneuromod.ca/datasets/movie10.html
    * DataLad BIDS repo: https://github.com/courtois-neuromod/movie10
    * DataLad fMRIPrep repo: https://github.com/courtois-neuromod/movie10.fmriprep
    * DataLad stimuli repo: https://github.com/courtois-neuromod/movie10.stimuli
    * DataLad transcripts repo: https://github.com/courtois-neuromod/movie10.annotations

    Experimental Design:
        - 3T fMRI recordings (TR = 1.49 seconds)
        - 6 participants (sub-01 to sub-06)
        - Paradigm: passive movie watching, in English
            * The Bourne Supremacy (``bourne``, 10 segments)
            * The Wolf of Wall Street (``wolf``, 17 segments)
            * Hidden Figures (``figures``, 12 segments, watched twice)
            * Life, disc one: "Challenges of life, reptiles and amphibian
              mammals" (``life``, 5 segments, watched twice)
        - The two viewings of repeated movies are labelled ``run-1`` and ``run-2``

    Data Format:
        - fMRIPrep derivatives, volumes in MNI152NLin2009cAsym space (2 mm),
          with the fMRIPrep brain mask of each run
        - 366 total timelines (6 participants x 61 runs)
        - Event types: Fmri, Video, Word, Text
        - Stimuli: movie segments (.mkv)
        - Words: AssemblyAI speech-to-text transcripts (the bourne09 segment has none)

    Surface projection:
        The study provides volumes; project them to a surface mesh with the
        extractor, e.g.
        ``FmriExtractor(projection=SurfaceProjector(mesh="fsaverage5"))``.

    Download Requirements:
        - datalad
    """

    BIDS_REPO: tp.ClassVar[str] = "movie10"
    FMRIPREP_REPO: tp.ClassVar[str] = "movie10.fmriprep"
    STIMULI_REPO: tp.ClassVar[str] = "movie10.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "movie10.annotations"

    description: tp.ClassVar[str] = (
        "Six subjects watching 10 hours of Hollywood movies / BBC documentary during 3T fMRI."
    )

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _download_includes(self) -> dict[str, list[str]]:
        return {
            # one events file per movie segment, at the BIDS root
            self.BIDS_REPO: ["task-*_events.tsv"],
            # MNI: preprocessed BOLD, its brain mask, and the BOLD JSON sidecar
            self.FMRIPREP_REPO: [
                f"sub-*/ses-*/func/*_space-{DEFAULT_SPACE}_desc-preproc_bold.nii.gz",
                f"sub-*/ses-*/func/*_space-{DEFAULT_SPACE}_desc-brain_mask.nii.gz",
                f"sub-*/ses-*/func/*_space-{DEFAULT_SPACE}_desc-preproc_bold.json",
            ],
            # one movie segment per run
            self.STIMULI_REPO: ["*/*.mkv"],
            # one transcript per movie segment, at the annotations root
            self.TRANSCRIPTS_REPO: [
                "annotations/transcripts/*/*_model-AA_transcript.json"
            ],
        }

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_bids_event(self, timeline: dict[str, tp.Any]) -> pd.Series:
        """
        Return the single row of the segment's BIDS events file (at the BIDS root).
        e.g. task-bourne01_events.tsv: onset=0, duration=603.07, stim_file=bourne/bourne01.mkv

        Raises
        ------
        RuntimeError: If the BIDS events file does not contain exactly one event.
        """
        events_fp = self._repo_dir(self.BIDS_REPO) / f"task-{timeline['task']}_events.tsv"
        events = pd.read_csv(events_fp, sep="\t")
        if len(events) != 1:
            raise RuntimeError(f"Expected one event in {events_fp}, got {len(events)}")
        return events.iloc[0]

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the segmented movie file (.mkv) shown during a
        given run ('timeline').
        """
        stimulus_fp = (
            self._repo_dir(self.STIMULI_REPO) / self._get_bids_event(timeline).stim_file
        )
        self._check_file_exists(stimulus_fp)
        return stimulus_fp

    def _get_transcript_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the transcript file (.json) for a given movie segment.
        """
        task = timeline["task"]  # e.g. bourne01
        movie = task.rstrip("0123456789")
        return (
            self._repo_dir(self.TRANSCRIPTS_REPO)
            / "annotations"
            / "transcripts"
            / movie
            / f"movie10_{task}_model-AA_transcript.json"
        )

    def _get_stimulus_onset(self, timeline: dict[str, tp.Any]) -> float:
        return float(self._get_bids_event(timeline).onset)


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
    * DataLad stimuli repo: https://github.com/courtois-neuromod/friends.stimuli
    * DataLad annotations repo: https://github.com/courtois-neuromod/friends.annotations

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/friends/bids``,
        ``{path}/friends/fmriprep``,
        ``{path}/friends/stimuli`` and ``{path}/friends/annotations``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
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

    Not implemented yet.
    """

    TASK: tp.ClassVar[str] = "friends"
    BIDS_REPO: tp.ClassVar[str] = "friends"
    FMRIPREP_REPO: tp.ClassVar[str] = "friends.fmriprep"
    STIMULI_REPO: tp.ClassVar[str] = "friends.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "friends.annotations"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Friends"
    description: tp.ClassVar[str] = (
        "Six subjects watching Friends TV show seasons 1-6 (≈50 h) "
        "during 3T fMRI acquisition."
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        raise NotImplementedError("Friends is not implemented yet.")
