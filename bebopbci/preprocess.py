"""
Simple loading through MOABB and pre-processing pipelines.
"""

import numpy as np

from os.path import join

import moabb
from moabb.datasets.base import BaseDataset
from moabb.paradigms import LeftRightImagery

import mne
from mne.io import read_raw_gdf

from autoreject import get_rejection_threshold

from help_funcs import dir_creation
from zapline_iter import zapline_until_gone

import warnings

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


class Dreyer2023(BaseDataset):
    """
    Conversion of all datasets described in Dreyer et al. 2023 in a single dataset
    class.

    By default only loads the all recording trials. Can also only load the acquisition
    or online feedback trials.

    Parameters
    ----------
    basepath: str
              Absolute path where data are stored.
    runs: str {"all", "acquisition", "online"}, optional
          String to select specific trials for the analysis.
          Defaults to all.

    References
    ----------
    .. [1] Dreyer Pauline, Roc Aline, Rimbert Sébastien, Pillette Léa, Lotte Fabien
           A large EEG database with users' profile information for motor imagery
           Brain-Computer Interface research.
           https://www.nature.com/articles/s41597-023-02445-z
           https://zenodo.org/records/8089820
    """

    def __init__(
        self,
        basepath,
        runs="all",
    ):
        super().__init__( 
            subjects=list(range(1,88)),
            sessions_per_subject=1,
            events=dict(left_hand=7, right_hand=8),
            code="Dreyer2023",
            # MI 0-8s, prepare 0-3s, break 8-9.5s
            # boundary effects
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1038/s41597-023-02445-z",
        )
        self.basepath=basepath
        self.runs=runs


    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        files = self.data_path(subject)

        out = {}
        for sess_ind, runlist in enumerate(files):
            sess_key = str(sess_ind)
            out[sess_key] = {}
            for run_ind, fname in enumerate(runlist):
                run_key = str(run_ind)
                raw = read_raw_gdf(fname, preload=True, eog=["EOG1", "EOG2", "EOG3"], misc=["EMGg", "EMGd"])
                stim = raw.annotations.description.astype(np.dtype("<10U"))
                stim[stim == "769"] = "left_hand"
                stim[stim == "770"] = "right_hand"
                raw.annotations.description = stim
                out[sess_key][run_key] = raw
                out[sess_key][run_key].set_montage("standard_1020")
        return out

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        if subject <= 60:
            dataset = "A"
        elif subject > 60 and subject <= 81:
            dataset = "B"
        elif subject > 81:
            dataset = "C"
        
        path = join(self.basepath, "BCI_Database/Signals/DATA_{}/{}{}/".format(dataset, dataset, subject))

        return self.local_data_path(dataset, path, subject)

    def local_data_path(self, dataset, path, subject):

        if self.runs == "acquisition":
            dir_files = [[path+"{}{}_R{}_acquisition.gdf".format(dataset, subject, run) for run in range(1,3)]]
        
        elif self.runs == "online":
            if subject != 59:
                dir_files = [[path+"{}{}_R{}_onlineT.gdf".format(dataset, subject, run) for run in range(3,7)]]
            elif subject == 59:
                dir_files = [[path+"{}{}_R{}_onlineT.gdf".format(dataset, subject, run) for run in range(3,5)]]

        elif self.runs == "all":
            dir_files_1 = [path+"{}{}_R{}_acquisition.gdf".format(dataset, subject, run) for run in range(1,3)]
            if subject != 59:
                dir_files_2 = [path+"{}{}_R{}_onlineT.gdf".format(dataset, subject, run) for run in range(3,7)]
            elif subject == 59:
                dir_files_2 = [path+"{}{}_R{}_onlineT.gdf".format(dataset, subject, run) for run in range(3,5)]
            dir_files = [dir_files_1 + dir_files_2]

        return dir_files


def load_sub(
    subject, dataset, tmin, tmax, baseline, savepath, channels=None, band_pass=[0, 120]
):
    """
    Load the data of a given subject.

    Load the data of a given subject in for a dataset via MOABB.
    The data are bandpass-filtered, and trimmed in the time period
    of interest.

    Parameters
    ----------
    subject: int
             Subject for analysis.
    dataset: MOABB object
             Dataset from the MOABB project for the analysis.
    tmin, tmax: float
                Start and end time of the epochs in seconds, relative to
                the time-locked event.
    baseline: two-element list or 1D array
              Start and end time of baseline period in seconds, relative to
              the time-locked event.
    savepath: str
              Parent directory that contains all results.
    channels: None or list, optional
              If set to none, return all availbale channels. If set to a list,
              this list is presumed to include the names of channels to keep.
              Defaults to "None".
    band_pass: two-element list or 1D array, optional
               Bandpass limits for performing fitlering while loading
               recordings via MOABB. Lowpass filtering when lower limit
               is set to 0.
               Defaults to [0, 120] Hz.

    Returns
    -------
    epochs : MNE epochs object or Numpy array
             The recordings corresponding to the subject and classes we are
             interested in.
    labels : numpy array
             Array of strings containing the labels for each trial in 'epochs'.
    meta : numpy array
           Meta-information corresponding to each trial in 'epochs'.
    """

    # Create the necessary directory if it doesn't already exist.
    dir_creation(subject, savepath)

    # Paradigm selection for data available via MOABB.
    # Band pass filtering and epoch timings according to the corresponding parameters.
    paradigm = LeftRightImagery(
        fmin=band_pass[0],
        fmax=band_pass[1],
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        channels=channels,
    )

    # Load data (always as MNE epochs in order to have access to the info structure).
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=[subject], return_epochs=True
    )

    return epochs, labels, meta


def apply_preprocessing(
    epochs,
    labels,
    meta,
    channels=None,
    rereference=False,
    zapit=False,
    noise_freq=None,
    noise_wins=[10, 5],
    reject_trials=True,
    return_epochs=False,
    return_power=False,
    power_band=[13, 30],
):
    """
    Apply a simple pre-processing.

    The data can optionally be cleaned by:
    1) apllying a common average reference,
    2) removing power line noise,
    3) dropping trials marked as 'bad' by the autoreject package
    (2 & 3 only considering a subset of "sensorimotor" channels).

    Optionally return the signal power by filtering in a narrow band and
    applying Hilbert transform.

    Parameters
    ----------
    epochs: MNE epochs object or Numpy array
            The recordings corresponding to the subject and classes of
            interest.
    labels: numpy array
            Array of strings containing the labels for each trial in 'epochs'.
    meta: numpy array
          Meta-information corresponding to each trial in 'epochs'.
    channels: None or list, optional
              If set to "None", return all availbale channels. If set to a list,
              this list is presumed to include the names of channels to keep.
              Defaults to "None".
    rereference: bool, optional
                 If set to "True" apply common average reference.
                 Defaults to "False".
    zapit: bool, optional
           If set to "True", iteratively remove a noise artifact from the raw
           signal. The frequency of the artifact is provided by 'noise_freq'.
           Defaults to "False".
    noise_freq: int or None, optional
                When set to "int", frequency containing power line noise, or
                equivalent artifact. Only considered if 'zapit' is "True".
                Defaults to "None".
    noise_wins: list, optional
                Window sizes for removing line noise.  Only considered if
                'zapit' is "True".
                Defaults to [10, 5].
    reject_trials: bool, optional
                   If set to "True", use the autoreject package in order
                   to get a global amplitude rejection threshold and reject
                   trials based on that.
                   Defaults to "True".
    return_epochs: bool, optional
                   Return an MNE epochs object or a numpy array.
                   Defaults to "False".
    return_power: bool, optional
                  Return bandpassed signal or signal power in a specific frequency
                  band according to the 'power_band' parameter, when 'return_epochs'
                  is set to "False".
                  Defaults to "False".
    power_band: two-element list or 1D array, optional
                Bandpass limits for performing fitlering when returning signal
                power in a frequency band. Only used when 'return_power' is set
                to "True".
                Defaults to [13, 30] for 'canonical' beta band.

    Returns
    -------
    epochs: MNE epochs object or Numpy array
            The recordings corresponding to the subject and classes we are
            interested in.
    labels: numpy array
            Array of strings containing the labels for each trial in 'epochs'.
    meta: numpy array
          Meta-information corresponding to each trial in 'epochs'.
    info: MNE info object
          Info corresponding to 'epochs'.
    ntrials: numpy array
             Array containing the number of trials per condition, before and
             after trial rejection.

    References
    ----------
    Alain de Cheveigné, ZapLine: A simple and effective method to remove power
    line artifacts, NeuroImage, Volume 207, 2020, 116356, ISSN 1053-8119,
    https://doi.org/10.1016/j.neuroimage.2019.116356.
    """

    # Number of trials per condition before and after trial rejection.
    ntrials = np.empty([1, 4])

    info = epochs.info
    meta = np.array(meta)
    ntrials[0, 0] = len(np.where(labels == np.unique(labels)[0])[0])
    ntrials[0, 1] = len(np.where(labels == np.unique(labels)[1])[0])

    # Optionally apply Common Average Reference.
    if rereference == True:
        epochs.set_eeg_reference()

    # Optional channel selection.
    if channels != None:
        epochs.pick_channels(channels)

    # Optional removal power line noise (or equivalent).
    if zapit == True:
        data = np.copy(epochs._data)
        data, _ = zapline_until_gone(
            data, noise_freq, info["sfreq"], win_sz=noise_wins[0], spot_sz=noise_wins[1]
        )
        epochs._data = data

    # Optional trial rejection with a global threshold.
    if reject_trials == True:
        reject = get_rejection_threshold(epochs, decim=2)
        epochs.drop_bad(reject=reject)
        dropped = []
        muscle_id = 0
        for tr, ep in enumerate(epochs.drop_log):
            if len(ep) != 0:
                # Keep track of muscle artifacts
                # (automatically rejected), and
                # adjust the ids of 'dropped' 
                # accordingly.
                if ep == ("BAD_muscle",):
                    muscle_id += 1
                else:
                    dropped.append(tr - muscle_id)

        # Avoid problematic recordings.
        for d, dl in enumerate(epochs.drop_log):
            if dl == ("TOO_SHORT",):
                labels = np.insert(labels, d, "null")
                meta = np.insert(meta, d, np.array(["null", "null", "null"]), axis=0)

        labels = np.delete(labels, dropped)
        meta = np.delete(meta, dropped, axis=0)
    ntrials[0, 2] = len(np.where(labels == np.unique(labels)[0])[0])
    ntrials[0, 3] = len(np.where(labels == np.unique(labels)[1])[0])

    # For ERD-ERS band-pass filter in the specified band (beta by default),
    # apply Hilbert transform and square.
    if return_epochs == True:
        if return_power == True:
            epochs.filter(power_band[0], power_band[1])
            epochs.apply_hilbert(envelope=True)
            data = epochs.get_data()
            data = data**2
            epochs._data = data
            print(
                "Returning MNE Epochs object containing Hilbert-transform envelope data in-place."
            )
        else:
            pass
    elif return_epochs == False:
        if return_power == True:
            epochs.filter(power_band[0], power_band[1])
            epochs.apply_hilbert(envelope=True)
            epochs = epochs.get_data()
            epochs = epochs**2
        else:
            epochs = epochs.get_data()

    return epochs, labels, meta, info, ntrials
