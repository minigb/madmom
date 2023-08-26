# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains key recognition related functionality.

"""

import numpy as np

from ..processors import SequentialProcessor


KEY_LABELS = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
              'D major', 'Eb major', 'E major', 'F major', 'F# major',
              'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
              'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
              'F minor', 'F# minor', 'G minor', 'G# minor']

KEY_DICT = {
    0: 'major a',
    1: 'major a#',
    2: 'major b',
    3: 'major c',
    4: 'major c#',
    5: 'major d',
    6: 'major d#',
    7: 'major e',
    8: 'major f',
    9: 'major f#',
    10: 'major g',
    11: 'major g#',
    12: 'minor a',
    13: 'minor a#',
    14: 'minor b',
    15: 'minor c',
    16: 'minor c#',
    17: 'minor d',
    18: 'minor d#',
    19: 'minor e',
    20: 'minor f',
    21: 'minor f#',
    22: 'minor g',
    23: 'minor g#',
    -1: ""
}


def key_prediction_to_label(prediction):
    """
    Convert key class id to a human-readable key name.

    Parameters
    ----------
    prediction : numpy array
        Array containing the probabilities of each key class.

    Returns
    -------
    str
        Human-readable key name.

    """
    prediction = np.atleast_2d(prediction)
    key_index = prediction[0].argmax()
    return KEY_DICT[key_index]


def add_axis(x):
    return x[np.newaxis, ...]


class CNNKeyRecognitionProcessor(SequentialProcessor):
    """
    Recognise the global key of a musical piece using a Convolutional Neural
    Network as described in [1]_.

    Parameters
    ----------
    nn_files : list, optional
        List with trained CNN model files. Per default ('None'), an ensemble
        of networks will be used.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Genre-Agnostic Key Classification with Convolutional Neural
           Networks", In Proceedings of the 19th International Society for
           Music Information Retrieval Conference (ISMIR), Paris, France, 2018.

    Examples
    --------
    Create a CNNKeyRecognitionProcessor and pass a file through it.
    The returned array represents the probability of each key class.

    >>> proc = CNNKeyRecognitionProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.key.CNNKeyRecognitionProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +NORMALIZE_WHITESPACE
    array([[0.03426, 0.0331 , 0.02979, 0.04423, 0.04215, 0.0311 , 0.05225,
            0.04263, 0.04141, 0.02907, 0.03755, 0.09546, 0.0431 , 0.02792,
            0.02138, 0.05589, 0.03276, 0.02786, 0.02415, 0.04608, 0.05329,
            0.02804, 0.03868, 0.08786]])
    """

    def __init__(self, nn_files=None, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..ml.nn.activations import softmax
        from ..models import KEY_CNN

        # spectrogram computation
        sig = SignalProcessor(num_channels=1)
        frames = FramedSignalProcessor(frame_size=8192, fps=5)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_bands=24, fmin=65, fmax=2100, unique_filters=True
        )

        # neural network
        nn_files = nn_files or KEY_CNN
        nn = NeuralNetworkEnsemble.load(nn_files)

        # create processing pipeline
        super(CNNKeyRecognitionProcessor, self).__init__([
            sig, frames, stft, spec, nn, add_axis, softmax
        ])
