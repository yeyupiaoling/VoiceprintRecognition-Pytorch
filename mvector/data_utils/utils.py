import gc
import io
import itertools

import av
import librosa
import numpy as np
import torch


def vad(wav, top_db=20, overlap=200):
    # Split an audio signal into non-silent intervals
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    wav_output = [np.array([])]
    for sliced in intervals:
        seg = wav[sliced[0]:sliced[1]]
        if len(seg) < 2 * overlap:
            wav_output[-1] = np.concatenate((wav_output[-1], seg))
        else:
            wav_output.append(seg)
    wav_output = [x for x in wav_output if len(x) > 0]

    if len(wav_output) == 1:
        wav_output = wav_output[0]
    else:
        wav_output = concatenate(wav_output)
    return wav_output


def concatenate(wave, overlap=200):
    total_len = sum([len(x) for x in wave])
    unfolded = np.zeros(total_len)

    # Equal power crossfade
    window = np.hanning(2 * overlap)
    fade_in = window[:overlap]
    fade_out = window[-overlap:]

    end = total_len
    for i in range(1, len(wave)):
        prev = wave[i - 1]
        curr = wave[i]

        if i == 1:
            end = len(prev)
            unfolded[:end] += prev

        max_idx = 0
        max_corr = 0
        pattern = prev[-overlap:]
        # slide the curr batch to match with the pattern of previous one
        for j in range(overlap):
            match = curr[j:j + overlap]
            corr = np.sum(pattern * match) / [(np.sqrt(np.sum(pattern ** 2)) * np.sqrt(np.sum(match ** 2))) + 1e-8]
            if corr > max_corr:
                max_idx = j
                max_corr = corr

        # Apply the gain to the overlap samples
        start = end - overlap
        unfolded[start:end] *= fade_out
        end = start + (len(curr) - max_idx)
        curr[max_idx:max_idx + overlap] *= fade_in
        unfolded[start:end] += curr[max_idx:]
    return unfolded[:end]


def decode_audio(file, sample_rate: int = 16000):
    """读取音频，主要用于兜底读取，支持各种数据格式

    Args:
      file: Path to the input file or a file-like object.
      sample_rate: Resample the audio to this sample rate.

    Returns:
      A float32 Numpy array.
    """
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(file, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    return audio.astype(np.float32) / 32768.0


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


# 将音频流转换为numpy
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
