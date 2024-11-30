import torch
import torchaudio
import math
from typing import Optional, Union, Callable, List, Tuple, Sequence
# import interval_frequencies from librosa in 
from librosa.core.intervals import interval_frequencies
import numpy as np


fmin_def = 32.70319566257483  # C1 note frequency
def cqt(
    y: torch.Tensor,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: Optional[float] = 0.0,
    filter_scale: float = 1,
    norm: Optional[float] = 1,
    sparsity: float = 0.01,
    window: Union[str, Callable] = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    res_type: Optional[str] = "kaiser_best",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the constant-Q transform of an audio signal using PyTorch.
    
    This is a special case of VQT with gamma=0.
    """
    return vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        intervals="equal",
        gamma=0,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        window=window,
        scale=scale,
        pad_mode=pad_mode,
        res_type=res_type,
        dtype=dtype,
        device=device,
    )

def hybrid_cqt(
    y: torch.Tensor,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: Optional[float] = 0.0,
    filter_scale: float = 1,
    norm: Optional[float] = 1,
    sparsity: float = 0.01,
    window: Union[str, Callable] = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    res_type: str = "kaiser_best",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the hybrid constant-Q transform of an audio signal using PyTorch.
    """
    if device is None:
        device = y.device
        
    if dtype is None:
        dtype = torch.complex64

    if fmin is None:
        fmin = fmin_def  # C1 note frequency

    if tuning is None:
        #tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)  # Need to implement
        tuning = 0.0

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get all CQT frequencies
    freqs = cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    # Pre-compute alpha
    if n_bins == 1:
        alpha = _et_relative_bw(bins_per_octave)
    else:
        alpha = _relative_bandwidth(freqs=freqs)

    # Compute filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        filter_scale=filter_scale,
        window=window,
        alpha=alpha,
        device=device
    )

    # Determine which filters to use with Pseudo CQT
    pseudo_filters = (2.0 ** torch.ceil(torch.log2(lengths)) < 2 * hop_length)
    n_bins_pseudo = int(torch.sum(pseudo_filters).item())
    n_bins_full = n_bins - n_bins_pseudo
    
    cqt_resp = []

    if n_bins_pseudo > 0:
        fmin_pseudo = torch.min(freqs[pseudo_filters]).item()

        cqt_resp.append(
            pseudo_cqt(  # Need to implement
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin_pseudo,
                n_bins=n_bins_pseudo,
                bins_per_octave=bins_per_octave,
                filter_scale=filter_scale,
                norm=norm,
                sparsity=sparsity,
                window=window,
                scale=scale,
                pad_mode=pad_mode,
                dtype=dtype,
                device=device
            )
        )

    if n_bins_full > 0:
        cqt_resp.append(
            torch.abs(
                cqt(  # Need to implement
                    y,
                    sr=sr,
                    hop_length=hop_length,
                    fmin=fmin,
                    n_bins=n_bins_full,
                    bins_per_octave=bins_per_octave,
                    filter_scale=filter_scale,
                    norm=norm,
                    sparsity=sparsity,
                    window=window,
                    scale=scale,
                    pad_mode=pad_mode,
                    res_type=res_type,
                    dtype=dtype,
                    device=device
                )
            )
        )

    return _trim_stack(cqt_resp, n_bins, dtype)

def pseudo_cqt(
    y: torch.Tensor,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: Optional[float] = 0.0,
    filter_scale: float = 1,
    norm: Optional[float] = 1,
    sparsity: float = 0.01,
    window: Union[str, Callable] = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the pseudo constant-Q transform of an audio signal using PyTorch.
    """
    if device is None:
        device = y.device
        
    if fmin is None:
        fmin = fmin_def

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)  # Need to implement

    if dtype is None:
        dtype = torch.complex64 if y.dtype == torch.float32 else torch.complex64

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get frequencies
    freqs = cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        device=device
    )

    # Compute alpha
    if n_bins == 1:
        alpha = _et_relative_bw(bins_per_octave)
    else:
        alpha = _relative_bandwidth(freqs=freqs)

    # Get filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        alpha=alpha,
        device=device
    )

    # Get FFT basis
    fft_basis, n_fft, _ = _vqt_filter_fft(
        sr,
        freqs,
        filter_scale,
        norm,
        sparsity,
        hop_length=hop_length,
        window=window,
        dtype=dtype,
        alpha=alpha,
        device=device
    )

    # Take magnitude of FFT basis
    fft_basis = torch.abs(fft_basis)

    # Compute magnitude-only CQT response
    C = _cqt_response(
        y,
        n_fft,
        hop_length,
        fft_basis,
        pad_mode,
        window="hann",
        dtype=dtype,
        phase=False,
        device=device
    )

    if scale:
        C = C / torch.sqrt(torch.tensor(n_fft, dtype=C.dtype, device=device))
    else:
        # Reshape lengths to match dimensions
        lengths = _expand_to(lengths, ndim=C.dim(), axes=-2)
        C = C * torch.sqrt(lengths / n_fft)

    return C

def vqt(
    y: torch.Tensor,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    intervals: Union[str, Sequence[float]] = "equal",
    gamma: Optional[float] = None,
    bins_per_octave: int = 12,
    tuning: Optional[float] = 0.0,
    filter_scale: float = 1,
    norm: Optional[float] = 1,
    sparsity: float = 0.01,
    window: Union[str, Callable] = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    res_type: Optional[str] = "kaiser_best",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the variable-Q transform of an audio signal using PyTorch.
    """
    if device is None:
        device = y.device
        
    # Handle intervals
    if not isinstance(intervals, str):
        bins_per_octave = len(intervals)

    # Calculate octaves
    n_octaves = int(torch.ceil(torch.tensor(n_bins / bins_per_octave)).item())
    n_filters = min(bins_per_octave, n_bins)

    if fmin is None:
        fmin = fmin_def

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)  # Need to implement

    if dtype is None:
        dtype = torch.complex64 if y.dtype == torch.float32 else torch.complex64

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get frequencies
    # freqs = interval_frequencies(  # Need to implement
    #     n_bins=n_bins,
    #     fmin=fmin,
    #     intervals=intervals,
    #     bins_per_octave=bins_per_octave,
    #     sort=True,
    #     device=device
    # )
    # call the librosa one actually, converting any to numpy where necessary
    freqs_np = interval_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave, intervals=intervals)
    #conv to float
    freqs_np = freqs_np.astype(np.float32)
    freqs = torch.tensor(freqs_np, device=device)

    freqs_top = freqs[-bins_per_octave:]
    fmax_t = torch.max(freqs_top).item()

    # Compute alpha
    if n_bins == 1:
        alpha = _et_relative_bw(bins_per_octave)
    else:
        alpha = _relative_bandwidth(freqs=freqs)

    # Get filter lengths
    lengths, filter_cutoff = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
        device=device
    )

    # Check Nyquist frequency
    nyquist = sr / 2.0
    if filter_cutoff > nyquist:
        raise ValueError(
            f"Wavelet basis with max frequency={fmax_t} would exceed the Nyquist "
            f"frequency={nyquist}. Try reducing the number of frequency bins."
        )

    # Early downsampling
    y, sr, hop_length = _early_downsample(
        y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
    )

    vqt_resp = []
    my_y, my_sr, my_hop = y, sr, hop_length

    # Process each octave
    for i in range(n_octaves):
        # Get current octave slice
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)

        freqs_oct = freqs[sl]
        alpha_oct = alpha[sl]

        # Get FFT basis
        fft_basis, n_fft, _ = _vqt_filter_fft(
            my_sr,
            freqs_oct,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
            alpha=alpha_oct,
            device=device
        )

        # Rescale filters
        fft_basis *= torch.sqrt(torch.tensor(sr / my_sr, device=device))

        # Compute VQT response
        response = _cqt_response(
                my_y,
                n_fft,
                my_hop,
                fft_basis,
                pad_mode,
                dtype=dtype,
                device=device
            )

        vqt_resp.append(response)
        # Downsample if possible
        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            # my_y = resample(  # Need to implement
            #     my_y,
            #     orig_sr=2,
            #     target_sr=1,
            #     res_type=res_type,
            #     scale=True
            # )
            my_y = torchaudio.transforms.Resample(orig_freq=2,new_freq=1)(my_y)
            ratio = 1.0/2.0
            my_y = my_y / torch.sqrt(torch.tensor(ratio, device=y.device))

    # Stack and trim responses
    V = _trim_stack(vqt_resp, n_bins, dtype)

    if scale:
        # Recompute lengths for scaling
        lengths, _ = wavelet_lengths(
            freqs=freqs,
            sr=sr,
            window=window,
            filter_scale=filter_scale,
            gamma=gamma,
            alpha=alpha,
            device=device
        )

        # Reshape lengths and apply scaling
        lengths = _expand_to(lengths, ndim=V.dim(), axes=-2)
        V = V / torch.sqrt(lengths)

    return V

def _early_downsample(
    y: torch.Tensor,
    sr: float,
    hop_length: int,
    res_type: str,
    n_octaves: int,
    nyquist: float,
    filter_cutoff: float,
    scale: bool
) -> Tuple[torch.Tensor, float, int]:
    """
    Early downsampling stage for VQT computation.
    """
    if not scale:
        return y, sr, hop_length

    # Determine target downsampling rate
    downsample_count1 = max(0, int(np.ceil(np.log2(nyquist / filter_cutoff)) - 1))
    downsample_count2 = max(0, int(np.ceil(np.log2(hop_length))) - 1)
    downsample_count = min(downsample_count1, downsample_count2)

    if downsample_count > 0:
        downsample_factor = 2 ** (downsample_count)
        
        # Resample the signal
        # y = resample(
        #     y,
        #     orig_sr=sr,
        #     target_sr=sr / downsample_factor,
        #     res_type=res_type,
        #     scale=True
        # )
        y = torchaudio.transforms.Resample(orig_freq=downsample_factor, new_freq=1)(y)

        
        sr = sr / downsample_factor
        hop_length = hop_length // downsample_factor

    return y, sr, hop_length

def _expand_to(
    x: torch.Tensor,
    ndim: int,
    axes: Union[int, Sequence[int]]
) -> torch.Tensor:
    """
    Expand tensor dimensions to match target dimensionality.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    ndim : int
        Target number of dimensions
    axes : int or sequence of ints
        Axes along which to expand
        
    Returns
    -------
    x_exp : torch.Tensor
        Expanded tensor
    """
    if isinstance(axes, int):
        axes = [axes]
        
    shape = [1] * ndim
    for ax in axes:
        shape[ax] = x.size(0)
    
    return x.view(*shape)

def _cqt_response(
    y: torch.Tensor,
    n_fft: int,
    hop_length: int,
    fft_basis: torch.Tensor,
    pad_mode: str = "constant",
    window: str = "ones",
    dtype: Optional[torch.dtype] = None,
    phase: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the CQT response using STFT and provided filter basis.
    """
    if device is None:
        device = y.device
        
    # Compute STFT
    D = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=get_window(window, n_fft, device),
        center=True,
        pad_mode=pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True
    )

    # numpy code:
        # Reshape D to Dr
    # Dr = D.reshape((-1, D.shape[-2], D.shape[-1]))
    # output_flat = np.empty(
    #     (Dr.shape[0], fft_basis.shape[0], Dr.shape[-1]), dtype=D.dtype
    # )

    # # iterate over channels
    # #   project fft_basis.dot(Dr[i])
    # for i in range(Dr.shape[0]):
    #     output_flat[i] = fft_basis.dot(Dr[i])

    # # reshape Dr to match D's leading dimensions again
    # shape = list(D.shape)
    # shape[-2] = fft_basis.shape[0]
    # return output_flat.reshape(shape)

    # # pytorch version:
    # # Reshape D to Dr
    # Dr = D.reshape((-1, D.shape[-2], D.shape[-1]))
    # output_flat = torch.empty(
    #     (Dr.shape[0], fft_basis.shape[0], Dr.shape[-1]), dtype=D.dtype, device=device
    # )

    # # iterate over channels
    # #   project fft_basis.dot(Dr[i])
    # for i in range(Dr.shape[0]):
    #     output_flat[i] = torch.matmul(fft_basis, Dr[i])
    
    # # reshape Dr to match D's leading dimensions again
    # shape = list(D.shape)
    # shape[-2] = fft_basis.shape[0]
    # return output_flat.reshape(shape)

    # Project onto CQT basis
    if phase:
        C = torch.matmul(fft_basis, D)
    else:
        C = torch.matmul(fft_basis, torch.abs(D))
    
    return C.to(dtype=dtype)

def _trim_stack(
    cqt_resp: List[torch.Tensor],
    n_bins: int,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Stack and trim CQT responses in the correct order.
    """
    if len(cqt_resp) == 1:
        return cqt_resp[0]
    
    # Reverse the order of responses before concatenating
    # This ensures lower frequencies (bass) appear at the top
    return torch.cat(cqt_resp[::-1], dim=-2).to(dtype)

def cqt_frequencies(
    n_bins: int, 
    *, 
    fmin: float, 
    bins_per_octave: int = 12, 
    tuning: float = 0.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the center frequencies of Constant-Q bins using PyTorch.
    
    Parameters
    ----------
    n_bins : int > 0
        Number of constant-Q bins
    fmin : float > 0
        Minimum frequency
    bins_per_octave : int > 0
        Number of bins per octave
    tuning : float
        Deviation from A440 tuning in fractional bins
    device : torch.device or None
        Device to place the output tensor on
        
    Returns
    -------
    frequencies : torch.Tensor [shape=(n_bins,)]
        Center frequency for each CQT bin
    
    Examples
    --------
    >>> # Get the CQT frequencies for 24 notes, starting at C2
    >>> cqt_frequencies(24, fmin=65.406)  # C2 = 65.406 Hz
    tensor([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
              92.499,   97.999,  103.826,  110.000,  116.541,  123.471,
             130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
             184.997,  195.998,  207.652,  220.000,  233.082,  246.942])
    """
    if device is None:
        device = torch.device('cpu')
        
    correction: float = 2.0 ** (float(tuning) / bins_per_octave)
    
    frequencies = 2.0 ** (
        torch.arange(0, n_bins, dtype=torch.float32, device=device) / bins_per_octave
    )
    
    return correction * fmin * frequencies

def _et_relative_bw(
    bins_per_octave: int, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the relative bandwidth coefficient for equal (geometric) 
    frequency spacing for a given number of bins per octave using PyTorch.
    
    This is a special case of the more general `relative_bandwidth`
    calculation that can be used when only a single basis frequency
    is used.
    
    Parameters
    ----------
    bins_per_octave : int
        Number of bins per octave
    device : torch.device or None
        Device to place the output tensor on
        
    Returns
    -------
    alpha : torch.Tensor > 0 [shape=(1,)]
        Relative bandwidth coefficient
    """
    if device is None:
        device = torch.device('cpu')
        
    r = 2.0 ** (1.0 / bins_per_octave)
    return ((r ** 2 - 1) / (r ** 2 + 1)).unsqueeze(0).to(device)

def _relative_bandwidth(
    freqs: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the relative bandwidth for each of a set of specified frequencies using PyTorch.
    
    This function is used as a helper in wavelet basis construction.
    
    Parameters
    ----------
    freqs : torch.Tensor
        The array of frequencies
    device : torch.device or None
        Device to place the output tensor on
        
    Returns
    -------
    alpha : torch.Tensor
        Relative bandwidth
        
    Raises
    ------
    ValueError
        If fewer than 2 frequencies are provided
    """
    if device is None:
        device = freqs.device if torch.is_tensor(freqs) else torch.device('cpu')
    
    # Ensure input is a tensor
    if not torch.is_tensor(freqs):
        freqs = torch.tensor(freqs, device=device)
    
    if len(freqs) <= 1:
        raise ValueError(
            f"2 or more frequencies are required to compute bandwidths. Given freqs={freqs}"
        )
    
    # Initialize bandwidth array
    bpo = torch.empty_like(freqs, device=device)
    
    # Compute log2 of frequencies
    logf = torch.log2(freqs)
    
    # Reflect at the lowest and highest frequencies
    bpo[0] = 1 / (logf[1] - logf[0])
    bpo[-1] = 1 / (logf[-1] - logf[-2])
    
    # For everything else, do a centered difference
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])
    
    # Compute relative bandwidths
    two = torch.tensor(2.0, device=device)
    alpha = (two ** (2 / bpo) - 1) / (two ** (2 / bpo) + 1)
    
    return alpha

def wavelet_lengths(
    freqs: torch.Tensor,
    sr: float = 22050,
    window: Union[str, Callable] = "hann",
    filter_scale: float = 1,
    gamma: Optional[float] = 0,
    alpha: Optional[Union[float, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    """
    Return length of each filter in a wavelet basis using PyTorch.
    
    Parameters
    ----------
    freqs : torch.Tensor (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.
    sr : float > 0
        Audio sampling rate
    window : str or callable
        Window function to use on filters
    filter_scale : float > 0
        Resolution of filter windows. Larger values use longer windows.
    gamma : float >= 0 or None
        Bandwidth offset for determining filter lengths
    alpha : float > 0 or torch.Tensor or None
        Optional pre-computed relative bandwidth parameter
    device : torch.device or None
        Device to place tensors on
        
    Returns
    -------
    lengths : torch.Tensor
        The length of each filter
    f_cutoff : float
        The lowest frequency at which all filters' main lobes have decayed by at least 3dB
        
    Raises
    ------
    ValueError
        If input parameters don't meet requirements
    """
    if device is None:
        device = freqs.device if torch.is_tensor(freqs) else torch.device('cpu')
        
    # Ensure input is a tensor
    if not torch.is_tensor(freqs):
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    
    # Parameter validation
    if filter_scale <= 0:
        raise ValueError(f"filter_scale={filter_scale} must be positive")
        
    if gamma is not None and gamma < 0:
        raise ValueError(f"gamma={gamma} must be non-negative")
        
    if torch.any(freqs <= 0):
        raise ValueError("frequencies must be strictly positive")
        
    if len(freqs) > 1 and torch.any(freqs[:-1] > freqs[1:]):
        raise ValueError(f"Frequency array={freqs} must be in strictly ascending order")
    
    # Compute alpha if not provided
    if alpha is None:
        alpha = _relative_bandwidth(freqs=freqs)
    elif not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    
    # Handle gamma
    if gamma is None:
        gamma_ = alpha * 24.7 / 0.108
    else:
        gamma_ = torch.tensor(gamma, dtype=torch.float32, device=device)
    
    # Compute Q factor
    Q = float(filter_scale) / alpha
    
    # Compute cutoff frequency
    # Note: window_bandwidth needs to be implemented separately
    f_cutoff = torch.max(
        freqs * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma_
    ).item()

    
    # Convert frequencies to filter lengths
    lengths = Q * sr / (freqs + gamma_ / alpha)
    
    return lengths, f_cutoff

def window_bandwidth(window: Union[str, Callable]) -> float:
    """
    Compute the bandwidth of a window function.
    
    Parameters
    ----------
    window : str or callable
        Window specification
        
    Returns
    -------
    bandwidth : float
        The bandwidth of the window
    """
    # This is a simplified version - you might want to implement
    # more window types based on your needs
    if window == "hann":
        return 1.50018310546875
    elif window == "hamming":
        return 1.36328125
    elif window == "rectangular" or window == "ones":
        return 1.0
    else:
        # Default value for other windows
        return 1.0

def _vqt_filter_fft(
    sr: float,
    freqs: torch.Tensor,
    filter_scale: float,
    norm: Optional[float],
    sparsity: float,
    hop_length=None,
    window: str = "hann",
    gamma: float = 0.0,
    dtype: torch.dtype = torch.complex64,
    alpha: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Generate the frequency domain variable-Q filter basis in PyTorch.
    
    Parameters
    ----------
    sr : float
        Sample rate
    freqs : torch.Tensor
        Center frequencies
    filter_scale : float
        Filter scale factor
    norm : float or None
        Normalization factor
    sparsity : float
        Sparsity factor
    window : str
        Window type
    gamma : float
        Bandwidth offset
    dtype : torch.dtype
        Output dtype
    alpha : torch.Tensor or None
        Filter bandwidth
    device : torch.device or None
        Device to use for computation
        
    Returns
    -------
    fft_basis : torch.Tensor
        The FFT basis
    n_fft : int
        FFT size
    lengths : torch.Tensor
        Filter lengths
    """
    if device is None:
        device = freqs.device if torch.is_tensor(freqs) else torch.device('cpu')
    
    # Generate wavelet filters
    basis, lengths = wavelet(
        freqs=freqs,
        sr=sr,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
        gamma=gamma,
        alpha=alpha,
        device=device
    )
    # Get next power of 2 for FFT
    n_fft = basis.shape[1]
    if hop_length is not None:
        target_length = 2.0 ** (1 + math.ceil(math.log2(hop_length)))
        if n_fft < target_length:
            n_fft = int(target_length)

    # Normalize bases
    basis = basis * (lengths.unsqueeze(-1) / float(n_fft))

    # Compute FFT
    #fft_basis = torch.fft.rfft(basis, n=n_fft, dim=1)
    # fft that can take it complex float
    fft_basis = torch.fft.fft(basis, n=n_fft, dim=1)[..., :n_fft // 2 + 1]
    # Sparsify the basis
    if sparsity > 0:
        fft_basis = sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)
    else:
        fft_basis = fft_basis.to(dtype)

    return fft_basis, n_fft, lengths

def sparsify_rows(
    x: torch.Tensor, 
    quantile: float, 
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Sparsify rows of a tensor by zeroing out small elements.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    quantile : float
        Quantile threshold for sparsification
    dtype : torch.dtype
        Output dtype
        
    Returns
    -------
    torch.Tensor
        Sparsified tensor
    """
    if quantile == 0:
        return x.to(dtype)
    
    # Compute magnitude
    mag = torch.abs(x)
    
    # Compute threshold for each row
    thresh = torch.quantile(mag, quantile, dim=1, keepdim=True)
    
    # Create mask
    mask = mag >= thresh
    
    # Apply mask and convert to desired dtype
    return (x * mask).to(dtype)

def wavelet(
    freqs: torch.Tensor,
    sr: float = 22050,
    window: Union[str, Callable] = "hann",
    filter_scale: float = 1,
    pad_fft: bool = True,
    norm: Optional[float] = 1,
    dtype: torch.dtype = torch.complex64,
    gamma: float = 0,
    alpha: Optional[float] = None,
    device: Optional[torch.device] = None,
    pad_mode: str = 'constant'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a wavelet basis using windowed complex sinusoids in PyTorch.
    
    Parameters
    ----------
    freqs : torch.Tensor
        Center frequencies of the filters (in Hz)
    sr : float
        Audio sampling rate
    window : str or callable
        Windowing function to apply to filters
    filter_scale : float
        Scale of filter windows
    pad_fft : bool
        Center-pad all filters up to the nearest integral power of 2
    norm : float or None
        Type of norm for basis function normalization
    dtype : torch.dtype
        Output data type
    gamma : float
        Bandwidth offset for variable-Q transforms
    alpha : float or None
        Pre-computed relative bandwidth parameter
    device : torch.device or None
        Device to place tensors on
    pad_mode : str
        Padding mode for torch.nn.functional.pad
        
    Returns
    -------
    filters : torch.Tensor
        Complex time-domain filters
    lengths : torch.Tensor
        The length of each filter in samples
    """
    if device is None:
        device = freqs.device if torch.is_tensor(freqs) else torch.device('cpu')
        
    # Ensure freqs is a tensor
    if not torch.is_tensor(freqs):
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    
    # Get filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
        device=device
    )
    
    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Convert ilen to integer and ensure it's odd
        ilen = int(ilen.item())
        if ilen % 2 == 0:
            ilen += 1
            
        # Create time array centered around zero
        t = torch.arange(-ilen // 2, ilen // 2, dtype=torch.float32, device=device)
        
        # Build the complex sinusoid
        sig = torch.exp(1j * 2 * torch.pi * freq * t / sr)
        
        # Apply window
        window_vals = get_window(window, ilen, device=device)
        sig = sig * window_vals
        
        # Normalize
        if norm is not None:
            sig = normalize(sig, norm=norm)
            
        filters.append(sig)
    
    # Determine maximum length and pad if necessary
    max_len = int(max(lengths).item())
    if pad_fft:
        max_len = int(2.0 ** (torch.ceil(torch.log2(torch.tensor(max_len))).item()))
    
    # Pad and stack filters
    padded_filters = []
    for filt in filters:
        pad_size = max_len - len(filt)
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        padded = torch.nn.functional.pad(filt, (pad_left, pad_right), mode=pad_mode)
        padded_filters.append(padded)
    
    # Stack and convert to specified dtype
    filters = torch.stack(padded_filters).to(dtype)
    
    return filters, lengths

def get_window(
    window: Union[str, Callable],
    win_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Get window function values.
    
    Parameters
    ----------
    window : str or callable
        Window specification
    win_length : int
        Length of the window
    device : torch.device
        Device to place tensor on
        
    Returns
    -------
    window : torch.Tensor
        Window values
    """
    if callable(window):
        return torch.tensor(window(win_length), device=device)
    
    if window == "hann":
        return torch.hann_window(win_length, device=device)
    elif window == "hamming":
        return torch.hamming_window(win_length, device=device)
    elif window in ["rectangular", "ones"]:
        return torch.ones(win_length, device=device)
    else:
        return torch.hann_window(win_length, device=device)



def normalize(
    x: torch.Tensor,
    norm: float,
    axis: int = -1,
    threshold: float = 0.0,
    fill: bool = True
) -> torch.Tensor:
    """
    Normalize tensor along a given axis.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    norm : float
        Normalization order
    axis : int
        Axis along which to normalize
    threshold : float
        Minimum threshold for normalization
    fill : bool
        Whether to fill zeros with ones
        
    Returns
    -------
    x_norm : torch.Tensor
        Normalized tensor
    """
    # Handle complex tensors
    if x.is_complex():
        mag = torch.abs(x)
    else:
        mag = x
        
    # Compute norm
    if norm == float('inf'):
        length = torch.max(mag, dim=axis, keepdim=True).values
    elif norm == -float('inf'):
        length = torch.min(mag, dim=axis, keepdim=True).values
    elif norm == 0:
        length = torch.ones_like(x.sum(dim=axis, keepdim=True))
    else:
        length = torch.norm(mag, p=norm, dim=axis, keepdim=True)
    
    # Set small values to 1
    if fill:
        length = torch.where(length > threshold, length, torch.ones_like(length))
    
    return x / length

def icqt(
    C: torch.Tensor,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    filter_scale: float = 1,
    norm: Optional[float] = 1,
    sparsity: float = 0.01,
    window: Union[str, Callable] = "hann",
    scale: bool = True,
    length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Inverse Constant-Q Transform using PyTorch.
    
    Parameters are similar to librosa's icqt function.
    """
    if device is None:
        device = C.device

    if fmin is None:
        fmin = fmin_def  # C1 note frequency

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Get dimensions
    n_bins = C.shape[-2]
    n_octaves = int(math.ceil(float(n_bins) / bins_per_octave))

    # Calculate frequencies
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = _relative_bandwidth(freqs)

    # Get filter lengths
    lengths, f_cutoff = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        alpha=alpha
    )

    # Trim CQT if length is specified
    if length is not None:
        n_frames = int(math.ceil((length + max(lengths)) / hop_length))
        C = C[..., :n_frames]

    C_scale = torch.sqrt(torch.tensor(lengths, device=device))

    # Initialize output tensor
    y = None

    # Calculate sampling rates and hop lengths for each octave
    srs = [sr]
    hops = [hop_length]

    for i in range(n_octaves - 1):
        if hops[0] % 2 == 0:
            srs.insert(0, srs[0] * 0.5)
            hops.insert(0, hops[0] // 2)
        else:
            srs.insert(0, srs[0])
            hops.insert(0, hops[0])

    # Process each octave
    for i, (my_sr, my_hop) in enumerate(zip(srs, hops)):
        n_filters = min(bins_per_octave, n_bins - bins_per_octave * i)
        sl = slice(bins_per_octave * i, bins_per_octave * i + n_filters)

        # Get FFT basis
        fft_basis, n_fft, _ = _vqt_filter_fft(
            my_sr,
            freqs[sl],
            filter_scale,
            norm,
            sparsity,
            window=window,
            alpha=alpha[sl],
            device=device
        )

        # Convert to dense tensor and conjugate
        inv_basis = fft_basis.conj().T

        # Compute frequency domain power
        freq_power = 1 / torch.sum(torch.abs(inv_basis) ** 2, dim=0)
        freq_power *= n_fft / lengths[sl]

        # Inverse project the basis

        if scale:
            D_oct = torch.einsum(
                'fc,c,c,...ct->...ft',
                inv_basis,
                C_scale[sl],
                freq_power,
                C[..., sl, :],
            )
        else:
            D_oct = torch.einsum(
                'fc,c,...ct->...ft',
                inv_basis,
                freq_power,
                C[..., sl, :],
            )


        y_oct = torch.istft(
            D_oct,
            n_fft=n_fft,
            hop_length=my_hop,
            win_length=n_fft,
            window=torch.ones(n_fft, device=device),
            center=True,
            normalized=False,
            length=length,
        )

        # Resample with torchaudio native audio resample
        y_oct = torchaudio.transforms.Resample(orig_freq=1, new_freq=int(sr/my_sr))(y_oct)

        if y is None:
            y = y_oct
        else:
            y[..., :y_oct.shape[-1]] += y_oct

    if length is not None:
        y = y[..., :length]

    return y
