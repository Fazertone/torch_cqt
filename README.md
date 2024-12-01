# torch_cqt
Invertible Constant (and Variable) Q Transform in PyTorch, based on Librosa's implementation.
Avoids the mirror image present in the non-stationary gabor frames implementation at the cost of non-perfect invertibility.
![z_mag_nsgt](https://github.com/user-attachments/assets/a25c0b7d-f64a-418a-ad9c-b1638c6e3ccb)
NSGT (left) vs this repo (right)
# Prequisites
You only need PyTorch and Librosa (used to prepare frequencies).

# Example usage
```
import cqt_torch

hop_length = 256
bins_per_octave = 64

z = cqt_torch.cqt(audio, sr=sr, hop_length=hop_length, n_bins=9*bins_per_octave, bins_per_octave=bins_per_octave, tuning=0.0)
z_hat = cqt_torch.icqt(z, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)
```
