from __future__ import annotations

from pathlib import Path

# import os
import torch

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
# _Jd = torch.load("/usr/scratch2/tortin13/chexia/equiformer_v2/nets/equiformer_v2/Jd.pt")

# this_file_directory = os.path.dirname(os.path.realpath(__file__))

# try:
#     os.path.exists(this_file_directory + "/Jd.pt")
#     _Jd = torch.load(this_file_directory + "/Jd.pt")
# except FileNotFoundError:
#     raise FileNotFoundError("File Jd.pt not found in the current directory")

this_file_directory = Path(__file__).resolve().parent
jd_path = this_file_directory / "Jd.pt"

if not jd_path.exists():
    raise FileNotFoundError("File Jd.pt not found in the current directory")

_Jd = torch.load(jd_path)


def wigner_D(l_dx, alpha, beta, gamma):
    if not l_dx < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l_dx].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l_dx)
    Xb = _z_rot_mat(beta, l_dx)
    Xc = _z_rot_mat(gamma, l_dx)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l_dx):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l_dx + 1, 2 * l_dx + 1))
    inds = torch.arange(0, 2 * l_dx + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l_dx, -1, -1, device=device)
    frequencies = torch.arange(l_dx, -l_dx - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
