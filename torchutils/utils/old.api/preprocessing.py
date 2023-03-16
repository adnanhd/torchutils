#!/usr/bin/env python3
from typing import Tuple
import numpy as np
import torch
import warnings
# preprocessing


def _obtain_shape(dim, arr_shape):
    shape = tuple(1 if d in dim else -1 for d in range(len(arr_shape)))
    if shape.count(-1) > 1:
        warnings.warn(
            f'Obtained more than one dimension in dim={dim} and arr_shape={arr_shape}', RuntimeWarning)
    elif shape.count(-1) == 0:
        warnings.warn(
            f'Obtained no dimension in dim={dim} and arr_shape={arr_shape}', RuntimeWarning)
    else:
        return shape


def _default_dim_check(arr, dim):
    if isinstance(arr, torch.Tensor):
        return tuple()
    else:
        return None


def insert_shape_at_end(normalizer_fn):
    def wrapped_fn(arr, dim=None, shape=-1):
        if isinstance(dim, type(None)):
            dim = _default_dim_check(arr, dim)
        if shape == -1 and not isinstance(dim, type(None)):
            shape = _obtain_shape(dim, arr.shape)
            if shape is None:
                shape = -1
        return normalizer_fn(arr, dim, shape), shape
    return wrapped_fn


def append_shape_at_begin(denormalizer_fn):
    def wrapped_dfn(arr, key_shape):
        key, shape = key_shape
        return denormalizer_fn(arr, key, shape)
    return wrapped_dfn


def denormalize_model(solver, key_shape):
    def denormalize_model_interface(model):
        def denormalized_call_method(*args, **kwargs):
            res = model(*args, **kwargs)
            return solver(res, key_shape)
        model.__call__ = denormalized_call_method
        return model
    return denormalize_model_interface


@insert_shape_at_end
def l1_norm_key(arr, dim, shape=-1):
    return arr.sum(dim)


def to_key_tensor(key_shape, **kwargs):
    key, shape = key_shape
    def _to_tensor(k): return torch.from_numpy(
        k) if isinstance(k, np.ndarray) else torch.Tensor(k)
    if isinstance(key, tuple):
        key = tuple(_to_tensor(k).to(**kwargs) for k in key)
    else:
        key = _to_tensor(key).to(**kwargs)
    return key, shape


def l1_norm(arr, dim=None) -> Tuple[int, int]:
    nsum, shape = l1_norm_key(arr, dim)
    arr /= nsum.reshape(shape)
    return nsum, shape


@append_shape_at_begin
def l1_norm_solver(arr, key, shape):
    return arr * key.reshape(shape)


def l1_denorm(arr, key_shape) -> None:
    arr = l1_norm_solver(arr, key_shape)


@insert_shape_at_end
def mean_std_key(arr, dim, shape):
    return arr.mean(dim), arr.std(dim)


def mean_std_norm(arr, dim=None) -> Tuple[int, int]:
    (nmean, nstd), shape = mean_std_key(arr, dim)

    arr -= nmean.reshape(shape)
    arr /= nstd.reshape(shape)

    return (nmean, nstd), shape


@append_shape_at_begin
def mean_std_solver(arr, key, shape=-1):
    nmean, nstd = key
    return arr * nstd + nmean


def mean_std_denorm(arr, key_shape) -> None:
    arr = mean_std_solver(arr, key_shape)


@insert_shape_at_end
def min_max_key(arr, dim, shape=-1):
    return arr.min(dim), arr.max(dim)


def min_max_norm(arr, dim=None) -> Tuple[int, int]:
    (nmin, nmax), shape = min_max_key(arr, dim)

    arr -= nmin.reshape(shape)
    arr /= (nmax - nmin).reshape(shape)

    return (nmin, nmax), shape


@append_shape_at_begin
def min_max_solver(arr, key, shape=-1) -> None:
    nmin, nmax = key

    return arr * (nmax - nmin).reshape(shape) + nmin.reshape(shape)


def min_max_denorm(arr, key_shape):
    arr = min_max_solver(arr, key_shape)
