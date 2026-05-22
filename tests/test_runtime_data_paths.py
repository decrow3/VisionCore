import numpy as np
import torch

from models.data.datasets import DictDataset
from models.data import datafilters, loading


class _FakeSession:
    def __init__(self, missing_pct):
        self._missing_pct = missing_pct

    def get_cluster_ids(self):
        return np.array([0, 1])

    def get_missing_pct_interp(self, cids):
        def interp(_times):
            return self._missing_pct

        return interp


def test_get_embedded_datasets_maps_cluster_ids_metadata(monkeypatch):
    dset = DictDataset(
        {
            "robs": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "dfs": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.bool),
            "trial_inds": torch.tensor([0, 0]),
        },
        metadata={"cluster_ids": np.array([10, 20, 30])},
    )

    class _StubCombinedDataset:
        def __init__(self, dsets, dsets_inds, keys_lags):
            self.dsets = dsets
            self.dsets_inds = dsets_inds
            self.keys_lags = keys_lags

    monkeypatch.setattr(loading, "CombinedEmbeddedDataset", _StubCombinedDataset)
    monkeypatch.setattr(
        loading,
        "split_inds_by_trial",
        lambda dset, inds, train_val_split, seed: (inds, inds[:0]),
    )

    train_dset, val_dset = loading.get_embedded_datasets(
        sess=None,
        types=[dset],
        keys_lags={"robs": np.array([0])},
        train_val_split=0.5,
        cids=np.array([20, 30]),
        pre_func=lambda x: x,
    )

    expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
    assert torch.equal(train_dset.dsets[0]["robs"], expected)
    assert torch.equal(val_dset.dsets[0]["robs"], expected)


def test_missing_pct_accepts_numpy_output_and_cluster_ids_metadata():
    missing_pct = np.array(
        [
            [10.0, 80.0],
            [20.0, 90.0],
        ]
    )
    dset = DictDataset(
        {
            "t_bins": torch.tensor([0.0, 1.0]),
        },
        metadata={
            "cluster_ids": np.array([10, 20]),
            "sess": _FakeSession(missing_pct),
        },
    )

    mask = datafilters._make_missing_pct(45)(dset)

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 2)
    assert torch.equal(mask[:, 0], torch.tensor([True, True]))
    assert torch.equal(mask[:, 1], torch.tensor([True, True]))