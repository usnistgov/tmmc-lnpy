from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lnpy.combine import combine


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


def test_check_windows_overlap() -> None:
    # no overlaps:
    overlap = pd.DataFrame({
        "n": [],
        "window": [],
    })

    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(3),
            macrostate_names="n",
            window_index_name="window",
            verbose=False,
        )

    overlap = pd.DataFrame({"n": [0, 0, 1, 1, 2, 2], "window": [0, 1, 1, 2, 2, 3]})

    combine.check_windows_overlap(
        overlap,
        windows=range(4),
        macrostate_names="n",
        window_index_name="window",
        verbose=True,
    )

    # missing node
    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(5),
            macrostate_names="n",
            window_index_name="window",
            verbose=True,
        )
    # two graphs
    overlap = pd.DataFrame({
        "n": [0, 0, 1, 1],
        "window": [0, 1, 2, 3],
    })

    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(4),
            macrostate_names="n",
            window_index_name="window",
            verbose=True,
        )


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    x = np.linspace(0, 10)
    return pd.DataFrame({
        "x": x,
        "y": np.sin(x),
        "z": np.cos(x),
    })


@pytest.fixture(params=[1, 2, 4, 5])
def table_sequence(
    table: pd.DataFrame, request: pytest.FixtureRequest
) -> list[pd.DataFrame]:
    if (n := request.param) == 1:
        return [table]

    split = len(table) // n
    lb = 0
    table_sequence = []
    for i in range(n + 1):
        ub = lb + split + 1
        table_sequence.append(table.iloc[lb:ub].assign(y=lambda x: x["y"] + 10 * i))  # noqa: B023  # pylint: disable=cell-var-from-loop

        lb += split
        if lb >= len(table):
            break

    return table_sequence


@pytest.fixture
def table_dataset(table: pd.DataFrame) -> xr.Dataset:
    return table.set_index("x").to_xarray()


@pytest.fixture
def table_dataset_sequence(table_sequence: list[pd.DataFrame]) -> list[xr.Dataset]:
    return [x.set_index("x").to_xarray() for x in table_sequence]


def test_table_sequence(
    table: pd.DataFrame, table_sequence: list[pd.DataFrame]
) -> None:
    if len(table_sequence) == 1:
        pd.testing.assert_frame_equal(table, table_sequence[0])

    else:
        other = pd.concat(table_sequence).groupby("x", as_index=False).mean()

        np.testing.assert_allclose(table["x"], other["x"])
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(table["y"], other["y"])


# * combine_scaled
def test_shift_lnpi_windows_no_overlap(table: pd.DataFrame) -> None:
    mid = len(table) // 2
    # no overlap:
    table_sequence = [table.iloc[:mid], table.iloc[mid:]]
    with pytest.raises(combine.OverlapError):
        combine.shift_lnpi_windows(
            combine.concat_windows(table_sequence),
            macrostate_names="x",
            lnpi_name="y",
        ).groupby("x", as_index=False).mean()


def test_shift_lnpi_windows_single_table(table: pd.DataFrame) -> None:
    # single table
    table_sequence = [table]
    new = (
        combine
        .shift_lnpi_windows(
            combine.concat_windows(table_sequence),
            macrostate_names="x",
            lnpi_name="y",
        )
        .groupby("x", as_index=False)
        .mean()
    )
    pd.testing.assert_frame_equal(table, new[table.columns])


mark_connected = pytest.mark.parametrize("check_connected", [False, True])
mark_sparse = pytest.mark.parametrize("use_sparse", [False, True])


@mark_connected
@mark_sparse
def test_shift_lnpi_windows_split(
    table: pd.DataFrame,
    table_sequence: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    new = (
        combine
        .shift_lnpi_windows(
            combine.concat_windows(table_sequence),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])


@mark_connected
@mark_sparse
def test_shift_lnpi_windows_split_other(
    table: pd.DataFrame,
    table_sequence: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    # with window name already there
    new = (
        combine
        .shift_lnpi_windows(
            combine.concat_windows([
                x.assign(window="hello", other=1, _window_index="there")
                for x in table_sequence
            ]),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .drop("window", axis=1)
        .groupby("x", as_index=False)
        .mean()
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])


@mark_connected
@mark_sparse
def test_shift_lnpi_windows_split_single_table(
    table: pd.DataFrame,
    table_sequence: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    # already formed single table:
    new = (
        combine
        .shift_lnpi_windows(
            pd.concat((x.assign(window=i) for i, x in enumerate(table_sequence))),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])

    # wrong name:
    with pytest.raises(ValueError, match=r".* single table must contain .*"):
        combine.shift_lnpi_windows(
            pd.concat((x.assign(window_wrong=i) for i, x in enumerate(table_sequence))),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )


# * combine.keep_first
def test_combine_keep_first_no_overlap(table: pd.DataFrame) -> None:
    mid = len(table) // 2
    table_sequence = [table.iloc[:mid], table.iloc[mid:]]
    with pytest.raises(combine.OverlapError):
        combine.keep_first(
            combine.concat_windows(table_sequence), state_name="x", check_connected=True
        )

    # dataset
    dss = [x.set_index("x").to_xarray() for x in table_sequence]
    with pytest.raises(combine.OverlapError):
        combine.keep_first(
            combine.concat_windows(dss, coord_names="x"),
            state_name="x",
            check_connected=True,
        )

    # dataarray
    das = combine.concat_windows([x["z"] for x in dss], coord_names="x")
    with pytest.raises(combine.OverlapError):
        combine.keep_first(das, state_name="x", check_connected=True)


def test_combine_keep_first_single_table(table: pd.DataFrame) -> None:
    new = combine.keep_first(combine.concat_windows([table]), state_name="x")
    pd.testing.assert_frame_equal(table, new[table.columns])

    new = combine.keep_first(table.assign(window=0), state_name="x")
    pd.testing.assert_frame_equal(table, new[table.columns])

    # dataset
    ds = table.set_index("x").to_xarray()
    ds_out = combine.keep_first(
        combine.concat_windows([ds], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(ds, ds_out.drop_vars("window"))

    ds_expand = ds.expand_dims("window")
    ds_out = combine.keep_first(ds_expand, state_name="x")
    xr.testing.assert_allclose(ds, ds_out.drop_vars("window"))

    ds_out = combine.keep_first(
        combine.concat_windows([ds_expand], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(ds, ds_out.drop_vars("window"))

    ds_stack = ds.expand_dims("window").stack(index=["window", "x"])  # noqa: PD013
    ds_out = combine.keep_first(ds_stack, state_name="x")
    xr.testing.assert_allclose(ds, ds_out.drop_vars("window"))

    ds_out = combine.keep_first(ds_stack, state_name="x", reset_window=False)
    xr.testing.assert_allclose(ds_stack, ds_out)

    ds_out = combine.keep_first(
        combine.concat_windows([ds_stack], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(ds, ds_out.drop_vars("window"))

    # dataarray
    da = ds["z"]
    da_out = combine.keep_first(
        combine.concat_windows([da], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(da, da_out.drop_vars("window"))

    da_expanded = da.expand_dims("window")
    da_out = combine.keep_first(da_expanded, state_name="x")
    xr.testing.assert_allclose(da, da_out.drop_vars("window"))

    da_out = combine.keep_first(
        combine.concat_windows([da_expanded], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(da, da_out.drop_vars("window"))

    da_stack = da.expand_dims("window").stack(index=["window", "x"])  # noqa: PD013
    da_out = combine.keep_first(da_stack, state_name="x")
    xr.testing.assert_allclose(da, da_out.drop_vars("window"))

    da_out = combine.keep_first(da_stack, state_name="x", reset_window=False)
    xr.testing.assert_allclose(da_stack, da_out)

    da_out = combine.keep_first(
        combine.concat_windows([da_stack], coord_names="x"), state_name="x"
    )
    xr.testing.assert_allclose(da, da_out.drop_vars("window"))

    # multiple variables in index
    da_stack = da.expand_dims(["rec", "window"]).stack(index=["rec", "window", "x"])  # noqa: PD013
    da_out = combine.keep_first(da_stack, state_name="x")
    expected = da_stack.reset_index("window")
    xr.testing.assert_allclose(expected, da_out)


def test_combine_keep_first_xarray_routines(table_dataset: xr.Dataset) -> None:
    # no window in coords:
    with pytest.raises(ValueError, match=r".* tables.indexes.*"):
        combine.keep_first(table_dataset.stack(index=["x"]))  # noqa: PD013

    # not window in dims
    with pytest.raises(ValueError, match=r".* in dimensions"):
        combine.keep_first(table_dataset)

    with pytest.raises(TypeError, match=r"Unknown .*"):
        combine.concat_windows(["hello"])  # type: ignore[list-item]  # on purpose error  # pyright: ignore[reportCallIssue,reportArgumentType]


def test_combine_keep_first_split(
    table: pd.DataFrame, table_sequence: list[pd.DataFrame]
) -> None:
    new = combine.keep_first(
        combine.concat_windows(table_sequence, coord_names="x"), state_name="x"
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # test odd window names:
    new = combine.keep_first(
        combine.concat_windows(
            [
                x.assign(window="hello", other=1, _window_index="there")
                for x in table_sequence
            ],
            coord_names="x",
        ),
        state_name="x",
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # using single table:
    new = combine.keep_first(
        pd.concat((x.assign(window=i) for i, x in enumerate(table_sequence))),
        state_name="x",
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # wrong name
    with pytest.raises(ValueError, match=r".* single table must contain .*"):
        combine.keep_first(
            pd.concat((x.assign(window_wrong=i) for i, x in enumerate(table_sequence))),
            state_name="x",
        )


@pytest.mark.parametrize("use_array", [False, True])
def test_combine_keep_first_split_dataset(
    table_dataset: xr.Dataset, table_dataset_sequence: list[xr.Dataset], use_array: bool
) -> None:
    seq: list[xr.DataArray] | list[xr.Dataset]
    seq = (
        [_["z"] for _ in table_dataset_sequence]
        if use_array
        else table_dataset_sequence
    )

    def _test_output(new: xr.DataArray | xr.Dataset) -> None:
        np.testing.assert_allclose(table_dataset["x"], new["x"])  # type: ignore[arg-type, unused-ignore]
        if isinstance(new, xr.DataArray):
            np.testing.assert_allclose(table_dataset["z"], new)
        else:
            np.testing.assert_allclose(table_dataset["z"], new["z"])

    _test_output(
        combine.keep_first(combine.concat_windows(seq, coord_names="x"), state_name="x")  # type: ignore[type-var]  # pyright: ignore[reportArgumentType]
    )

    # test odd window names:
    _test_output(
        combine.keep_first(
            combine.concat_windows(
                [  # pyright: ignore[reportCallIssue, reportArgumentType]
                    x.expand_dims("window").assign_coords(
                        window=("window", [str(-i)]),
                        _window_index=("window", ["there"]),
                    )
                    for i, x in enumerate(seq)
                ],
                coord_names="x",
            ),
            state_name="x",
        )
    )

    # using single table:
    stacked = xr.concat(
        (
            x  # noqa: PD013
            .expand_dims("window")
            .assign_coords(window=("window", [i]))
            .stack(index=["window", "x"])
            for i, x in enumerate(table_dataset_sequence)
        ),
        dim="index",
    )
    _test_output(
        combine.keep_first(
            stacked,
            state_name="x",
        )
    )

    # wrong name
    with pytest.raises(ValueError, match=r".*names"):
        combine.keep_first(
            stacked,
            state_name="x",
            window_name="window_other",
        )


# * Utilities
@pytest.mark.parametrize("use_running", [True, False])
@pytest.mark.parametrize("as_index", [True, False])
def test_updown_mean(
    as_index: bool, use_running: bool, rng: np.random.Generator
) -> None:
    nstate = 10
    nwindow = 2
    weight, down, up = rng.random((3, nwindow, nstate))

    df_updown = pd.DataFrame({
        "window": np.repeat(range(nwindow), nstate),
        "state": np.tile(range(nstate), nwindow),
        "n_trials": weight.reshape(-1),
        "prob_down": down.reshape(-1),
        "prob_up": up.reshape(-1),
    })

    # add a single value
    df_add = pd.DataFrame([
        {
            "window": nwindow,
            "state": nstate,
            "n_trials": rng.random(),
            "prob_down": rng.random(),
            "prob_up": rng.random(),
        }
    ])
    df_total = pd.concat((df_updown, df_add))

    out = combine.updown_mean(
        df_total, by="state", as_index=as_index, use_running=use_running
    )

    out_head = out.iloc[:-1]
    np.testing.assert_allclose(out_head["n_trials"], weight.sum(0))
    np.testing.assert_allclose(
        out_head["prob_down"], np.average(down, axis=0, weights=weight)
    )
    np.testing.assert_allclose(
        out_head["prob_up"], np.average(up, axis=0, weights=weight)
    )

    out_tail = out.iloc[-1]
    for k in ("n_trials", "prob_down", "prob_up"):
        np.testing.assert_allclose(out_tail[k], df_add.iloc[-1][k])


def test_updown_from_collectionmatrix(rng: np.random.Generator) -> None:
    c = rng.random((10, 3))

    table = pd.DataFrame({
        "n_trials": c.sum(-1),
        "prob_down": c[:, 0] / c.sum(-1),
        "prob_up": c[:, -1] / c.sum(-1),
    })

    out = combine.assign_updown_from_collectionmatrix(
        pd.DataFrame(c, columns=["c0", "c1", "c2"])
    )

    pd.testing.assert_frame_equal(table, out[table.columns])


@pytest.mark.parametrize(
    ("shape", "axis"), [((10, 2), 0), ((2, 10, 2), 1), ((1, 2, 3, 10), -1), ((10,), -1)]
)
@pytest.mark.parametrize("norm", [True, False])
def test_lnpi_from_updown_axis(rng, shape, axis, norm) -> None:
    def _idx(index):
        axes = [slice(None)] * len(shape)
        axes[axis] = index
        return tuple(axes)

    up = rng.random(shape)
    down = rng.random(shape)

    expected = np.zeros(shape)
    expected[_idx(0)] = 0.0
    expected[_idx(slice(1, None))] = np.log(
        up[_idx(slice(None, -1))] / down[_idx(slice(1, None))]
    )

    expected = expected.cumsum(axis=axis)
    expected -= expected.max(axis=axis, keepdims=True)  # pylint: disable=unexpected-keyword-arg

    if norm:
        expected = combine.normalize_lnpi(expected, axis=axis)

    # numpy
    np.testing.assert_allclose(
        expected, combine.lnpi_from_updown(down=down, up=up, axis=axis, norm=norm)
    )

    # xarray
    xup = xr.DataArray(up)
    xdown = xr.DataArray(down)

    dim = xdown.dims[axis]
    np.testing.assert_allclose(
        expected, combine.lnpi_from_updown(down=xdown, up=xup, dim=dim, norm=norm)
    )

    # series?
    if len(shape) == 1:
        sup = pd.Series(up)
        sdown = pd.Series(down)

        np.testing.assert_allclose(
            expected, combine.lnpi_from_updown(down=sdown, up=sup, norm=norm)
        )


@pytest.fixture
def table_updown(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        rng.random((10, 3)), columns=["n_trials", "prob_down", "prob_up"]
    )


def test_assign_delta_assign_lnpi_from_updown(table_updown: pd.DataFrame) -> None:
    delta_lnpi = np.log(  # type: ignore[attr-defined]
        table_updown["prob_up"].shift(1) / table_updown["prob_down"]
    ).fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]

    out = combine.assign_delta_lnpi_from_updown_indexed(table_updown)
    np.testing.assert_allclose(delta_lnpi, out["delta_lnpi"])

    # test dataset
    ds = table_updown.to_xarray()

    dim = list(ds.dims)[-1]

    out_ds = combine.assign_delta_lnpi_from_updown_indexed(
        ds, delta_lnpi_name="my_delta", dim=dim
    )
    np.testing.assert_allclose(delta_lnpi, out_ds["my_delta"])

    # Series with name
    out_series = combine.delta_lnpi_from_updown(
        down=table_updown["prob_down"], up=table_updown["prob_up"], name="my_delta"
    )
    assert isinstance(out_series, pd.Series)
    assert out_series.name == "my_delta"
    np.testing.assert_allclose(delta_lnpi, out_series)

    # xarray
    out_dataarray = combine.delta_lnpi_from_updown(
        down=table_updown["prob_down"].to_xarray(),
        up=table_updown["prob_up"].to_xarray(),
        name="my_delta",
    )
    assert isinstance(out_dataarray, xr.DataArray)
    assert out_dataarray.name == "my_delta"
    np.testing.assert_allclose(delta_lnpi, out_series)

    # numpy
    delta = combine.delta_lnpi_from_updown(
        down=table_updown["prob_down"].to_numpy(),
        up=table_updown["prob_up"].to_numpy(),
    )
    np.testing.assert_allclose(delta_lnpi, delta)


def test_assign_lnpi_from_updown(table_updown: pd.DataFrame) -> None:
    table = (
        combine
        .assign_delta_lnpi_from_updown_indexed(table_updown)
        .assign(ln_prob=lambda x: x["delta_lnpi"].cumsum())
        .assign(ln_prob=lambda x: x["ln_prob"] - x["ln_prob"].max())
    )
    out = combine.assign_lnpi_from_updown(table_updown, norm=False)
    np.testing.assert_allclose(out["ln_prob"], table["ln_prob"])

    out = combine.assign_lnpi_from_updown(table_updown, norm=True)
    table = table.assign(
        ln_prob=lambda x: x["ln_prob"] - np.log(np.exp(x["ln_prob"]).sum())
    )
    np.testing.assert_allclose(out["ln_prob"], table["ln_prob"])

    # dataset
    ds = combine.assign_lnpi_from_updown(table_updown.to_xarray(), norm=True)
    assert isinstance(ds, xr.Dataset)

    np.testing.assert_allclose(out["ln_prob"], table["ln_prob"])
