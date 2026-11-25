import copy
import gc
import re

import numpy as np
import patsy
import polars as pl
import pytest
from pandas.api.types import CategoricalDtype

# needs torch
torch = pytest.importorskip("torch")

from pySEQTarget import SEQopts, SEQuential  # noqa: E402
from pySEQTarget.condense import RecordStore, condense  # noqa: E402
from pySEQTarget.condense._census import _Census  # noqa: E402
from pySEQTarget.condense._frame import _bind_weights  # noqa: E402
from pySEQTarget.condense._frame import (_expanded, _outcome_frame,
                                         _record_frame, _segments)
from pySEQTarget.condense._models import _Block, _Model  # noqa: E402
from pySEQTarget.condense._probes import _pilot_fits  # noqa: E402
from pySEQTarget.condense._select import _targets  # noqa: E402
from pySEQTarget.condense._weights import _aligned  # noqa: E402
from pySEQTarget.condense._weights import (_attach_sensitivities, _classes,
                                           _package_params, _Theta,
                                           _weight_models)
from pySEQTarget.data import load_data  # noqa: E402

import statsmodels.api as sm  # noqa: E402

PRE = dict(weighted=True, weight_preexpansion=True)


def _glm(X, y, w):
    return sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w).fit()

EXCUSED = dict(PRE, excused=True, excused_colnames=["excusedZero", "excusedOne"])
EXCUSED3 = dict(
    PRE, excused=True, excused_colnames=["excusedZero", "excusedOne", "excusedOne"]
)
# weight models fit on the expansion
POST = dict(weighted=True, weight_preexpansion=False)
POST_EXCUSED = dict(POST, excused=True, excused_colnames=["excusedZero", "excusedOne"])
WEIGHT_FORMULAS = ("numerator", "denominator", "cense_numerator", "cense_denominator")


def _analysis(method, data, **opts):
    return SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method=method,
        parameters=SEQopts(**opts),
    )


def _fit(method, data, **opts):
    s = _analysis(method, data, **opts)
    s.expand()
    s.fit()
    return s


def _levels(frame):
    return {
        c: list(frame[c].cat.categories)
        for c in frame.columns
        if isinstance(frame[c].dtype, CategoricalDtype)
    }


def _z(fit, ref):
    params = ref.params.iloc[:, 0] if ref.params.ndim == 2 else ref.params
    bse = ref.bse.iloc[:, 0] if ref.bse.ndim == 2 else ref.bse
    names = list(fit.params.index)
    params, bse = _aligned(params, names).to_numpy(), _aligned(bse, names).to_numpy()
    return float(np.max(np.abs((fit.params.to_numpy() - params) / bse)))


def _weight_model(s, name):
    if name.startswith(("numerator_", "denominator_")):
        kind, i = name.rsplit("_", 1)
        return getattr(s, f"{kind}_model")[int(i)]
    return getattr(s, f"{name}_model")


def _record_block(s, frame):
    formulas = {name: getattr(s, name) for name in WEIGHT_FORMULAS}
    models = [
        m for m in _weight_models(s, formulas) if _package_params(s, m.name) is not None
    ]
    for m in models:
        m.classes = _classes(s, m.name)
    return _Block(models, frame, _levels(frame), "ID")


def _three_levels(data):
    # treated periods alternate between levels 1 and 2
    return data.with_columns(
        (pl.col("tx_init") * (1 + (pl.col("ID") + pl.col("time") // 10) % 2)).alias(
            "tx_init"
        )
    )


def test_record_store_round_trip(tmp_path):
    data = load_data("SEQdata")
    store = RecordStore.write(data, "ID", tmp_path, people_per_shard=70)

    assert store.ids() == sorted(data["ID"].unique().to_list())
    # whole people in ID order with rows in original order
    batches = list(store.batches(25))
    assert all(b["ID"].n_unique() <= 25 for b in batches)
    assert pl.concat(batches).equals(data.sort("ID", maintain_order=True))

    some = [5, 150, 71, 299]
    assert store.read(some).equals(
        data.filter(pl.col("ID").is_in(some)).sort("ID", maintain_order=True)
    )


def test_census_reproduces_the_package_formulas(tmp_path):
    data = load_data("SEQdata")
    opts = dict(PRE, weight_spline=True, followup_spline=True, followup_include=False)
    s = _fit("censoring", data, **opts)
    store = RecordStore.write(data, "ID", tmp_path, people_per_shard=70)
    census = _Census(_analysis("censoring", data, **opts), store, 40)

    # knots and centring from batches match the package
    record = s.data.to_pandas()
    for name in ("numerator", "denominator"):
        assert "cr(time" in census.formulas[name]
        np.testing.assert_allclose(
            patsy.dmatrix(census.formulas[name], record),
            patsy.dmatrix(getattr(s, name), record),
            rtol=1e-12,
            atol=1e-12,
        )
    followup = {"followup": s.DT["followup"].to_numpy()}
    np.testing.assert_allclose(
        patsy.dmatrix(census.followup_knots, followup),
        patsy.dmatrix(s._followup_spline_knots, followup),
        rtol=1e-12,
        atol=1e-12,
    )
    frame = _outcome_frame(s, s)
    block = _Block(
        [_Model("outcome", "outcome", census.covariates)],
        frame,
        census.levels["outcome"],
        "ID",
        "weight",
    )

    # the names carry the centring constants, equal to rounding (checked above)
    def names(columns):
        return [
            re.sub(r"constraints=\[\[[^]]*\]\]", "constraints", c) for c in columns
        ]

    exog_names = s.outcome_model[0]["outcome"].model.exog_names
    assert names(block.columns) == names(exog_names)


@pytest.mark.parametrize(
    "dataset,method,opts",
    [
        ("SEQdata", "ITT", {}),
        ("SEQdata", "censoring", PRE),
        ("SEQdata", "dose-response", PRE),
        ("SEQdata", "censoring", EXCUSED),
        ("SEQdata", "censoring", POST),
        ("SEQdata_LTFU", "ITT", dict(PRE, cense_colname="LTFU")),
    ],
)
def test_batches_reproduce_the_expansion(tmp_path, dataset, method, opts):
    data = load_data(dataset)
    s = _fit(method, data, **opts)
    store = RecordStore.write(data, "ID", tmp_path, people_per_shard=100)

    full = _outcome_frame(s, s)
    block = _Block(
        [_Model("outcome", "outcome", s.covariates)],
        full,
        _levels(full),
        "ID",
        "weight" if s.weighted else None,
    )
    assert block.columns == list(s.outcome_model[0]["outcome"].model.exog_names)

    outcome, record = [], []
    for r in store.batches(40):
        b = _expanded(s, r)
        if s.weighted:
            record.append(_record_frame(s, b))
            _bind_weights(s, b)
        outcome.append(block.matrices(_outcome_frame(s, b)))
    want = block.matrices(full)
    for field in ("ids", "X", "Y"):
        np.testing.assert_array_equal(
            np.concatenate([getattr(rows, field) for rows in outcome]),
            getattr(want, field),
        )
    # batch weights match to rounding
    np.testing.assert_allclose(
        np.concatenate([rows.w for rows in outcome]), want.w, rtol=1e-12, atol=0
    )

    if s.weighted:
        # fresh expansion as fit adds weights to DT
        full_record = _record_frame(s, _expanded(s, data))
        rblock = _record_block(s, full_record)
        parts = [rblock.matrices(frame) for frame in record]
        want = rblock.matrices(full_record)
        for field in ("ids", "X", "Y", "member"):
            np.testing.assert_array_equal(
                np.concatenate([getattr(rows, field) for rows in parts]),
                getattr(want, field),
            )


def test_unit_targets_add_up_to_the_package_score():
    s = _fit("censoring", load_data("SEQdata"), **PRE)
    ref = s.outcome_model[0]["outcome"]
    full = _outcome_frame(s, s)
    block = _Block(
        [_Model("outcome", "outcome", s.covariates)],
        full,
        _levels(full),
        "ID",
        "weight",
    )
    rows = block.matrices(full)
    seg, starts = _segments(rows.ids)

    G, H, _ = _targets(
        torch.tensor(rows.X),
        torch.tensor(rows.w),
        torch.tensor(seg),
        [torch.tensor(ref.params.to_numpy())[None, :, None]],
        len(starts),
    )
    G, H = G[:, 0].numpy(), H[:, 0].numpy()
    # outcome enters through XWy alone
    score = G.sum(0) - rows.X.T @ (rows.w * rows.Y[:, 0])
    cov = np.linalg.inv(H.sum(0))
    # score is zero and information gives the package SEs
    assert np.abs(cov @ score / ref.bse.to_numpy()).max() < 1e-4
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), ref.bse, rtol=1e-4)


def _with_theta(s, theta, layout):
    s = copy.copy(s)
    layout.apply(s, theta)
    return s


@pytest.mark.parametrize(
    "dataset,method,opts",
    [
        ("SEQdata", "censoring", PRE),
        ("SEQdata", "dose-response", PRE),
        ("SEQdata", "censoring", EXCUSED),
        ("SEQdata_LTFU", "censoring", dict(EXCUSED, cense_colname="LTFU")),
        ("SEQdata", "censoring", POST),
        ("SEQdata", "dose-response", POST),
        ("SEQdata_LTFU", "ITT", dict(POST, cense_colname="LTFU")),
        ("SEQdata_LTFU", "censoring", dict(POST, cense_colname="LTFU")),
        ("SEQdata", "censoring", POST_EXCUSED),
        ("SEQdata_LTFU", "censoring", dict(POST_EXCUSED, cense_colname="LTFU")),
        ("three", "censoring", dict(PRE, treatment_level=[0, 1, 2])),
        ("three", "censoring", dict(POST, treatment_level=[0, 1, 2])),
        ("three", "censoring", dict(PRE, treatment_level=[1, 2])),
        ("three", "censoring", dict(EXCUSED3, treatment_level=[0, 1, 2])),
        (
            "three",
            "censoring",
            dict(EXCUSED3, treatment_level=[0, 1, 2], weight_preexpansion=False),
        ),
        ("SEQdata_LTFU", "ITT", dict(PRE, cense_colname="LTFU")),
        ("SEQdata_LTFU", "ITT", dict(PRE, visit_colname="LTFU")),
    ],
)
def test_weight_sensitivities_are_derivatives_of_the_package_weights(
    dataset, method, opts
):
    three = dataset == "three"
    data = _three_levels(load_data("SEQdata")) if three else load_data(dataset)
    s = _fit(method, data, **opts)
    # fresh expansion as fit adds weights to DT
    frame = _record_frame(s, _expanded(s, data))
    block = _record_block(s, frame)
    layout = _Theta(block)
    theta = layout.of(s)

    # also checks rebuilt weights against the package
    b = _expanded(s, data)
    _bind_weights(s, b)
    s_cols = _attach_sensitivities(s, block, theta, frame, b)
    S = b.DT.select(s_cols).to_numpy()

    # central differences of the package log weights
    bse = np.concatenate(
        [np.ravel(_weight_model(s, name).bse.to_numpy().T) for name in layout.names]
    )
    v = np.random.default_rng(0).standard_normal(layout.size) * bse * 1e-4

    def log_weights(th):
        b = _expanded(s, data)
        _bind_weights(_with_theta(s, th, layout), b)
        return np.log(b.DT["weight"].to_numpy())

    fd = (log_weights(theta + v) - log_weights(theta - v)) / 2
    assert np.abs(S @ v).max() > 1e-6
    np.testing.assert_allclose(S @ v, fd, rtol=0, atol=1e-6 * np.abs(fd).max())


def test_separated_gold_subsample_is_an_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    X = np.column_stack([np.ones(200), x, rng.standard_normal(200)])
    y = (x > 0).astype(float)
    with pytest.raises(ValueError, match="gold_fraction"):
        _pilot_fits(X, y, np.ones(200), 0, 4, rng)


def test_enough_pseudo_rows_are_exact(tmp_path):
    # at most 10 record rows and 55 expanded rows per person
    data = load_data("SEQdata").filter(pl.col("time") < 10)
    s = _fit("censoring", data, **PRE)
    store = RecordStore.write(data, "ID", tmp_path)

    c = condense(
        _analysis("censoring", data, **PRE),
        store,
        k=55,
        k_record=10,
        gold_fraction=0.5,
        n_probes=8,
        n_submodels=2,
    )
    assert c.outcome.diagnostics["exact"].all()
    assert c.record.diagnostics["exact"].all()

    # weight models equal the package
    for name, fit in c.fit_weights().items():
        assert _z(fit, _weight_model(s, name)) < 1e-6

    # outcome fit is exact at the pilot weights
    frames = []
    for r in store.batches():
        b = _expanded(c.analysis, r)
        _bind_weights(c.analysis, b)
        frames.append(_outcome_frame(c.analysis, b))
    block = _Block(
        [_Model("outcome", "outcome", c.census.covariates)],
        frames[0],
        c.census.levels["outcome"],
        "ID",
        "weight",
    )
    rows = [block.matrices(f) for f in frames]
    exact = _glm(
        np.vstack([r.X for r in rows]),
        np.concatenate([r.Y[:, 0] for r in rows]),
        np.concatenate([r.w for r in rows]),
    )
    np.testing.assert_allclose(
        c.fit(theta="pilot").params, exact.params, rtol=1e-6, atol=1e-8
    )


def test_pseudo_rows_match_the_package_fit(tmp_path):
    data = load_data("SEQdata")
    s = _fit("ITT", data)
    store = RecordStore.write(data, "ID", tmp_path)

    c = condense(_analysis("ITT", data), store, k=32, gold_fraction=0.2)
    ref = s.outcome_model[0]["outcome"]
    fit = c.fit()
    assert c.outcome.X.shape[0] <= 300 * 32
    assert _z(fit, ref) < 0.1
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.03
    # chosen rows reproduce the gradient on held out probes
    d = c.outcome.diagnostics.filter(~pl.col("exact"))
    assert d["rel_grad"].median() < 1e-2


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    data = load_data("SEQdata")
    s = _fit("censoring", data, **PRE)
    store = RecordStore.write(data, "ID", tmp_path_factory.mktemp("store"))
    c = condense(
        _analysis("censoring", data, **PRE), store, k=32, gold_fraction=0.2
    )
    return data, s, c, store


def test_weighted_fit_matches_the_package(weighted):
    _, s, c, _ = weighted
    for name, fit in c.fit_weights().items():
        assert _z(fit, _weight_model(s, name)) < 0.05
    ref = s.outcome_model[0]["outcome"]
    # weights come from the condensed weight models
    fit = c.fit()
    assert _z(fit, ref) < 0.1
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.03


def test_bootstrap_replicate_refits_the_weights(weighted):
    data, _, c, _ = weighted
    rng = np.random.default_rng(1)
    ids = data["ID"].unique().sort().to_numpy()
    draw = rng.choice(ids, size=len(ids), replace=True)
    counts = dict(zip(*np.unique(draw, return_counts=True)))

    # package fit on the same resample
    copies = pl.DataFrame({"ID": draw, "copy": np.arange(len(draw))})
    resampled = (
        data.join(copies, on="ID", how="inner")
        .with_columns((pl.col("copy") + 1).cast(data.schema["ID"]).alias("ID"))
        .drop("copy")
        .sort(["ID", "time"])
    )
    s = _fit("censoring", resampled, **PRE)

    for name, fit in c.fit_weights(counts).items():
        assert _z(fit, _weight_model(s, name)) < 0.1
    assert _z(c.fit(counts=counts), s.outcome_model[0]["outcome"]) < 0.25


def test_competing_event_shares_the_design(tmp_path):
    # half the people to keep it short
    data = load_data("SEQdata_LTFU").filter(pl.col("ID") <= 500)
    opts = dict(compevent_colname="LTFU")
    s = _fit("ITT", data, **opts)
    store = RecordStore.write(data, "ID", tmp_path)

    c = condense(_analysis("ITT", data, **opts), store, gold_fraction=0.5)
    assert c.outcome.models == ["outcome", "compevent"]
    assert _z(c.fit("outcome"), s.outcome_model[0]["outcome"]) < 0.1
    assert _z(c.fit("compevent"), s.outcome_model[0]["compevent"]) < 0.1


def test_corrected_fit_on_everyone_is_the_exact_fit(weighted):
    _, s, c, store = weighted
    everyone = c.exact(store)
    for lam in (0.0, 1.0, "optimal"):
        fit = c.fit_corrected(sample=everyone, lam=lam)
        # condensed rows cancel when everyone is sampled
        for name, wf in fit.weights.items():
            assert _z(wf, _weight_model(s, name)) < 1e-6
            assert (wf.bse_design == 0).all()
        # weights move by first order sensitivity
        assert _z(fit, s.outcome_model[0]["outcome"]) < 1e-3
        assert (fit.bse_design == 0).all()


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    # eight rows per person is several SEs off
    data = load_data("SEQdata")
    s = _fit("ITT", data)
    store = RecordStore.write(data, "ID", tmp_path_factory.mktemp("coarse"))
    c = condense(_analysis("ITT", data), store, k=8, gold_fraction=0.2)
    return s, c, c.exact(store)


def test_corrected_fit_removes_the_condensed_rows_error(coarse):
    s, c, everyone = coarse
    ref = s.outcome_model[0]["outcome"]
    assert _z(c.fit(), ref) > 2

    rng = np.random.default_rng(0)
    errors, ses = [], []
    for _ in range(40):
        sample = everyone.subset(rng.choice(c.cohort, size=60, replace=False))
        fit = c.fit_corrected(sample=sample)
        assert 0.9 < fit.lam <= 1
        errors.append((fit.params - ref.params).to_numpy())
        ses.append(fit.bse_design.to_numpy())
    errors, ses = np.array(errors), np.array(ses)
    # corrected fit is centred and its design SE calibrated
    mc_se = errors.std(0, ddof=1) / np.sqrt(len(errors))
    assert np.abs(errors.mean(0) / mc_se).max() < 3.5
    assert np.abs(errors.mean(0) / ref.bse.to_numpy()).max() < 0.1 * _z(c.fit(), ref)
    spread = errors.std(0, ddof=1) / np.sqrt((ses**2).mean(0))
    assert (0.6 < spread).all() and (spread < 1.5).all()
    # design SE well below the sample alone
    alone = c.fit_corrected(sample=sample, lam=0.0).bse_design.to_numpy()
    assert (ses.mean(0) < 0.5 * alone).all()


def test_corrected_fit_needs_the_sample_alone_at_lam_zero(coarse):
    s, c, everyone = coarse
    sample = everyone.subset(np.random.default_rng(1).choice(c.cohort, 60, False))
    fit = c.fit_corrected(sample=sample, lam=0.0)
    # lam 0 is the sample fit alone
    rows = sample.outcome
    exact = _glm(rows.X, rows.Y[:, 0], rows.w)
    np.testing.assert_allclose(fit.params, exact.params, rtol=1e-6, atol=1e-8)


def test_cox_from_counts_is_the_cox_fit():
    from lifelines import CoxPHFitter

    from pySEQTarget.condense._estimands import _cox

    rng = np.random.default_rng(0)
    n = 4000
    arm = rng.integers(0, 2, n)
    time = np.minimum(rng.geometric(0.05 * np.exp(0.4 * arm)) - 1, 30)
    event = (time < 30) & (rng.random(n) < 0.9)
    events = np.zeros((2, 31))
    kept = np.zeros((2, 31))
    np.add.at(events, (arm[event], time[event]), 1)
    np.add.at(kept, (arm, time), 1)
    at_risk = kept[:, ::-1].cumsum(1)[:, ::-1]

    import pandas as pd

    cph = CoxPHFitter().fit(
        pd.DataFrame({"t": time, "e": event.astype(int), "arm": arm}), "t", "e"
    )
    assert abs(_cox(events, at_risk, [0, 1]) - cph.params_.iloc[0]) < 1e-6


@pytest.fixture(scope="module")
def estimands(tmp_path_factory):
    from types import SimpleNamespace

    import pandas as pd

    # equal width IDs avoid the package trial key collision
    data = load_data("SEQdata").with_columns(pl.col("ID") + 1000)
    s = _analysis("ITT", data, bootstrap_nboot=4)
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    s.hazard()
    store = RecordStore.write(data, "ID", tmp_path_factory.mktemp("estimands"))
    c = condense(_analysis("ITT", data), store, gold_fraction=0.2)

    def params(models):
        model = s._offloader.load_model(models["outcome"])
        return np.asarray(model.params)

    assert list(c.fit().params.index) == list(
        s._offloader.load_model(s.outcome_model[0]["outcome"]).model.exog_names
    )
    main = {"outcome": SimpleNamespace(params=pd.Series(params(s.outcome_model[0])))}
    replicates = [
        (dict(counts), {"outcome": params(s.outcome_model[i + 1])})
        for i, counts in enumerate(s._boot_samples)
    ]
    return s, c, main, replicates


def test_risks_are_the_package_risks_at_the_same_coefficients(estimands):
    s, c, main, replicates = estimands
    c.replicates = replicates
    out = c.survival(fits=main)
    ref = s.km_data.filter(pl.col("estimate") == "risk")
    both = ref.join(
        out["km_data"].filter(pl.col("estimate") == "risk"),
        on=["followup", "tx_init"],
        suffix="_c",
    )
    assert both.height == ref.height
    for col in ("pred", "SE", "LCI", "UCI"):
        np.testing.assert_allclose(both[col], both[f"{col}_c"], rtol=0, atol=1e-10)
    for key in ("risk_difference", "risk_ratio"):
        for col in s.risk_estimates[key].columns:
            np.testing.assert_allclose(
                s.risk_estimates[key][col], out[key][col], rtol=0, atol=1e-10
            )


def test_condensed_risks_and_hazard_ratio_match_the_package(estimands):
    s, c, main, replicates = estimands
    del c.__dict__["replicates"]
    risk = c.survival()["km_data"].filter(pl.col("estimate") == "risk")
    ref = s.km_data.filter(pl.col("estimate") == "risk")
    both = ref.join(risk, on=["followup", "tx_init"], suffix="_c")
    assert (both["pred"] - both["pred_c"]).abs().max() < 1e-3

    # simulated so the package value is one draw
    log_hr = np.log([c.hazard(seed=seed)["Hazard ratio"][0] for seed in range(12)])
    package = np.log(s.hazard_ratio["Hazard ratio"][0])
    assert abs(package - log_hr.mean()) < 4 * log_hr.std(ddof=1)

    c.bootstrap(6, seed=1)
    out = c.survival()
    assert {"SE", "LCI", "UCI"} <= set(out["km_data"].columns)
    assert out["risk_difference"]["RD SE"][0] > 0
    hr = c.hazard()
    assert hr["LCI"][0] < hr["Hazard ratio"][0] < hr["UCI"][0]


def test_pseudo_rows_are_the_persons_own_rows(tmp_path):
    data = load_data("SEQdata")
    s = _fit("censoring", data, **PRE)
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(
        _analysis("censoring", data, **PRE),
        store,
        k=32,
        gold_fraction=0.2,
    )
    # every condensed row is one of the persons own rows
    exact = c.exact(store).outcome
    cols = c.outcome.model_cols[0]
    for person in (0, 57, 211):
        pid = c.outcome.ids[person]
        own = np.round(exact.X[np.ix_(exact.ids == pid, cols)], 9)
        mine = np.round(c.outcome.X[np.ix_(c.outcome.person == person, cols)], 9)
        known = {row.tobytes() for row in own}
        assert all(row.tobytes() in known for row in mine)
    assert (c.outcome.w >= 0).all()

    for name, fit in c.fit_weights().items():
        assert _z(fit, _weight_model(s, name)) < 0.05
    ref = s.outcome_model[0]["outcome"]
    fit = c.fit()
    assert _z(fit, ref) < 0.05
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.02
    assert c.outcome.S is not None
    d = c.outcome.diagnostics.filter(~pl.col("exact"))
    assert d["rel_cross"].median() < 0.1


def test_unweighted_table_keeps_its_event_rows(tmp_path):
    data = load_data("SEQdata")
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis("ITT", data), store, k=8, gold_fraction=0.2)
    rows = c.outcome.rows("outcome")
    # event rows add up to XWy
    t = np.zeros_like(c.outcome.t[0])
    np.add.at(t, rows["person_events"], rows["wy_events"][:, None] * rows["X_events"])
    np.testing.assert_allclose(t, c.outcome.t[0], rtol=1e-12, atol=1e-12)
    # event rows are the expansion events
    exact = c.exact(store).outcome
    assert len(rows["X_events"]) == int((exact.Y[:, 0] == 1).sum())
    assert rows["X"].shape == (len(rows["w"]), len(rows["columns"]))


def test_excused_switches_match_the_package(tmp_path):
    data = load_data("SEQdata")
    s = _fit("censoring", data, **EXCUSED)
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis("censoring", data, **EXCUSED), store, gold_fraction=0.2)
    # excused has denominator models only
    assert all(name.startswith("denominator") for name in c.record.models)
    for name, fit in c.fit_weights().items():
        assert _z(fit, _weight_model(s, name)) < 0.05
    ref = s.outcome_model[0]["outcome"]
    fit = c.fit()
    assert _z(fit, ref) < 0.1
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.03


@pytest.mark.parametrize(
    "dataset,method,opts",
    [
        ("SEQdata", "censoring", POST),
        ("SEQdata", "censoring", POST_EXCUSED),
        ("SEQdata_LTFU", "ITT", dict(POST, cense_colname="LTFU")),
    ],
)
def test_post_expansion_weights_match_the_package(tmp_path, dataset, method, opts):
    data = load_data(dataset)
    if dataset == "SEQdata_LTFU":
        data = data.filter(pl.col("ID") <= 500)
    s = _fit(method, data, **opts)
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis(method, data, **opts), store, gold_fraction=0.5)
    # weight table built from expanded rows
    assert c.record.X.shape[0] > data.height * 0.5
    for name, fit in c.fit_weights().items():
        assert _z(fit, _weight_model(s, name)) < 0.1
    ref = s.outcome_model[0]["outcome"]
    fit = c.fit()
    assert _z(fit, ref) < 0.1
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.03


@pytest.mark.parametrize("opts", [PRE, POST])
def test_percentile_truncation_matches_the_package(tmp_path, opts):
    data = load_data("SEQdata")
    opts = dict(opts, weight_p99=True)
    s = _fit("censoring", data, **opts)
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis("censoring", data, **opts), store, gold_fraction=0.5)
    # bounds are the package 1st and 99th percentiles
    stats = c.analysis.weight_stats
    for name in ("weight_p01", "weight_p99"):
        assert abs(stats[name][0] / s.weight_stats[name][0] - 1) < 0.01
    ref = s.outcome_model[0]["outcome"]
    fit = c.fit()
    assert _z(fit, ref) < 0.1
    assert np.abs(fit.bse / ref.bse - 1).max() < 0.03


def test_subgroups_match_the_package(tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    # equal width IDs avoid the package trial key collision
    data = load_data("SEQdata").with_columns(pl.col("ID") + 1000)
    opts = dict(subgroup_colname="sex")
    s = _analysis("ITT", data, bootstrap_nboot=3, **opts)
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    s.hazard()
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis("ITT", data, **opts), store, gold_fraction=0.5)
    assert c.outcome.models == ["outcome[0]", "outcome[1]"]

    def model(replicate, value):
        return s._offloader.load_model(s.outcome_model[replicate][value]["outcome"])

    # outcome model per subgroup
    for value in (0, 1):
        assert _z(c.fit(subgroup=value), model(0, value)) < 0.1

    # risks per subgroup at the package coefficients
    main = {
        f"outcome[{v}]": SimpleNamespace(
            params=pd.Series(np.asarray(model(0, v).params))
        )
        for v in (0, 1)
    }
    c.replicates = [
        (
            dict(counts),
            {f"outcome[{v}]": np.asarray(model(i + 1, v).params) for v in (0, 1)},
        )
        for i, counts in enumerate(s._boot_samples)
    ]
    out = c.survival(fits=main)
    keys = ["sex", "followup", "tx_init"]
    ref = s.km_data.filter(pl.col("estimate") == "risk")
    both = ref.join(
        out["km_data"].filter(pl.col("estimate") == "risk"), on=keys, suffix="_c"
    )
    assert both.height == ref.height
    for col in ("pred", "SE", "LCI", "UCI"):
        np.testing.assert_allclose(both[col], both[f"{col}_c"], rtol=0, atol=1e-10)
    for key in ("risk_difference", "risk_ratio"):
        want, got = s.risk_estimates[key], out[key]
        assert want.columns == got.columns
        for col in want.columns:
            np.testing.assert_allclose(want[col], got[col], rtol=0, atol=1e-10)

    # one simulated hazard ratio per subgroup
    hr = c.hazard(fits=main)
    assert hr["sex"].to_list() == s.hazard_ratio["sex"].to_list()
    assert (hr["LCI"] < hr["UCI"]).all()


def _same_tables(want, got, atol):
    assert want.columns == got.columns and want.height == got.height
    for col in want.columns:
        if want[col].dtype.is_numeric():
            np.testing.assert_allclose(
                want[col].to_numpy().astype(float),
                got[col].to_numpy().astype(float),
                rtol=0,
                atol=atol,
            )
        else:
            assert want[col].to_list() == got[col].to_list()


@pytest.mark.parametrize(
    "method,opts,atol",
    [
        # unweighted estimates are exact
        ("ITT", dict(), 1e-10),
        ("ITT", dict(subgroup_colname="sex"), 1e-10),
        # weighted estimates differ by the condensed weight models
        ("censoring", PRE, 5e-3),
        ("censoring", POST, 5e-3),
    ],
)
def test_end_of_followup_matches_the_package(tmp_path, method, opts, atol):
    data = load_data("SEQdata")
    opts = dict(opts, end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)
    s = _analysis(method, data, bootstrap_nboot=4, **opts)
    s.expand()
    s.bootstrap()
    s.fit()
    s.end_of_followup()
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis(method, data, **opts), store, gold_fraction=0.5)
    assert c.outcome is None
    with pytest.raises(ValueError, match="end_of_followup"):
        c.fit()

    # point estimates and counts
    point = c.end_of_followup()
    keep = [col for col in point["eof_data"].columns]
    _same_tables(s.eof_data.select(keep), point["eof_data"], atol)
    if not opts.get("weighted"):
        # intervals and contrasts on the package resamples
        out = c.end_of_followup(resamples=[dict(counts) for counts in s._boot_samples])
        _same_tables(s.eof_data, out["eof_data"], atol)
        _same_tables(s.eof_comparison, out["eof_comparison"], atol)


def test_condense_option_runs_the_pipeline():
    # equal width IDs avoid the package trial key collision
    data = load_data("SEQdata").with_columns(pl.col("ID") + 1000)
    opts = dict(PRE, km_curves=True)
    full = _fit("censoring", data, **opts)
    full.survival()

    s = _analysis("censoring", data, condense=True, bootstrap_nboot=4, **opts)
    s.expand()
    assert s.DT is None
    s.bootstrap()
    s.fit()
    s.survival()
    s.hazard()
    assert _z(s.outcome_model[0]["outcome"], full.outcome_model[0]["outcome"]) < 0.1
    keys = ["estimate", "tx_init", "followup"]
    both = full.km_data.join(s.km_data, on=keys, suffix="_c")
    assert (both["pred"] - both["pred_c"]).abs().max() < 1e-3
    assert {"SE", "LCI", "UCI"} <= set(s.km_data.columns)
    assert s.hazard_ratio["LCI"][0] < s.hazard_ratio["UCI"][0]
    out = s.collect()
    assert out.km_data is s.km_data
    assert set(out.diagnostic_tables) == set(full.diagnostics)
    assert out.risk_difference.height > 0
    # weight models and weight statistics are reported as by the package
    assert len(out.numerator_models) == len(out.denominator_models) == 2
    for name in ("weight_min", "weight_mean", "weight_std", "weight_p50", "weight_max"):
        ratio = out.weight_statistics[name][0] / full.weight_stats[name][0]
        assert abs(ratio - 1) < 0.02


def test_condense_option_end_of_followup():
    data = load_data("SEQdata")
    opts = dict(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)
    full = _analysis("ITT", data, **opts)
    full.expand()
    full.fit()
    full.end_of_followup()
    s = _analysis("ITT", data, condense=True, **opts)
    s.expand()
    s.fit()
    s.end_of_followup()
    np.testing.assert_allclose(
        s.eof_data["Proportion"], full.eof_data["Proportion"], rtol=0, atol=1e-10
    )


def test_trial_selection(tmp_path):
    data = load_data("SEQdata")
    store = RecordStore.write(data, "ID", tmp_path)

    # first trial only is the package expansion batch by batch
    opts = dict(selection_first_trial=True)
    s = _fit("ITT", data, **opts)
    c = condense(_analysis("ITT", data, **opts), store, gold_fraction=0.5)
    ref, fit = s.outcome_model[0]["outcome"], c.fit()
    # trial terms are all zero here so skip them
    real = (ref.bse > 1e-8).to_numpy()
    z = np.abs((fit.params - ref.params) / ref.bse).to_numpy()[real]
    assert z.max() < 0.1

    # random selection keeps every treated trial and a share of the controls
    opts = dict(selection_random=True, selection_sample=0.5)
    everyone = _expanded(_analysis("ITT", data), data).DT
    a = _analysis("ITT", data, **opts)
    kept = _expanded(a, data).DT
    trials = ["ID", "trial"]

    def count(DT, treated):
        return DT.filter(pl.col("tx_init_bas") == treated).select(trials).n_unique()

    assert count(kept, 1) == count(everyone, 1)
    # diagnostics count the kept trials
    diag = _expanded(a, data).diagnostics["nonunique_followup"]
    assert diag["len"].sum() == kept.height
    assert abs(count(kept, 0) / count(everyone, 0) - 0.5) < 0.05
    # the same trials are kept whatever the batches
    batches = pl.concat([_expanded(a, r).DT for r in store.batches(40)])
    assert batches.select(trials).unique().sort(trials).equals(
        kept.select(trials).unique().sort(trials)
    )
    # and the condensed fit is the fit on those rows
    c = condense(a, store, gold_fraction=0.5)
    rows = c.exact(store).outcome
    exact = _glm(rows.X, rows.Y[:, 0], rows.w)
    fit = c.fit()
    z = np.abs((fit.params.to_numpy() - exact.params) / exact.bse).max()
    assert z < 0.1


@pytest.mark.parametrize("opts", [PRE, POST])
def test_three_treatment_levels_match_the_package(tmp_path, opts):
    import pandas as pd

    # equal width IDs avoid the package trial key collision
    data = _three_levels(load_data("SEQdata")).with_columns(pl.col("ID") + 1000)
    opts = dict(opts, treatment_level=[0, 1, 2])
    s = _fit("censoring", data, **opts)
    s.survival()
    s.hazard()
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(_analysis("censoring", data, **opts), store, gold_fraction=0.5)
    assert all(m.classes == [0, 1, 2] for m in c.record.block.models)

    # multinomial weight models stacked class by class
    for name, fit in c.fit_weights().items():
        ref = _package_params(s, name)
        bse = _weight_model(s, name).bse.to_numpy().T.ravel()
        assert np.abs((fit.params[ref.index] - ref) / bse).max() < 0.1
    assert _z(c.fit(), s.outcome_model[0]["outcome"]) < 0.1

    keys = ["followup", "tx_init"]
    risk = c.survival()["km_data"].filter(pl.col("estimate") == "risk")
    ref = s.km_data.filter(pl.col("estimate") == "risk")
    both = ref.join(risk, on=keys, suffix="_c")
    assert both.height == ref.height
    assert (both["pred"] - both["pred_c"]).abs().max() < 5e-3

    # one simulated hazard ratio over the three arms as the package codes them
    log_hr = np.log([c.hazard(seed=i)["Hazard ratio"][0] for i in range(12)])
    package = np.log(s.hazard_ratio["Hazard ratio"][0])
    assert abs(package - log_hr.mean()) < 4 * log_hr.std(ddof=1)

    # corrected with everyone the weight models are the exact ones
    fit = c.fit_corrected(sample=c.exact(store))
    for name, wf in fit.weights.items():
        ref = _package_params(s, name)
        bse = _weight_model(s, name).bse.to_numpy().T.ravel()
        assert np.abs((wf.params[ref.index] - ref) / bse).max() < 1e-4
    assert _z(fit, s.outcome_model[0]["outcome"]) < 0.02


def _no_one_eligible_in_the_first_batch(data, n):
    first = data["ID"].unique().sort()[:n].implode()
    return data.with_columns(
        pl.when(pl.col("ID").is_in(first))
        .then(0)
        .otherwise(pl.col("eligible"))
        .alias("eligible")
    )


@pytest.mark.parametrize(
    "method, opts",
    [
        ("censoring", dict(PRE, weight_p99=True)),
        (
            "censoring",
            dict(PRE, end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3),
        ),
    ],
)
def test_batches_without_eligible_people_are_skipped(tmp_path, method, opts):
    data = _no_one_eligible_in_the_first_batch(load_data("SEQdata"), 50)
    store = RecordStore.write(data, "ID", tmp_path)
    c = condense(
        _analysis(method, data, **opts), store, gold_fraction=0.5, people_per_batch=50
    )
    assert c.weight_stats["weight_max"][0] >= c.weight_stats["weight_min"][0] > 0
    if c.outcome is not None:
        assert c.outcome.ids.min() > 50
        # the fit is the package's on the same data
        ref = _fit(method, data, **opts).outcome_model[0]["outcome"]
        assert _z(c.fit(), ref) < 0.1


def test_ci_settings_changed_after_fit_are_used():
    data = load_data("SEQdata").with_columns(pl.col("ID") + 1000)
    s = _analysis("ITT", data, condense=True, bootstrap_nboot=4, km_curves=True)
    s.expand()
    s.bootstrap()
    s.fit()

    def widths():
        s.survival()
        s.hazard()
        risk = s.km_data.filter(pl.col("estimate") == "risk")
        return (risk["UCI"] - risk["LCI"]).max(), (
            s.hazard_ratio["UCI"][0] - s.hazard_ratio["LCI"][0]
        )

    wide = widths()
    s.bootstrap_CI = 0.5
    narrow = widths()
    assert narrow[0] < wide[0] and narrow[1] < wide[1]


@pytest.mark.parametrize("offload", [False, True])
def test_condensed_record_store_is_removed_with_the_analysis(tmp_path, offload):
    s = _analysis(
        "ITT",
        load_data("SEQdata"),
        condense=True,
        offload=offload,
        offload_dir=str(tmp_path / "models"),
    )
    s.expand()
    path = s._store.path
    assert path.exists()
    # offloaded analyses keep their files under offload_dir
    assert path.is_relative_to(tmp_path) == offload
    del s
    gc.collect()
    assert not path.exists()


def test_too_many_models_for_one_block_is_an_error():
    frame = pl.DataFrame({"ID": [1, 2], "y": [0, 1], "x": [0.5, 1.5]}).to_pandas()
    models = [_Model(f"m{i}", "y", "x") for i in range(64)]
    with pytest.raises(ValueError, match="63"):
        _Block(models, frame, {}, "ID")


def test_weighted_cox_from_counts_is_the_weighted_cox_fit():
    import pandas as pd
    from lifelines import CoxPHFitter

    from pySEQTarget.condense._estimands import _cox

    rng = np.random.default_rng(1)
    n = 4000
    arm = rng.integers(0, 2, n)
    w = np.where(rng.random(n) < 0.5, 2.0, 1.0)
    time = np.minimum(rng.geometric(0.05 * np.exp(0.4 * arm)) - 1, 30)
    event = (time < 30) & (rng.random(n) < 0.9)
    events, counts, kept = np.zeros((3, 2, 31))
    np.add.at(events, (arm[event], time[event]), w[event])
    np.add.at(counts, (arm[event], time[event]), 1)
    np.add.at(kept, (arm, time), w)
    at_risk = kept[:, ::-1].cumsum(1)[:, ::-1]

    frame = pd.DataFrame({"t": time, "e": event.astype(int), "arm": arm, "w": w})
    cph = CoxPHFitter().fit(frame, "t", "e", weights_col="w")
    beta = _cox(events, at_risk, [0, 1], counts)
    assert abs(beta - cph.params_.iloc[0]) < 1e-6


def test_random_selection_standardizes_over_every_trial(tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    # equal width IDs avoid the package trial key collision
    data = load_data("SEQdata").with_columns(pl.col("ID") + 1000)
    opts = dict(selection_random=True, selection_sample=0.3, km_curves=True)
    s = _analysis("ITT", data, **opts)
    s.expand()
    s.fit()
    store = RecordStore.write(data, "ID", tmp_path)
    a = _analysis("ITT", data, **opts)
    c = condense(a, store, gold_fraction=0.2)
    # the package standardizing over the trials the condensed run kept
    s.DT = _expanded(a, data).DT
    s.survival()

    params = np.asarray(s.outcome_model[0]["outcome"].params)
    main = {"outcome": SimpleNamespace(params=pd.Series(params))}
    risk = c.survival(fits=main)["km_data"].filter(pl.col("estimate") == "risk")
    ref = s.km_data.filter(pl.col("estimate") == "risk")
    both = ref.join(risk, on=["followup", "tx_init"], suffix="_c")
    assert both.height == ref.height
    np.testing.assert_allclose(both["pred"], both["pred_c"], rtol=0, atol=1e-10)
    hr = c.hazard(fits=main)["Hazard ratio"][0]
    assert np.isfinite(hr) and hr > 0


def _continuous_outcome(data):
    rng = np.random.default_rng(42)
    return data.with_columns(
        (
            10
            + 2 * pl.col("tx_init")
            + pl.col("N")
            + pl.Series(rng.standard_normal(data.height))
        ).alias("outcome")
    )


EOF_OPTS = dict(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)

# value columns of the diagnostic tables
_COUNT_COLUMNS = {
    "len",
    "N",
    "Mean",
    "SD",
    "Eligible",
    "At k",
    "In window",
    "Excluded (outside window)",
    "Excluded (no measurement)",
}


@pytest.mark.parametrize(
    "dataset, method, opts",
    [
        ("SEQdata", "censoring", PRE),
        ("SEQdata_LTFU", "ITT", dict(compevent_colname="LTFU")),
        ("SEQdata", "censoring", dict(PRE, **EOF_OPTS)),
        ("continuous", "ITT", dict(EOF_OPTS, end_of_fup_type="continuous")),
    ],
)
def test_diagnostics_are_the_package_tables(tmp_path, dataset, method, opts):
    if dataset == "continuous":
        data = _continuous_outcome(load_data("SEQdata"))
    else:
        data = load_data(dataset).filter(pl.col("ID") <= 300)
    s = _fit(method, data, **opts)
    store = RecordStore.write(data, "ID", tmp_path)
    # small batches so every table is summed over several
    c = condense(
        _analysis(method, data, **opts), store, gold_fraction=0.5, people_per_batch=40
    )
    assert set(c.diagnostics) == set(s.diagnostics)
    for name, want in s.diagnostics.items():
        by = [col for col in want.columns if col not in _COUNT_COLUMNS]
        _same_tables(want.sort(by), c.diagnostics[name].sort(by), 1e-10)


def test_condense_with_expand_only_is_an_error():
    with pytest.raises(ValueError, match="expand_only"):
        _analysis("ITT", load_data("SEQdata"), condense=True, expand_only=True)
