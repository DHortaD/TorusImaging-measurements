from functools import partial

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import torusimaging as oti
from astropy.constants import G
from gala.units import galactic
from torusimaging.data import OTIData


def label_func_base(r, label_vals, knots):
    return oti.model_helpers.monotonic_quadratic_spline(knots, label_vals, r)


def e_func_base(r_e, vals, sign, knots):
    return sign * oti.model_helpers.monotonic_quadratic_spline(
        knots, jnp.concatenate((jnp.array([0.0]), jnp.exp(vals))), r_e
    )


def regularization_func_base(
    params,
    e_funcs: dict,
    e_knots: dict,
    e_sigmas: dict,
    e_smooth_sigmas: dict,
    label_func,
    label_knots,
    label_sigma: float,
    label_smooth_sigma: float,
):
    p = 0.0

    # L2 regularization to keep the value of the functions small:
    for m, func in e_funcs.items():
        p += jnp.sum((func(e_knots[m], **params["e_params"][m]) / e_sigmas[m]) ** 2)

    p += jnp.sum((label_func(label_knots, **params["label_params"]) / label_sigma) ** 2)

    # L2 regularization for smoothness:
    for m in params["e_params"]:
        diff = params["e_params"][m]["vals"][1:] - params["e_params"][m]["vals"][:-1]
        p += jnp.sum((diff / e_smooth_sigmas[m]) ** 2)

    diff = (
        params["label_params"]["label_vals"][2:]
        - params["label_params"]["label_vals"][1:-1]
    )
    p += jnp.sum((diff / label_smooth_sigma) ** 2)

    return p


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


class SplineLabelModelWrapper:
    def __init__(
        self,
        r_e_max: float,
        label_n_knots: int,
        label0_bounds: tuple,
        label_grad_sign: float,
        label_regularize_sigma: float,
        label_smooth_sigma: float,
        e_n_knots: dict,
        e_knots_scale=None,
        e_bounds=None,
        e_signs=None,
        e_regularize_sigmas=None,
        e_smooth_sigmas=None,
        unit_sys=galactic,
        # regularize=True,
        label_model_kwargs=None,
        pos0_bounds=(-0.5, 0.5),  # 500 pc
        vel0_bounds=(-0.1, 0.1),  # ~100 km/s
    ):
        self.unit_sys = unit_sys

        # ------------------------------------------------------------------------------
        # Set up the label function bits:

        # Knot locations, spaced equally in r_z
        self.label_knots = jnp.linspace(0, r_e_max, label_n_knots)
        label_func = partial(label_func_base, knots=self.label_knots)

        if label_grad_sign > 0:
            label_func_bounds = {
                "label_vals": (
                    np.concatenate(
                        ([label0_bounds[0]], jnp.full(label_n_knots - 1, 0.0))
                    ),
                    np.concatenate(
                        ([label0_bounds[1]], jnp.full(label_n_knots - 1, 10.0))
                    ),
                )
            }
        else:
            label_func_bounds = {
                "label_vals": (
                    np.concatenate(
                        ([label0_bounds[0]], jnp.full(label_n_knots - 1, -10.0))
                    ),
                    np.concatenate(
                        ([label0_bounds[1]], jnp.full(label_n_knots - 1, 0.0))
                    ),
                )
            }

        # ------------------------------------------------------------------------------
        # Set up the e function components:
        if e_knots_scale is None:
            e_knots_scale = {}
        for m in e_n_knots:
            e_knots_scale.setdefault(m, (lambda x: x, lambda x: x))

        self.e_n_knots = e_n_knots
        self.e_knots = {
            m: e_knots_scale[m][0](jnp.linspace(0, e_knots_scale[m][1](r_e_max), n))
            for m, n in e_n_knots.items()
        }
        if e_signs is None:
            e_signs = {m: (-1.0 if (m / 2) % 2 == 0 else 1.0) for m in self.e_knots}

        e_funcs = {
            m: partial(e_func_base, sign=e_signs[m], knots=self.e_knots[m])
            for m in self.e_knots
        }

        if e_bounds is None:
            e_bounds = {}

        for m, n in self.e_n_knots.items():
            # TODO: hard-set numbers
            e_bounds.setdefault(
                m, {"vals": (jnp.full(n - 1, -15.0), jnp.full(n - 1, 1.5))}
            )

        if e_regularize_sigmas is None:
            # Default value of L2 regularization stddev:
            e_regularize_sigmas = {m: 0.1 for m in self.e_knots}

        # ------------------------------------------------------------------------------
        # Setup the regularization function:
        reg_func = partial(
            regularization_func_base,
            e_funcs=e_funcs,
            e_knots=self.e_knots,
            e_sigmas=e_regularize_sigmas,
            e_smooth_sigmas=e_smooth_sigmas,
            label_func=label_func,
            label_knots=self.label_knots,
            label_sigma=label_regularize_sigma,
            label_smooth_sigma=label_smooth_sigma,
        )

        if label_model_kwargs is None:
            label_model_kwargs = {}

        self.label_model = oti.LabelOrbitModel(
            label_func=label_func,
            e_funcs=e_funcs,
            regularization_func=reg_func,
            unit_sys=self.unit_sys,
            **label_model_kwargs,
        )

        self._bounds = {}

        # Reasonable bounds for the midplane density
        dens0_bounds = [0.01, 10] * u.Msun / u.pc**3
        self._bounds["ln_Omega"] = 0.5 * np.log(
            (4 * np.pi * G * dens0_bounds).decompose(self.unit_sys).value
        )
        self._bounds["pos0"] = pos0_bounds
        self._bounds["vel0"] = vel0_bounds
        self._bounds["e_params"] = e_bounds
        self._bounds["label_params"] = label_func_bounds

    def get_init_params(self, oti_data, label_name=None):
        if label_name is None:
            if len(oti_data.labels) == 1:
                label_name = list(oti_data.labels.keys())[0]
            else:
                raise ValueError("must specify label_name")

        label = oti_data.labels[label_name]

        params0 = self.label_model.get_params_init(oti_data.pos, oti_data.vel, label)
        r_e, _ = self.label_model.get_elliptical_coords(
            oti_data._pos, oti_data._vel, params0
        )

        params0["e_params"] = {
            m: {"vals": jnp.full(self.e_n_knots[m] - 1, -10)} for m in self.e_knots
        }

        # Estimate the label value near r_e = 0 and slopes for knot values:
        r1, r2 = np.nanpercentile(r_e, [10, 90])
        label0 = np.nanmean(label[r_e <= r1])
        label_slope = (np.nanmedian(label[r_e >= r2]) - label0) / (r2 - r1)

        params0["label_params"] = {
            "label_vals": np.concatenate(
                (
                    [label0],
                    np.full(len(self.label_knots) - 1, label_slope),
                )
            )
        }

        return params0

    def run(
        self,
        oti_data,
        bins,
        p0=None,
        label_name=None,
        label_err_floor=0.0,
        jaxopt_kw=None,
    ):
        if jaxopt_kw is None:
            jaxopt_kw = {}
        jaxopt_kw.setdefault("tol", 1e-12)

        if isinstance(oti_data, OTIData):
            bdata, label_name = oti_data.get_binned_label(bins, label_name=label_name)
            bdata[f"{label_name}_err"] = np.sqrt(
                label_err_floor**2 + bdata[f"{label_name}_err"] ** 2
            )
        else:
            bdata = oti_data

        if p0 is None:
            if not isinstance(oti_data, OTIData):
                raise ValueError(
                    "If not passing in initial parameter values, you must pass in an "
                    "OTIData instance for `oti_data`"
                )
            p0 = self.get_init_params(oti_data, label_name=label_name)

        # First check that objective evaluates to a finite value:
        mask = (
            np.isfinite(bdata[label_name])
            & np.isfinite(bdata[f"{label_name}_err"])
            & (bdata[f"{label_name}_err"] > 0)
        )
        data = dict(
            pos=bdata["pos"].decompose(self.unit_sys).value[mask],
            vel=bdata["vel"].decompose(self.unit_sys).value[mask],
            label=bdata[label_name][mask],
            label_err=bdata[f"{label_name}_err"][mask],
        )
        test_val = self.label_model.objective(p0, **data)
        if not np.isfinite(test_val):
            raise RuntimeError("Objective function evaluated to non-finite value")

        res = self.label_model.optimize(
            params0=p0, bounds=self._bounds, jaxopt_kwargs=jaxopt_kw, **data
        )

        return bdata, res

    def run_mcmc(
        self,
        oti_data,
        bins=None,
        p0=None,
        label_name=None,
        label_err_floor=0.0,
        rng_seed=0,
        num_steps=1000,
        num_warmup=1000,
    ):
        import blackjax

        if isinstance(oti_data, OTIData):
            bdata, label_name = oti_data.get_binned_label(bins, label_name=label_name)
            bdata[f"{label_name}_err"] = np.sqrt(
                label_err_floor**2 + bdata[f"{label_name}_err"] ** 2
            )
        else:
            bdata = oti_data

        if p0 is None:
            if not isinstance(oti_data, OTIData):
                raise ValueError(
                    "If not passing in initial parameter values, you must pass in an "
                    "OTIData instance for `oti_data`"
                )
            p0 = self.get_init_params(oti_data, label_name=label_name)

        # First check that objective evaluates to a finite value:
        mask = (
            np.isfinite(bdata[label_name])
            & np.isfinite(bdata[f"{label_name}_err"])
            & (bdata[f"{label_name}_err"] > 0)
        )
        data = dict(
            pos=bdata["pos"].decompose(self.unit_sys).value[mask],
            vel=bdata["vel"].decompose(self.unit_sys).value[mask],
            label=bdata[label_name][mask],
            label_err=bdata[f"{label_name}_err"][mask],
        )
        test_val = self.label_model.objective(p0, **data)
        if not np.isfinite(test_val):
            raise RuntimeError("Objective function evaluated to non-finite value")

        lb, ub = self.label_model.unpack_bounds(self._bounds)
        lb_arrs = jax.tree_util.tree_flatten(lb)[0]
        ub_arrs = jax.tree_util.tree_flatten(ub)[0]

        def logprob(p):
            lp = 0.0
            pars, treedef = jax.tree_util.tree_flatten(p)
            for i in range(len(pars)):
                lp += jnp.where(
                    jnp.any(pars[i] < lb_arrs[i]) | jnp.any(pars[i] > ub_arrs[i]),
                    -jnp.inf,
                    0.0,
                )

            lp += self.label_model.ln_label_likelihood(p, **data)

            lp -= self.label_model.regularization_func(p)

            return lp

        rng_key = jax.random.PRNGKey(rng_seed)
        warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
        (state, parameters), _ = warmup.run(rng_key, p0, num_steps=num_warmup)

        kernel = blackjax.nuts(logprob, **parameters).step
        states = inference_loop(rng_key, kernel, state, num_steps)

        # Get the pytree structure of a single sample based on the starting point:
        treedef = jax.tree_util.tree_structure(p0)
        arrs, _ = jax.tree_util.tree_flatten(states.position)

        mcmc_samples = []
        for n in range(arrs[0].shape[0]):
            mcmc_samples.append(
                jax.tree_util.tree_unflatten(treedef, [arr[n] for arr in arrs])
            )

        return states, mcmc_samples