
# general
import copy
import os
from astropy.constants import G
import astropy.table as at
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from astropy.io import fits
import astropy.units as u

# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
import gala.integrate as gi
from gala.units import galactic

# jax
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from torusimaging import LabelOrbitModel

# from empaf import DensityOrbitModel
from torusimaging.plot import plot_data_models_residual
from torusimaging.model_helpers import generalized_logistic_func_alt
from torusimaging import LabelOrbitModel
# from torusimaging.plot import plot_data_models_label_residual


def plot_data_model_residual(
    model,
    bdata,
    params,
    zlim,
    vzlim=None,
    aspect=True,
    residual_sigma_lim=3.0,
    subplots_kwargs=None,
    suptitle1="",
    suptitle2="",
):
    title_fontsize = 20
    title_pad = 10

    cb_labelsize = 16
    mgfe_cbar_xlim = (0, 0.15)
    mgfe_cbar_vlim = (-0.05, 0.18)

    tmp_aaf = model.compute_action_angle(
        np.atleast_1d(zlim) * 0.75, [0.0] * u.km / u.s, params
    )
    Omega = tmp_aaf["Omega"][0]
    if vzlim is None:
        vzlim = zlim * Omega
    vzlim = vzlim.to_value(u.km / u.s, u.dimensionless_angles())

    if subplots_kwargs is None:
        subplots_kwargs = dict()
    subplots_kwargs.setdefault("figsize", (16, 4.2))
    subplots_kwargs.setdefault("sharex", True)
    subplots_kwargs.setdefault("sharey", True)
    subplots_kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(1, 3, **subplots_kwargs)

    cs = axes[0].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        bdata["label"],
        cmap="magma",
        rasterized=True,
        vmin=-0.02,
        vmax=0.15,
    )
    cb = fig.colorbar(cs, ax=axes[0:1])
    cb.set_label("mean [Mg/Fe]", fontsize=cb_labelsize)
    cb.ax.set_ylim(mgfe_cbar_xlim)
    cb.ax.set_yticks(np.arange(mgfe_cbar_xlim[0], mgfe_cbar_xlim[1] + 1e-3, 0.05))
    cb.ax.yaxis.set_tick_params(labelsize=14)

    model_mgfe = np.array(model._get_label(bdata["pos"], bdata["vel"], params))
    cs = axes[1].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        model_mgfe,
        cmap="magma",
        rasterized=True,
        vmin=mgfe_cbar_vlim[0],
        vmax=mgfe_cbar_vlim[1],
    )

    cs = axes[2].pcolormesh(
        bdata["vel"].to_value(u.km / u.s),
        bdata["pos"].to_value(u.kpc),
        (bdata["label"] - model_mgfe) / bdata["label_err"] / np.sqrt(2),
        cmap="coolwarm",
        vmin=-residual_sigma_lim,
        vmax=residual_sigma_lim,
        rasterized=True,
    )
    cb = fig.colorbar(cs, ax=axes[2])  # , orientation="horizontal")
    cb.set_label("(data $-$ model) / error", fontsize=cb_labelsize)
    cb.ax.yaxis.set_tick_params(labelsize=14)

    # Titles
    axes[0].set_title("Label Data", fontsize=title_fontsize, pad=title_pad)
    axes[1].set_title("OTI Model", fontsize=title_fontsize, pad=title_pad)
    axes[2].set_title("Residuals", fontsize=title_fontsize, pad=title_pad)
    fig.suptitle(f"{suptitle1} {suptitle2}", fontsize=24)

    # Labels
    axes[0].set_ylabel(f"$z$ [{u.kpc:latex_inline}]")
    for ax in axes:
        ax.set_xlabel(f"$v_z$ [{u.km/u.s:latex_inline}]")

    # Ticks
    if vzlim >= 100:
        axes[0].set_xticks(np.arange(-300, 300 + 1, 100))
        axes[0].set_xticks(np.arange(-300, 300 + 1, 50), minor=True)
    else:
        axes[0].set_xticks(np.arange(-300, 300 + 1, 50))

    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 1))
    axes[1].set_yticks(np.arange(-3, 3 + 1e-3, 0.5), minor=True)

    if aspect:
        aspect_val = Omega.to_value(u.km / u.s / u.kpc, u.dimensionless_angles())

    for ax in axes:
        if aspect:
            ax.set_aspect(aspect_val)
        ax.set_xlim(-vzlim, vzlim)
        ax.set_ylim(-zlim.to_value(u.kpc), zlim.to_value(u.kpc))

    return fig, axes




def fit_bins_pos(
    z,
    vz,
    label,
    n_label_knots = 5,
    n_e2_knots = 4,
    n_e4_knots = 3,
    binlength=91
    ):
    
    tbl = at.QTable()
    tbl['z'] = z * u.kpc
    tbl['vz'] = vz * u.kpc / u.Gyr
    tbl['label'] = label
    
    # bin the data
#     data_ = LabelOrbitModel.get_data_im(
#     z=tbl['z'].decompose(galactic).value,
#     vz=tbl['vz'].decompose(galactic).value,
#     label=tbl['label'],
#     bins={"z": np.linspace(-2., 2., binlength), "vz": np.linspace(-0.08, 0.08, binlength)},
#     )


    # define the label function and the number of knots
#     n_label_knots = 9
#     def label_func(rz, label_vals):
#         # Knot locations, spaced equally in sqrt(r_z)
#         xs = jnp.linspace(0, 1.0, n_label_knots) ** 2

#         spl = InterpolatedUnivariateSpline(xs, label_vals, k=2)
#         return spl(rz)

    def label_func(rz, label_vals):
        # for the funciton to be monotonic 
        # Knot locations, spaced equally in sqrt(r_z)
        xs = jnp.linspace(0, jnp.sqrt(0.6), n_label_knots) ** 2

        return monotonic_quadratic_spline(xs, label_vals, rz)

    # define the bounds for the knots
    # TO DO: make the bounds configurable at the function level. I.e., make this applicable to all elements regardless of magnitude direction 
    label_bounds = {
    "label_vals": (
        np.concatenate(([-5.],np.full(n_label_knots-1, -0.01))), #MAGIC number (to stop the gradients going to 0: APW to check this)
        np.concatenate(([5.],np.full(n_label_knots-1, 5.0)))
#         jnp.full(n_label_knots, -5.0),
#         jnp.full(n_label_knots, 5.0)
        )
    }
#     label_bounds = in_label_bounds

    # create the functions for the e2 and e4 parameters
    from empaf.model_helpers import monotonic_quadratic_spline

    def e2_func(rzp, e2_vals):
        e2_knots = jnp.linspace(0, jnp.sqrt(0.6), n_e2_knots) ** 2
        vals = monotonic_quadratic_spline(
            e2_knots, jnp.concatenate((jnp.array([0.0]), e2_vals)), rzp
        )
        return vals


    def e4_func(rzp, e4_vals):
        e4_knots = jnp.linspace(0, jnp.sqrt(0.6), n_e4_knots) ** 2
        vals = monotonic_quadratic_spline(
            e4_knots, jnp.concatenate((jnp.array([0.0]), e4_vals)), rzp
        )
        return -vals
    
    
    # set up the internal model functions
    e_params0 = {}
    e_bounds = {}
    # e_params0[2] = {"e2_vals": np.full(n_e2_knots - 1, 0.2)}
    e_params0[2] = {"e2_vals": np.linspace(1.5, 0.2, n_e2_knots - 1) / 0.6 * 0.2}
    e_params0[4] = {"e4_vals": np.full(n_e4_knots - 1, 0.08)}
    e_bounds[2] = {"e2_vals": (np.full(n_e2_knots-1, 0), np.full(n_e2_knots-1, 10))}
    e_bounds[4] = {"e4_vals": (np.full(n_e4_knots-1, 0), np.full(n_e4_knots-1, 10))}
    
    # run the label model
    label_model = LabelOrbitModel(
    label_func=label_func,
    e_funcs={2: e2_func, 4: e4_func},
    unit_sys=galactic,
    )

#     # get the parameters for the initial fit
#     label_params0 = label_model.get_params_init(
#         vz=data_["vz"] * u.kpc/u.Myr, z=data_["z"] * u.kpc, label=data_['label'],
#         label_params0={"label_vals": np.zeros(n_label_knots)}
#     )
    # in the new version of the code (log ems one), the "get_data_im" function is not necessary, as the "get_params_init"
    # already bins the data within it
    
        # get the parameters for the initial fit
    label_params0 = label_model.get_params_init(
        pos=z * u.kpc, vel=vz* u.kpc/u.Myr, label=label,
        label_params0={"label_vals": np.zeros(n_label_knots)}
    )

    label_params0['e_params'] = e_params0
    
    
    # define the bounds for the final model fit
    label_model_bounds = {}

    _dens0 = [0.01, 2] * u.Msun / u.pc**3
    label_model_bounds["ln_Omega"] = np.log(np.sqrt(_dens0 * 4 * np.pi * G).to_value(1 / u.Myr))
    label_model_bounds["z0"] = (-0.3, 0.3)
    label_model_bounds["vz0"] = (-0.01, 0.01)

    label_model_bounds["e_params"] = e_bounds
    label_model_bounds["label_params"] = label_bounds
        
    label_model.objective(label_params0, z*u.kpc, vz*u.kpc/u.Myr)

    clean_mask = np.isfinite(label) 
#     & np.isfinite(data_['label_err'])
    clean_label_data = {k: v[clean_mask] for k, v in data_.items()}
    
    # optimise the model
    label_res = label_model.optimize(
    params0=label_params0,
    bounds=label_model_bounds,
    **clean_label_data
    )
    label_res.state
    
    
    return label_model, label_params0, label_res.params

def fit_bins_neg(
    z,
    vz,
    label,
    n_label_knots = 5,
    n_e2_knots = 4,
    n_e4_knots = 3,
    binlength=91):
    
    tbl = at.QTable()
    tbl['z'] = z * u.kpc
    tbl['vz'] = vz * u.kpc / u.Gyr
    tbl['label'] = label
    
    # bin the data
    data_ = LabelOrbitModel.get_data_im(
    z=tbl['z'].decompose(galactic).value,
    vz=tbl['vz'].decompose(galactic).value,
    label=tbl['label'],
    bins={"z": np.linspace(-2., 2., binlength), "vz": np.linspace(-0.08, 0.08, binlength)},
    )


    # define the label function and the number of knots
#     n_label_knots = 9
#     def label_func(rz, label_vals):
#         # Knot locations, spaced equally in sqrt(r_z)
#         xs = jnp.linspace(0, 1.0, n_label_knots) ** 2

#         spl = InterpolatedUnivariateSpline(xs, label_vals, k=2)
#         return spl(rz)

    def label_func(rz, label_vals):
        # for the funciton to be monotonic 
        # Knot locations, spaced equally in sqrt(r_z)
        xs = jnp.linspace(0, jnp.sqrt(0.6), n_label_knots) ** 2

        return monotonic_quadratic_spline(xs, label_vals, rz)

    # define the bounds for the knots
    # TO DO: make the bounds configurable at the function level. I.e., make this applicable to all elements regardless of magnitude direction 
    label_bounds = {
    "label_vals": (
        np.concatenate(([-5.],np.full(n_label_knots-1, -5.))),
        np.concatenate(([5.],np.full(n_label_knots-1, -0.01))) # MAGIC number (to stop the gradients going to 0: APW to check this)
#         jnp.full(n_label_knots, -5.0),
#         jnp.full(n_label_knots, 5.0)
        )
    }

    # create the functions for the e2 and e4 parameters
    from empaf.model_helpers import monotonic_quadratic_spline

    def e2_func(rzp, e2_vals):
        e2_knots = jnp.linspace(0, jnp.sqrt(0.6), n_e2_knots) ** 2
        vals = monotonic_quadratic_spline(
            e2_knots, jnp.concatenate((jnp.array([0.0]), e2_vals)), rzp
        )
        return vals


    def e4_func(rzp, e4_vals):
        e4_knots = jnp.linspace(0, jnp.sqrt(0.6), n_e4_knots) ** 2
        vals = monotonic_quadratic_spline(
            e4_knots, jnp.concatenate((jnp.array([0.0]), e4_vals)), rzp
        )
        return -vals
    
    
    # set up the internal model functions
    e_params0 = {}
    e_bounds = {}
    # e_params0[2] = {"e2_vals": np.full(n_e2_knots - 1, 0.2)}
    e_params0[2] = {"e2_vals": np.linspace(1.5, 0.2, n_e2_knots - 1) / 0.6 * 0.2}
    e_params0[4] = {"e4_vals": np.full(n_e4_knots - 1, 0.08)}
    e_bounds[2] = {"e2_vals": (np.full(n_e2_knots-1, 0), np.full(n_e2_knots-1, 10))}
    e_bounds[4] = {"e4_vals": (np.full(n_e4_knots-1, 0), np.full(n_e4_knots-1, 10))}
    
    # run the label model
    label_model = LabelOrbitModel(
    label_func=label_func,
    e_funcs={2: e2_func, 4: e4_func},
    unit_sys=galactic,
    )

    # get the parameters for the initial fit
    label_params0 = label_model.get_params_init(
        vz=data_["vz"] * u.kpc/u.Myr, z=data_["z"] * u.kpc, label=data_['label'],
        label_params0={"label_vals": np.zeros(n_label_knots)}
    )

    label_params0['e_params'] = e_params0
    
    
    # define the bounds for the final model fit
    label_model_bounds = {}

    _dens0 = [0.01, 2] * u.Msun / u.pc**3
    label_model_bounds["ln_Omega"] = np.log(np.sqrt(_dens0 * 4 * np.pi * G).to_value(1 / u.Myr))
    label_model_bounds["z0"] = (-0.05, 0.05)
    label_model_bounds["vz0"] = (-0.01, 0.01)

    label_model_bounds["e_params"] = e_bounds
    label_model_bounds["label_params"] = label_bounds
    
    
    label_model.objective(params=label_params0, **data_)

    clean_mask = np.isfinite(data_['label']) & np.isfinite(data_['label_err'])
    clean_label_data = {k: v[clean_mask] for k, v in data_.items()}
    
    # optimise the model
    label_res = label_model.optimize(
    params0=label_params0,
    bounds=label_model_bounds,
    **clean_label_data
    )
    label_res.state
    
    
    return data_, label_model, label_params0, label_res.params



from astropy.convolution import Gaussian2DKernel, convolve
def plot_data_models_label_residual(
    data_H,
    model,
    params_init,
    params_fit,
    smooth_residual=None,
    vlim=None,
    vlim_residual=0.02,
    usys=None,
    mask_no_data=True,
    vmin=0.,
    vmax=0.12,
    cmap='bone',
    label='[Mg/Fe]'
):
#     if usys is None:
#         usys = model.unit_sys

    if vlim is None:
        vlim = np.nanpercentile(data_H["label"], [1, 99])
    pcolor_kw = dict(shading="auto", vmin=vlim[0], vmax=vlim[1])

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.4), sharex=True, sharey=True, constrained_layout=True, facecolor='white'
    )

    cs = axes[0].pcolormesh(
        data_H["vz"], data_H["z"], data_H["label"],vmin=vmin,vmax=vmax, cmap=cmap
    )

    # Initial model:
    model0_H = np.array(model.label(z=data_H["z"], vz=data_H["vz"], params=params_init))
#     if mask_no_data:
#         model0_H[~np.isfinite(data_H["label_stat"])] = np.nan
    cs = axes[1].pcolormesh(
        data_H["vz"],
        data_H["z"],
        model0_H,
        vmin=vmin,
        vmax=vmax,
#         **pcolor_kw,
        cmap=cmap
    )

    # Fitted model:
    model_H = np.array(model.label(z=data_H["z"], vz=data_H["vz"], params=params_fit))
#     if mask_no_data:
#         model_H[~np.isfinite(data_H["label_stat"])] = np.nan
    cs = axes[2].pcolormesh(
        data_H["vz"],
        data_H["z"],
        model_H,
        vmin=vmin,
        vmax=vmax,
#         **pcolor_kw,
        cmap=cmap,
    )
    cax = axes[2].inset_axes([1.08, 0.02, 0.06, 0.95])
    cbar = fig.colorbar(cs, ax=axes[:3],cax=cax)
    cbar.set_label(label=r'$\langle$'+str(label)+'$\rangle$',fontsize=25)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
#     fig.colorbar(cs, ax=axes[:3], label=r'$\langle$[Mg/Fe]$\rangle$')

    # Residual:
    #     resid = np.array((data_H['label_stat'] - model_H) / model_H)
    resid = np.array(data_H["label"] - model_H)
#     resid[data_H['H'] < 5] = np.nan
    if smooth_residual is not None:
        resid = convolve(resid, Gaussian2DKernel(smooth_residual))
    cs = axes[3].pcolormesh(
        data_H["vz"],
        data_H["z"],
        resid,
#         vmin=-vlim_residual,
#         vmax=vlim_residual,
        vmin=-0.025,
        vmax=0.025,
        cmap="coolwarm",
        shading="auto",
    )
#     fig.colorbar(cs, ax=axes[3])
    cax = axes[3].inset_axes([1.08, 0.02, 0.06, 0.95])
    cbar = fig.colorbar(cs, ax=axes[:3],cax=cax)
#     cbar.set_label(ontsize=25)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    for ax in axes:
        ax.set_xlabel(f'$v_z$ [kpc/Gyr]',fontsize=25)
        ax.set_xlim(-0.06,0.06)
        ax.set_ylim(-2.,2.)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
        ax.tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
        ax.tick_params(which='minor', length=4, direction='in',top=True,right=True)
        ax.minorticks_on()
        
    axes[0].set_ylabel(f'$z$ [kpc]',fontsize=25)

    axes[0].set_title("data",fontsize=28)
    axes[1].set_title("initial model",fontsize=28)
    axes[2].set_title("fitted model",fontsize=28)
    axes[3].set_title("residual",fontsize=28)

    return fig, axes


def plot_data_model_es_labels(
    data_H,
    model,
    params_init,
    params_fit,
    indx,
    smooth_residual=None,
    vlim=None,
    vlim_residual=0.02,
    usys=None,
    mask_no_data=True,
    vmin=0.,
    vmax=0.12,
    cmap='bone',
    label='[Mg/Fe]'
):

    if vlim is None:
        vlim = np.nanpercentile(data_H["label"], [1, 99])
    pcolor_kw = dict(shading="auto", vmin=vlim[0], vmax=vlim[1])

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.4), constrained_layout=True, facecolor='white'
    )

    cs = axes[0].pcolormesh(
        data_H["vz"], data_H["z"], data_H["label"],vmin=vmin,vmax=vmax, cmap=cmap
    )
    axes[0].set_xlabel(f'$v_z$ [kpc/Gyr]',fontsize=25)
    axes[0].set_xlim(-0.06,0.06)
    axes[0].set_ylim(-2.,2.)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
    axes[0].tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
    axes[0].tick_params(which='minor', length=4, direction='in',top=True,right=True)
    axes[0].minorticks_on()
    axes[0].set_ylabel(f'$z$ [kpc]',fontsize=25)
    axes[0].set_title("data",fontsize=28)


        # Fitted model:
    model_H = np.array(model.label(z=data_H["z"], vz=data_H["vz"], params=params_fit))
#     if mask_no_data:
#         model_H[~np.isfinite(data_H["label_stat"])] = np.nan
    cs = axes[1].pcolormesh(
        data_H["vz"],
        data_H["z"],
        model_H,
        vmin=vmin,
        vmax=vmax,
#         **pcolor_kw,
        cmap=cmap,
    )
    axes[1].set_xlabel(f'$v_z$ [kpc/Gyr]',fontsize=25)
    axes[1].set_xlim(-0.06,0.06)
    axes[1].set_ylim(-2.,2.)
    axes[1].set_title("fitted model",fontsize=28)

#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
    axes[1].tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
    axes[1].tick_params(which='minor', length=4, direction='in',top=True,right=True)
    axes[1].minorticks_on()
    cax = axes[1].inset_axes([1.08, 0.02, 0.06, 0.95])
    cbar = fig.colorbar(cs, ax=axes[:3],cax=cax)
    cbar.set_label(label=r'$\langle$'+str(label)+'$\rangle$',fontsize=25)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    
#     i = 0
#     tmp = label_model

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    rzp_grid = np.linspace(0, 0.5, 128)
    tmp_es = model.get_es(rzp_grid, params_fit['e_params'])
    for k, vals in tmp_es.items():
        axes[2].plot(rzp_grid, vals, label=f'm={k}')

    axes[2].legend()
    axes[2].set_xlabel(r"$r_z'$",fontsize=25)
#     axes[2].set_xlim(0.,0.5)
#     axes[2].set_ylim(0.,np.max(vals)+0.2)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
    axes[2].tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
    axes[2].tick_params(which='minor', length=4, direction='in',top=True,right=True)
    axes[2].minorticks_on()
    axes[2].set_title("e$_{2}$ and e$_{4}$",fontsize=28)

    # --

    axes[3].plot(rzp_grid, model.label_func(rzp_grid, **params_fit['label_params']))
    axes[3].set_title("label val",fontsize=28)

    for ax in axes[:1]:
        ax.set_xlabel(f'$v_z$ [kpc/Gyr]',fontsize=25)
        ax.set_xlim(-0.06,0.06)
        ax.set_ylim(-2.,2.)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
        ax.tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
        ax.tick_params(which='minor', length=4, direction='in',top=True,right=True)
        ax.minorticks_on()
        
    for ax in axes[2:3]:
        ax.set_xlabel(f'$r$',fontsize=25)
        ax.set_xlim(0,0.5)
        ax.set_ylim(-0.3,0.3)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
        ax.tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
        ax.tick_params(which='minor', length=4, direction='in',top=True,right=True)
        ax.minorticks_on()
        
    for ax in axes[3:]:
        ax.set_xlabel(f'$r$',fontsize=25)
        ax.set_xlim(0,0.5)
        ax.set_ylim(0.,0.2)
#         ax.tick_params(labelsize=18,direction='in',top=True,right=True,length=6)
        ax.tick_params(which='major',labelsize=22,direction='in',top=True,right=True,length=8)
        ax.tick_params(which='minor', length=4, direction='in',top=True,right=True)
        ax.minorticks_on()
#     plt.savefig('/Users/dhortadarrington/Documents/Projects/orbital-torus-imaging/py/plots_tests_e2e4/test_mg_bin'+str(indx))
    return fig, axes
    

