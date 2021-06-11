import pickle
import sys
import warnings
from math import pi
from pathlib import Path

import matplotlib.colors
import numpy as np
from jax import numpy as jnp
from jax import random
import numpyro
from numpyro.contrib.funsor import config_enumerate
from numpyro.distributions import (
    Dirichlet,
    Gamma,
    Uniform,
    VonMises,
    Beta,
    Categorical,
    Sine,
    SineSkewed,
    HalfNormal,
)
from numpyro.infer import NUTS, MCMC, init_to_value, Predictive, init_to_median
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import math

np.set_printoptions(threshold=sys.maxsize)

AMINO_ACIDS = [
    "M",
    "N",
    "I",
    "F",
    "E",
    "L",
    "R",
    "D",
    "G",
    "K",
    "Y",
    "T",
    "H",
    "S",
    "P",
    "A",
    "V",
    "Q",
    "W",
    "C",
]


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


@config_enumerate
def sine_model(data, num_data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample("mix_weights", Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample("beta_mean_phi", Uniform(0.0, 1.0))
    beta_prec_phi = numpyro.sample("beta_prec_phi", Gamma(1.0, 1 / 20.0))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample("beta_mean_psi", Uniform(0, 1.0))
    beta_prec_psi = numpyro.sample("beta_prec_psi", Gamma(1.0, 1 / 20.0))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate("mixture", num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample("phi_loc", VonMises(pi, 2.0))
        psi_loc = numpyro.sample("psi_loc", VonMises(0.0, 0.1))
        phi_conc = numpyro.sample(
            "phi_conc", Beta(halpha_phi, beta_prec_phi - halpha_phi)
        )
        psi_conc = numpyro.sample(
            "psi_conc", Beta(halpha_psi, beta_prec_psi - halpha_psi)
        )
        corr_scale = numpyro.sample("corr_scale", Beta(2.0, 10.0))

    with numpyro.plate("obs_plate", num_data, dim=-1):
        assign = numpyro.sample(
            "mix_comp", Categorical(mix_weights), infer={"enumerate": "parallel"}
        )
        sine = Sine(
            phi_loc=phi_loc[assign],
            psi_loc=psi_loc[assign],
            phi_concentration=1000 * phi_conc[assign],
            psi_concentration=1000 * psi_conc[assign],
            weighted_correlation=corr_scale[assign],
        )
        return numpyro.sample("phi_psi", sine, obs=data)


@config_enumerate
def ss_model(data, num_data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample("mix_weights", Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample("beta_mean_phi", Uniform(0.0, 1.0))
    beta_prec_phi = numpyro.sample("beta_prec_phi", Gamma(1.0, 1 / 20.0))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample("beta_mean_psi", Uniform(0, 1.0))
    beta_prec_psi = numpyro.sample("beta_prec_psi", Gamma(1.0, 1 / 20.0))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate("mixture", num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample("phi_loc", VonMises(pi, 2.0))
        psi_loc = numpyro.sample("psi_loc", VonMises(0.0, 0.1))
        phi_conc = numpyro.sample(
            "phi_conc", Beta(halpha_phi, beta_prec_phi - halpha_phi)
        )
        psi_conc = numpyro.sample(
            "psi_conc", Beta(halpha_psi, beta_prec_psi - halpha_psi)
        )
        corr_scale = numpyro.sample("corr_scale", Beta(2.0, 10.0))

        skew_phi = numpyro.sample("skew_phi", Uniform(-1.0, 1.0))
        psi_bound = 1 - jnp.abs(skew_phi)
        skew_psi = numpyro.sample("skew_psi", Uniform(-1.0, 1.0))
        skewness = jnp.stack((skew_phi, psi_bound * skew_psi), axis=-1)
        assert skewness.shape == (num_mix_comp, 2)

    with numpyro.plate("obs_plate", num_data, dim=-1):
        assign = numpyro.sample(
            "mix_comp", Categorical(mix_weights), infer={"enumerate": "parallel"}
        )
        sine = Sine(
            phi_loc=phi_loc[assign],
            psi_loc=psi_loc[assign],
            phi_concentration=1000 * phi_conc[assign],
            psi_concentration=1000 * psi_conc[assign],
            weighted_correlation=corr_scale[assign],
        )
        return numpyro.sample("phi_psi", SineSkewed(sine, skewness[assign]), obs=data)


def run_hmc(model, data, num_mix_comp, num_samples, bvm_init_locs):
    rng_key = random.PRNGKey(0)
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=bvm_init_locs),
        dense_mass=True,
        max_tree_depth=5,
    )
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_samples // 5)
    mcmc.run(rng_key, data, len(data), num_mix_comp)
    mcmc.print_summary()
    post_samples = mcmc.get_samples()
    return post_samples


def fetch_aa_dihedrals(split="train", subsample_to=1000_000, reuse_shuffles=True):
    file = Path(__file__).parent / "data/9mer_fragments_processed.pkl"
    data = pickle.load(file.open("rb"))[split]["sequences"]
    data_aa = np.argmax(data[..., :20], -1)
    data = {aa: data[..., -2:][data_aa == i] for i, aa in enumerate(AMINO_ACIDS)}

    shuffle_file = Path(__file__).parent / "runs/sample_indices.pkl"
    if reuse_shuffles:
        shuffles = pickle.load(shuffle_file.open("rb"))
    else:
        shuffles = {
            k: np.random.permutation(np.arange(v.shape[0]))[
                : min(subsample_to, v.shape[0])
            ]
            for k, v in data.items()
        }
        pickle.dump(shuffles, shuffle_file.open("wb"))
    data = {aa: aa_data[shuffles[aa]] for aa, aa_data in data.items()}
    data = {aa: jnp.array(aa_data, dtype=float) for aa, aa_data in data.items()}
    return data


def ramachandran_plot(data, pred_data, aas, file_name="rama_plots.png"):
    amino_acids = {"S": "Serine", "P": "Proline", "G": "Glycine"}
    fig, axss = plt.subplots(2, len(aas), dpi=300)
    cdata = data
    for i in range(len(axss)):
        if i == 1:
            cdata = pred_data
        for ax, aa in zip(axss[i], aas):
            aa_data = cdata[aa]
            nbins = 628  # 100 * 2pi

            ax.hexbin(
                aa_data[:, 0],
                aa_data[:, 1],
                norm=matplotlib.colors.LogNorm(),
                bins=nbins,
                gridsize=200,
                cmap="Blues",
            )

            # label the contours
            ax.set_aspect("equal", "box")
            ax.set_xlim([-math.pi, math.pi])
            ax.set_ylim([-math.pi, math.pi])
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            if i == 0:
                axtop = ax.secondary_xaxis("top")
                axtop.set_xlabel(amino_acids[aa])
                axtop.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
                axtop.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
                axtop.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

            if i == 1:
                ax.set_xlabel("$\phi$")

    for i in range(len(axss)):
        axss[i, 0].set_ylabel("$\psi$")
        axss[i, 0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        axss[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        axss[i, 0].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axright = axss[i, -1].secondary_yaxis("right")
        axright.set_ylabel("data" if i == 0 else "simulation")
        axright.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        axright.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        axright.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for ax in axss[:, 1:].reshape(-1):
        ax.tick_params(labelleft=False)
        ax.tick_params(labelleft=False)

    for ax in axss[0, :].reshape(-1):
        ax.tick_params(labelbottom=False)
        ax.tick_params(labelbottom=False)

    if file_name:
        viz_dir = Path(__file__).parent.parent / "viz"
        viz_dir.mkdir(exist_ok=True)
        fig.tight_layout()
        plt.savefig(str(viz_dir / file_name), bbox_inches="tight")
    else:
        fig.tight_layout()
        plt.show()
    plt.clf()


def show_center(data, aa, means, num_comp, file_name="kde_rama_pred.png"):
    fig, ax = plt.subplots(1, 1)
    aa_data = data[aa]

    ax.scatter(aa_data[:, 0], aa_data[:, 1], color="k", s=1)
    ax.scatter(means[:, 0], means[:, 1], marker="x", color="r", s=5, label=num_comp)
    ax.set_xlim([-math.pi, math.pi])
    ax.set_ylim([-math.pi, math.pi])
    fig.tight_layout()
    plt.show()
    plt.clf()


def main(
    num_samples=1_000,
    aas=("S", "G", "P"),
    capture_std=True,
    rerun_inference=True,
    reuse_shuffles=True,
):
    data = fetch_aa_dihedrals(subsample_to=25_000, reuse_shuffles=reuse_shuffles)
    num_mix_comps = {"S": 9, "G": 10, "P": 7}
    pred_datas = {}
    rng_key = random.PRNGKey(123)
    for aa in aas:
        num_mix_comp = num_mix_comps[aa]
        if capture_std:
            sys.stdout = (
                Path(__file__).parent
                / "runs"
                / f"ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.out"
            ).open("w")

        chain_file = (
            Path(__file__).parent
            / "runs"
            / f"ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.pkl"
        )

        kmeans = KMeans(num_mix_comp)
        kmeans.fit(data[aa])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            show_center(data, aa, kmeans.cluster_centers_, num_mix_comp, file_name="")
        if rerun_inference or not chain_file.exists():
            means = {
                "phi_loc": kmeans.cluster_centers_[:, 0],
                "psi_loc": kmeans.cluster_centers_[:, 1],
            }
            posterior_samples = {
                "ss": run_hmc(ss_model, data[aa], num_mix_comp, num_samples, means)
            }
            pickle.dump(posterior_samples, chain_file.open("wb"))
        else:
            posterior_samples = pickle.load(chain_file.open("rb"))
            predictive = Predictive(ss_model, posterior_samples["ss"], parallel=True)
            rng_key, pred_key = random.split(rng_key)
            pred_datas[aa] = predictive(pred_key, None, 25_000, num_mix_comp)["phi_psi"]
        if capture_std:
            sys.stdout.close()

    if not rerun_inference:
        ramachandran_plot(data, pred_datas, aas)


if __name__ == "__main__":
    main()
