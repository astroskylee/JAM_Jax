import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import init_to_median
from scipy.special import roots_legendre
import numpy as np
import pandas as pd
import arviz as az
from jax.scipy.special import gammaln
from jax.scipy.special import logsumexp
from scipy.stats import truncnorm


class tool:
    c_km_s = 299792.458
    c_m_s = 299792458
    G = 6.6743e-11
    msun = 1.988409870698051e+30
    Mpc = 3.0856776e+22
    pc = 3.0856776e+16
    @staticmethod
    def func(z, Omegam, Omegak, w0, wa = 0):
        # Normalized Hubble parameter E(z)
        Omegal = 1 - Omegam - Omegak
        w_z = w0 + wa * z / (1 + z)
        return (Omegam * (1 + z)**3 + Omegak * (1 + z)**2 + Omegal * (1 + z)**(3 * (1 + w_z)))**-0.5

    @staticmethod
    def nth_order_quad(n=20):
        xval, weights = map(jnp.array, roots_legendre(n))
        xval = xval.reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        def integrate(func, a, b, *args):
            return 0.5 * (b - a) * jnp.sum(
                weights * func(0.5 * ((b - a) * xval + (b + a)), *args),
                axis=0
            )
        return integrate

    @staticmethod
    def integrate(func, a, b, *args, n=20):
        # 对外提供统一入口，可直接调用tool.integrate
        quad = tool.nth_order_quad(n)
        return quad(func, a, b, *args)

    @staticmethod
    def Dplus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sinh(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sinh(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sinh(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dminus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sin(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sin(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sin(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dflat(Es, El, zs, zl):
        Ds = Es / (1 + zs)
        Dls = (Es - El) / (1 + zs)
        Dl = El / (1 + zl)
        return Dl, Ds, Dls
    #####################################################################################
    @staticmethod
    def Dpos(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sinh(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dneg(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sin(sqrt_ok * E) / sqrt_ok / (1 + z)
        
    @staticmethod
    def Dzero(E, z):
        return E / (1 + z)
    #####################################################################################

    @staticmethod
    def angular_diameter_distance(z, cosmology, n=20):
        Omegam = cosmology['Omegam']
        Omegak = cosmology['Omegak']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        h = cosmology['h0']
        E = tool.integrate(tool.func, 0, z, Omegam, Omegak, w0, wa, n=n)
        
        Dl = lax.cond(
            Omegak > 0,
            lambda _: tool.Dpos(Omegak, E, z),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: tool.Dneg(Omegak, E, z),
                lambda _: tool.Dzero(E, z),
                None
            ),
            None
        )
        return Dl*tool.c_km_s/h

    @staticmethod
    def dldsdls(zl, zs, cosmology, n=20):
        """
        Compute distances based on cosmological parameters.
        cosmology: dict, output of cosmology_model function.
        """
        Omegam = cosmology['Omegam']
        Omegak = cosmology['Omegak']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        h = cosmology['h0']

        El = tool.integrate(tool.func, 0, zl, Omegam, Omegak, w0, wa, n=n)
        Es = tool.integrate(tool.func, 0, zs, Omegam, Omegak, w0, wa, n=n)

        Dl, Ds, Dls = lax.cond(
            Omegak > 0,
            lambda _: tool.Dplus(Omegak, Es, El, zs, zl),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: tool.Dminus(Omegak, Es, El, zs, zl),
                lambda _: tool.Dflat(Es, El, zs, zl),
                None
            ),
            None
        )
        return Dl*tool.c_km_s/h, Ds*tool.c_km_s/h, Dls*tool.c_km_s/h
    
    def sigma_crit_jax(zl, zs, cosmology, n=20):
        # output is in Solar mass / pc**2
        Dl, Ds, Dls = tool.dldsdls(zl = zl, zs = zs,cosmology = cosmology, n=n)
        factor = tool.c_km_s**2 / (4.0 * jnp.pi * tool.G)
        return factor * (Ds / (Dl * Dls))*tool.pc/tool.msun

    @staticmethod
    def compute_distances(zl, zs, cosmology, n=20):
        """
        Compute distances based on cosmological parameters.
        cosmology: dict, output of cosmology_model function.
        """
        Omegam = cosmology['Omegam']
        Omegak = cosmology['Omegak']
        w0 = cosmology['w0']
        wa = cosmology['wa']

        El = tool.integrate(tool.func, 0, zl, Omegam, Omegak, w0, wa, n=n)
        Es = tool.integrate(tool.func, 0, zs, Omegam, Omegak, w0, wa, n=n)

        Dl, Ds, Dls = lax.cond(
            Omegak > 0,
            lambda _: tool.Dplus(Omegak, Es, El, zs, zl),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: tool.Dminus(Omegak, Es, El, zs, zl),
                lambda _: tool.Dflat(Es, El, zs, zl),
                None
            ),
            None
        )
        return Dl, Ds, Dls

    @staticmethod
    def jgamma(n):
        return jnp.exp(gammaln(n))

    @staticmethod
    def f_mass(ygamma,delta,beta):
        eps=ygamma+delta-2
        return (3-delta)/(eps-2*beta)/(3-eps)*(
            tool.jgamma(eps/2-1/2)/tool.jgamma(eps/2)- beta*tool.jgamma(eps/2+1/2)/tool.jgamma(eps/2+1)
        )*tool.jgamma(ygamma/2)*tool.jgamma(delta/2)/(tool.jgamma(ygamma/2-1/2)*tool.jgamma(delta/2-1/2))

    @staticmethod
    def FoM(samples_df, nbins=100, confidence=0.95):
        w0 = samples_df['w0'].values
        wa = samples_df['wa'].values if 'wa' in samples_df.columns else np.zeros_like(w0)

        range_w0 = (np.min(w0), np.max(w0))
        range_wa = (np.min(wa), np.max(wa))
        H, w0_edges, wa_edges = np.histogram2d(w0, wa, bins=nbins, range=[range_w0, range_wa])
        H = H / np.sum(H)

        H_flat = H.flatten()
        idx_sorted = np.argsort(H_flat)[::-1]
        H_sorted = H_flat[idx_sorted]

        cumsum = np.cumsum(H_sorted)
        threshold_idx = np.where(cumsum >= confidence)[0][0]
        level_conf = H_sorted[threshold_idx]

        inside_bins = (H >= level_conf)
        bin_area = (w0_edges[1] - w0_edges[0]) * (wa_edges[1] - wa_edges[0])
        A_conf = np.sum(inside_bins) * bin_area

        FoM_val = (6.17 * np.pi) / A_conf
        return FoM_val

    @staticmethod
    def FoM_cov(samples_df):
        cov_matrix = samples_df[['w0', 'wa']].cov()
        det_cov = np.linalg.det(cov_matrix.values)
        return 1 / np.sqrt(det_cov)

    @staticmethod
    def inf2pd(inf_data):
        posterior = inf_data.posterior
        samples_dict = {}
        for var_name in posterior.data_vars:
            samples_dict[var_name] = posterior[var_name].values.flatten()
        return pd.DataFrame(samples_dict)
    ########################################################################################
    @staticmethod
    def beta_mst(beta, mst):
        eta = 1/beta
        eta_mst = eta*(eta+(1-eta)*mst)
        return 1/eta_mst
        
    @staticmethod
    def beta_antimst(beta_mst, mst):
        eta_mst = 1/beta_mst
        eta = eta_mst*mst/(1-eta_mst*(1-mst))
        return 1/eta

    @staticmethod
    def beta_mst_v3(beta, mst):
        eta = 1/beta
        eta_mst = 1 - mst(1-eta)
        return 1/eta_mst
        
    @staticmethod
    def beta_antimst_v3(beta_mst, mst):
        eta_mst = 1/beta_mst
        eta = 1 - (1 - eta_mst)/mst
        return 1/eta
    ########################################################################################
    
    @staticmethod
    def EPL(R, thetaE, gamma):
        kappa = (3 - gamma) / 2 * (thetaE / R) ** (gamma - 1)
        return kappa

    @staticmethod
    def EPL_msunmpc(R, thetaE, gamma, zl, zs, cosmology):
        kappa = (3 - gamma) / 2 * (thetaE / R) ** (gamma - 1)
        return kappa*tool.sigma_crit_jax(zl = zl, zs = zs, cosmology = cosmology)

    def sersic_constant(sersic_index):
        # use less accurate one for direct comparison with lenstronomy
        bn = 1.9992 * sersic_index - 0.3271
        bn = jnp.maximum(bn, 0.00001)  # make sure bn is strictly positive as a save guard for very low n_sersic
        return bn

    def sersic_fn(r, mass_to_light_ratio, intensity, effective_radius, sersic_index, **_):
        b = tool.sersic_constant(sersic_index)
        r_ = (r / effective_radius)**(1.0 / sersic_index)
        return mass_to_light_ratio * intensity * jnp.exp(-b * (r_ - 1.0))
    @staticmethod
    def trunc_norm(mu, sigma, a, b, size=1):
        return truncnorm((a-mu)/sigma, (b-mu)/sigma, loc=mu, scale=sigma).rvs(size)


class SLCOSMO:
    def __init__(self):
        # Cosmological parameters (如果不需要可去掉)
        self.Omegam_true = 0.3
        self.Omegak_true = 0.0  # Flat universe
        self.w0_true = -1.0
        self.wa_true = 0.0

    def run_inference(self, data_dict, sampler_type, cosmology_type='wcdm', cosmo_prior = None, sample_h0 = False,
                     num_warmup=500, num_samples=2000, num_chains=10,jax_key = 0):

        # 自动提取 selected_models 从 data_dict 的键
        selected_models = list(data_dict.keys())

        sampler = MCMC(
            sampler_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            chain_method='vectorized'
        )

        print(f'Running inference with models: {"+".join(selected_models)}')
        sampler.run(
            jax.random.PRNGKey(jax_key),
            selected_models=selected_models,
            data_dict=data_dict,
            cosmology_type=cosmology_type,
            cosmo_prior = cosmo_prior,
            sample_h0 = sample_h0
        )

        inf_data = az.from_numpyro(sampler)
        return inf_data
        
class SLmodel:
    def __init__(self, slcosmo):
        self.slcosmo = slcosmo
        self.c = 299792.458  # km/s

        # 字典映射模型名称到子模型方法
        self.model_dict = {
            'dspl': self.DSPL_model,
            'dspl_mst': self.DSPL_model_mst,
            'dspl_mst_bias': self.DSPL_model_mst_bias,
            'sne': self.LensedSNe_model,
            'sne_bias': self.LensedSNe_model_mst_bias,
            'kinematic': self.lens_kinematic_model,
            'kinematic_bias': self.lens_kinematic_model_bias,
            'kinematic_gamma': self.lens_kinematic_gamma
        }

    def cosmology_model(self, cosmology_type, cosmo_prior,  sample_h0=False):
        """
        定义宇宙学参数模型。
        """

        cosmology = {
            'Omegam': numpyro.sample('Omegam', dist.Uniform(cosmo_prior['omegam_low'], cosmo_prior['omegam_up'])),
            'Omegak': 0.0,
            'w0': -1.0,
            'wa': 0.0,
            'h0': 70
        }

        if cosmology_type == 'wcdm':
            cosmology['w0'] = numpyro.sample('w0', dist.Uniform(cosmo_prior['w0_low'], cosmo_prior['w0_up']))
        elif cosmology_type == 'owcdm':
            cosmology['Omegak'] = numpyro.sample('Omegak', dist.Uniform(cosmo_prior['omegak_low'], cosmo_prior['omegak_up']))
            cosmology['w0'] = numpyro.sample('w0', dist.Uniform(cosmo_prior['w0_low'], cosmo_prior['w0_up']))
        elif cosmology_type == 'waw0cdm':
            cosmology['w0'] = numpyro.sample('w0', dist.Uniform(cosmo_prior['w0_low'], cosmo_prior['w0_up']))
            cosmology['wa'] = numpyro.sample('wa', dist.Uniform(cosmo_prior['wa_low'], cosmo_prior['wa_up']))
        elif cosmology_type == 'owaw0cdm':
            cosmology['Omegak'] = numpyro.sample('Omegak', dist.Uniform(cosmo_prior['omegak_low'], cosmo_prior['omegak_up']))
            cosmology['w0'] = numpyro.sample('w0', dist.Uniform(cosmo_prior['w0_low'], cosmo_prior['w0_up']))
            cosmology['wa'] = numpyro.sample('wa', dist.Uniform(cosmo_prior['wa_low'], cosmo_prior['wa_up']))
        elif cosmology_type != 'lambdacdm':
            raise ValueError(f"Unknown cosmology_type: {cosmology_type} available types: lambdacdm, wcdm, waw0cdm, owcdm, owaw0cdm")

        if sample_h0:
            cosmology['h0'] = numpyro.sample('h0', dist.Uniform(cosmo_prior['h0_low'], cosmo_prior['h0_up']))

        return cosmology

    def DSPL_model(self, dspl_data, cosmology):
        """
        DSPL 模型。
        """
        zl = dspl_data['zl_dspl']
        zs1 = dspl_data['zs1_dspl']
        zs2 = dspl_data['zs2_dspl']
        beta_obs = dspl_data['beta_obs_dspl']
        sigma_beta = dspl_data['sigma_beta_dspl']
        
        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)

        beta_model_dspl = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            numpyro.sample('beta_obs_dspl', dist.Normal(beta_model_dspl, sigma_beta), obs=beta_obs)



    def DSPL_model_mst(self, dspl_data, cosmology):
        zl          = dspl_data['zl_dspl']
        zs1         = dspl_data['zs1_dspl']
        zs2         = dspl_data['zs2_dspl']
        theta_1     = dspl_data['theta_1']
        theta_2     = dspl_data['theta_2']
        beta_obs    = dspl_data['beta_obs_dspl']
        sigma_beta  = dspl_data['sigma_beta_dspl']
        lambda_int  = dspl_data['lambda_int']
        lambda_int_err = dspl_data['lambda_int_err']

        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)
        
        beta_mst = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            mst = numpyro.sample('mst_dspl', dist.TruncatedNormal(lambda_int, lambda_int_err, low = 0.5, high  = 1.5))
            beta = tool.beta_antimst(beta_mst, mst)
            numpyro.sample("beta_obs_dspl", dist.TruncatedNormal(beta, sigma_beta, low = 0, high = 1), obs=beta_obs)
        numpyro.deterministic("mean_mst", jnp.mean(mst))

    def DSPL_model_mst_bias(self, dspl_data, cosmology, bias):
        zl          = dspl_data['zl_dspl']
        zs1         = dspl_data['zs1_dspl']
        zs2         = dspl_data['zs2_dspl']
        theta_1     = dspl_data['theta_1']
        theta_2     = dspl_data['theta_2']
        beta_obs    = dspl_data['beta_obs_dspl']
        sigma_beta  = dspl_data['sigma_beta_dspl']
        lambda_int  = dspl_data['lambda_int']
        lambda_int_err = dspl_data['lambda_int_err']

        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)
        
        beta_mst = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            mst = numpyro.sample('mst_dspl', dist.TruncatedNormal(lambda_int, lambda_int_err, low = 0.3, high  = 1.8))+bias
            beta = tool.beta_antimst(beta_mst, mst)
            numpyro.sample("beta_obs_dspl", dist.TruncatedNormal(beta, sigma_beta, low = 0, high = 1), obs=beta_obs)
        numpyro.deterministic("mean_mst", jnp.mean(mst))

    def LensedSNe_model(self, sne_data, cosmology):
        """
        LensedSNe 模型。
        """
        zl = sne_data['zl_sne']
        zs = sne_data['zs_sne']
        tmax = sne_data['tmax_sne']
        Ddt_obs = sne_data['Ddt_obs_sne']
        Ddt_err = sne_data['Ddt_err_sne']
        
        Dl_sne, Ds_sne, Dls_sne = tool.compute_distances(zl, zs, cosmology)
        Ddt_sne = (1 + zl) * Dl_sne * Ds_sne / Dls_sne * self.c / cosmology['H0'] / 1000

        with numpyro.plate("LensedSNe_data", len(zl)):
            numpyro.sample('Ddt_obs_sne', dist.Normal(Ddt_sne, Ddt_err / 1000), obs=Ddt_obs / 1000)

    def LensedSNe_model_mst_bias(self, sne_data, cosmology, bias):
        """
        LensedSNe 模型。
        """
        zl = sne_data['zl_sne']
        zs = sne_data['zs_sne']
        tmax = sne_data['tmax_sne']
        Ddt_obs = sne_data['Ddt_obs_sne']
        Ddt_err = sne_data['Ddt_err_sne']
        lambda_int = sne_data['lambda_int']
        lambda_int_err = sne_data['lambda_int_err']
        
        Dl_sne, Ds_sne, Dls_sne = tool.compute_distances(zl, zs, cosmology)
        Ddt_sne_mst = (1 + zl) * Dl_sne * Ds_sne / Dls_sne * self.c / cosmology['H0'] / 1000
        
        with numpyro.plate("LensedSNe_data", len(zl)):
            Ddt_sne = Ddt_sne_mst*(numpyro.sample('mst_sn', dist.Normal(lambda_int, lambda_int_err))+bias)
            numpyro.sample('Ddt_obs_sne', dist.Normal(Ddt_sne, Ddt_err / 1000), obs=Ddt_obs / 1000)


    def lens_kinematic_model(self, kin_data, cosmology):
        """
        Lens Kinematic 模型
        """
        zl = kin_data['zl_kin']
        zs = kin_data['zs_kin']
        theta_E_obs = kin_data['theta_E_obs_kin']
        delta = kin_data['delta_kin']
        sigma_v_obs = kin_data['sigma_v_obs_kin']
        vel_err = kin_data['vel_err_kin']
        
        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        # 额外的先验
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample('gamma_sig', dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.6, 1.0))
        beta_sigma = numpyro.sample('beta_sig_kin', dist.TruncatedNormal(0.13, 1.0, low=0.0, high=0.4))

        # 单个透镜的参数
        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.5))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-1.0, high=0.8))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)

        # logfactor =  -300*(1-fmass>0)  
        # numpyro.factor("cusp", logfactor )
    
        pre_vel = jnp.sqrt((3e8**2) / (2 * jnp.pi**0.5) * Ds_kin/Dls_kin * (xsample / 206265) * fmass * (0.725 / xsample)**(2 - y_i)) / 1000 / 100

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def lens_kinematic_model_bias(self, kin_data, cosmology):
        """
        Lens Kinematic 模型
        """
        zl = kin_data['zl_kin']
        zs = kin_data['zs_kin']
        theta_E_obs = kin_data['theta_E_obs_kin']
        delta = kin_data['delta_kin']
        sigma_v_obs = kin_data['sigma_v_obs_kin']
        vel_err = kin_data['vel_err_kin']
        
        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        # 额外的先验
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample('gamma_sig', dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.6, 1.0))
        beta_sigma = numpyro.sample('beta_sig_kin', dist.TruncatedNormal(0.13, 1.0, low=0.0, high=0.4))
        lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.8, 1.2))
        lambda_sigma = numpyro.sample('lambda_mean_sig',dist.Uniform(0.0, 0.1))

        # 单个透镜的参数
        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            lambda_npr = numpyro.sample("lambda",dist.Normal(lambda_mean,lambda_sigma))
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.5))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-1.0, high=0.8))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)

        D_ratio = numpyro.deterministic('distance_ratio', Ds_kin/Dls_kin)
        pre_vel = jnp.sqrt(lambda_npr) *  jnp.sqrt((3e8**2) / (2 * jnp.pi**0.5) * D_ratio * (xsample / 206265) * fmass * (0.725 / xsample)**(2 - y_i)) / 1000 / 100

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def lens_kinematic_gamma(self, kin_data, cosmology):
        """
        Lens Kinematic 模型。
        """
        zl = kin_data['zl_kin']
        zs = kin_data['zs_kin']
        theta_E_obs = kin_data['theta_E_obs_kin']
        delta = kin_data['delta_kin']
        sigma_v_obs = kin_data['sigma_v_obs_kin']
        sigma_v_true = kin_data['sigma_v_obs_kin']
        vel_err = kin_data['vel_err_kin']
        gamma_pop = kin_data['gamma_pop']
        gamma_err = kin_data['gamma_err']
        
        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.4, 0.6))
        beta_sigma = numpyro.sample('beta_sig_kin', dist.TruncatedNormal(0.13, 1.0, low=0.05, high=0.4))
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample('gamma_sig', dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.8, 1.2))
        lambda_sigma = numpyro.sample('lambda_mean_sig',dist.Uniform(0.0, 0.1))
        #y_i = gamma_pop

        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            lambda_npr = numpyro.sample("lambda",dist.Normal(lambda_mean,lambda_sigma))
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.7))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-5, high=0.5))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)
        D_ratio = numpyro.deterministic('distance_ratio', Ds_kin/Dls_kin)
        pre_vel = jnp.sqrt(lambda_npr) * jnp.sqrt((3e8**2) / (2 * jnp.pi**0.5) * D_ratio * (xsample / 206265) * fmass * (0.725 / xsample)**(2 - y_i)) / 1000 / 100

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("gamma_obs", dist.Normal(y_i, gamma_err), obs=(gamma_pop))
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def joint_model(self, selected_models, data_dict, cosmology_type='wcdm', sample_h0 = False, cosmo_prior = None):
        """
        联合模型，根据 selected_models 动态调用子模型。
        
        参数:
            selected_models (list of str): 要包含的子模型名称，如 ['sne', 'dspl']。
            data_dict (dict): 包含所有子模型数据的字典。
            cosmology_type (str): 宇宙学模型类型。
        """
        if cosmo_prior is None:
            cosmo_prior = {
            'w0_up': -0.5,
            'w0_low': -1.5,
            'wa_up': 2,
            'wa_low': -2,
            'omegak_up': 1,
            'omegak_low': -1,
            'h0_up': 80,
            'h0_low': 60,
            'omegam_up': 0.5,
            'omegam_low': 0.1
            }
        
        cosmology = self.cosmology_model(cosmology_type, cosmo_prior,  sample_h0=sample_h0)
        bias = numpyro.sample('bias', dist.TruncatedNormal(0, 1, low=-0.1, high=0.1))
        # 动态调用选定的子模型
        for model_name in selected_models:
            if model_name in self.model_dict:
                if model_name == 'dspl_mst_bias' or model_name =='sne_bias':
                    self.model_dict[model_name](data_dict[model_name], cosmology, bias)
                else:
                    self.model_dict[model_name](data_dict[model_name], cosmology)



