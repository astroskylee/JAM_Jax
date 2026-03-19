"""
    Copyright (C) 2019-2023, Michele Cappellari

    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    https://purl.org/cappellari/software

    This software is provided as is without any warranty whatsoever.
    Permission to use, for non-commercial purposes is granted.
    Permission to modify for personal or internal use is granted,
    provided this copyright and disclaimer are included unchanged
    at the beginning of the file. All other rights are reserved.
    In particular, redistribution of the code is not allowed.

Changelog
---------

V1.0.0: Michele Cappellari, Oxford, 08 November 2019
    - Written and tested.
Vx.x.x: Additional changes are documented in the CHANGELOG of the JamPy package.

"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy import ndimage
from time import perf_counter as clock
from plotbin.plot_velfield import plot_velfield
import nquad as nq

##############################################################################

def _has_component_variation(values):
    """Best-effort static check for per-Gaussian anisotropy variation."""
    try:
        return bool(np.ptp(np.asarray(values)))
    except Exception:
        # Traced values are not concrete during NumPyro/NUTS compilation.
        # Fall back to the more general tensor-integral branch.
        return True

##############################################################################

def bilinear_interpolate(xv, yv, im, xout, yout):
    """
    The input array has size im[ny,nx] as in the output
    of im = f(meshgrid(xv, yv))
    xv and yv are vectors of size nx and ny respectively.

    """
    ny, nx = im.shape
    # assert (nx, ny) == (xv.size, yv.size), "Input arrays dimensions do not match"

    xi = (nx - 1.)/(xv[-1] - xv[0]) * (xout - xv[0])
    yi = (ny - 1.)/(yv[-1] - yv[0]) * (yout - yv[0])

    return ndimage.map_coordinates(im.T, [xi, yi], order=1, mode='nearest')

##############################################################################

def density(R, z, dens, sigma, qintr):
    """ Density for each luminous Gaussian at (R, z) """

    nu = dens*jnp.exp(-0.5/sigma**2*(R**2 + (z/qintr)**2))  # Cappellari (2008) eq.(13)

    return nu

def erfcx(x):
    return  jnp.exp( jnp.log(jax.scipy.special.erfc(x)) + x*x )

##############################################################################
def intrinsic_moments_cyl_bh(R, z, dens_lum, sigma_lum, qintr_lum, mbh, beta, gamma):
    """
    Compute analytic intrinsic moments for the black hole alone

    """
    G = 0.004301    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    h = sigma_lum*qintr_lum*jnp.sqrt(2)
    r = jnp.sqrt(R**2 + z**2)
    nu = density(R, z, dens_lum, sigma_lum, qintr_lum)
    nu /= nu.sum(-1)
    sig2z = G*mbh*nu*(1/r - erfcx(r/h)*jnp.sqrt(jnp.pi)/h)

    bani = 1/(1 - beta)  # Anisotropy ratio b = (sig_R/sig_z)**2
    sig2R = bani*sig2z
    sig2phi = (1 - gamma)*sig2R
    v2phi = G*mbh*R**2/r**3*(1 - bani)*nu \
            + bani*(1 + (1 - qintr_lum**2)*(R/(sigma_lum*qintr_lum))**2)*sig2z

    return sig2R.sum(-1), sig2z.sum(-1), sig2phi.sum(-1), v2phi.sum(-1)

##############################################################################

def integrand_cyl(u, R, z,
                  dens_lum, sigma_lum, qintr_lum,
                  dens_pot, sigma_pot, qintr_pot,
                  beta, gamma, logistic, component):
    """
    Compute all the non-zero JAM first and second intrinsic velocity moments
    using the formulas from Cappellari (2008, MNRAS, 390, 71; hereafter C08).
    https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C

    """
    if logistic:        # Variable anisotropy, same for all Gaussians
        zab, beta0, betainf, alpha = beta
        beta = (beta0 + betainf*(abs(z)/zab)**alpha)/(1 + (abs(z)/zab)**alpha)
        bani = 1/(1 - beta)
    else:               # Constant anisotropy, different per Gaussian
        bani = jnp.where(jnp.ptp(beta),1/(1 - beta[:, None, None]),1/(1 - beta[0]))
        #If the above does not work use : bani = 1/(1 - beta[:, None, None])
        #The following line is the original code
        #bani = 1/(1 - beta[:, None, None]) if jnp.ptp(beta) else 1/(1 - beta[0])  # Anisotropy ratio b = (sig_R/sig_z)**2



    dens_lum = dens_lum[:, None, None]
    sigma_lum = sigma_lum[:, None, None]
    qintr_lum = qintr_lum[:, None, None]

    dens_pot = dens_pot[None, :, None]
    sigma_pot = sigma_pot[None, :, None]
    qintr_pot = qintr_pot[None,:, None]

    # DE Change of variables for Chandrasekhar u-integral
    # jnp.arcsinh(jnp.log([rmin, rmax])*2/jnp.pi) -> [0, inf]
    x = jnp.exp(jnp.sinh(u)*jnp.pi/2)
    duds = x*jnp.cosh(u)*jnp.pi/2
    u = x[None, None, :]

    # Tracer
    R2, z2 = R**2, z**2
    a = -1/(2*sigma_lum**2)
    b = a/qintr_lum**2
    exp1 = dens_lum*jnp.exp(a*R2 + b*z2)

    # Mass
    G = 0.004301    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    u1, qu = 1 + u, qintr_pot**2 + u
    tmp = -1/(2*sigma_pot**2)
    c, d = tmp/u1, tmp/qu
    upot = 2*jnp.pi*G*dens_pot*qintr_pot*sigma_pot**2/(u1*jnp.sqrt(qu))
    exp2 = upot*jnp.exp(c*R2 + d*z2)

    bd = d/(b + d)
    sig2z = exp1*exp2*bd

    if component == 'sig2R':
        sig2z *= bani
    elif component == 'sig2phi':
        if logistic:        # Variable anisotropy, same for all Gaussians
            zag, gamma0, gammainf, alpha = gamma
            gamma = (gamma0 + gammainf*(abs(z)/zag)**alpha)/(1 + (abs(z)/zag)**alpha)
        else:               # Constant anisotropy, different per Gaussian
            gamma = jnp.where(jnp.ptp(gamma), gamma[:, None, None], gamma[0])
            #gamma = gamma[:, None, None] if jnp.ptp(gamma) else gamma[0]  # Anisotropy gamma = 1 - (sig_phi/sig_R)**2
        sig2z *= (1 - gamma)*bani
    elif component == 'v2phi':
        sig2z *= bani + 2*R2*((a + c)*bani - c/bd)

    return duds*sig2z.sum((0, 1))

##############################################################################
def intrinsic_moments_cyl(R, z,
                          dens_lum, sigma_lum, qintr_lum,
                          dens_pot, sigma_pot, qintr_pot,
                          mbh, beta, gamma, logistic, epsrel):
    """ Numerical quadratures of the Jeans solution """

    mds, mxs = jnp.median(sigma_lum), jnp.max(sigma_lum)
    xlim = jnp.arcsinh(jnp.log(jnp.array([1e-7*mds, 1e3*mxs]))*2/jnp.pi)

    args = [R, z, dens_lum, sigma_lum, qintr_lum, dens_pot, sigma_pot, qintr_pot, beta, gamma, logistic]
    nu = density(R, z, dens_lum, sigma_lum, qintr_lum).sum(-1)
    sig2z = nq.quad(integrand_cyl, xlim, args=args+['sig2z'])/nu
    v2phi = nq.quad(integrand_cyl, xlim, args=args+['v2phi'])/nu

    #Needtofix currently only support logisticTrue
    #if (not logistic) and (jnp.ptp(beta) or jnp.ptp(gamma)):
    sig2R = nq.quad(integrand_cyl, xlim, args=args+['sig2R'])/nu
    sig2phi = nq.quad(integrand_cyl, xlim, args=args+['sig2phi'])/nu
    # else:
    #     if logistic:
    #         zab, beta0, betainf, alpha = beta
    #         beta = (beta0 + betainf*(abs(z)/zab)**alpha)/(1 + (abs(z)/zab)**alpha)
    #         zag, gamma0, gammainf, alpha = gamma
    #         gamma = (gamma0 + gammainf*(abs(z)/zag)**alpha)/(1 + (abs(z)/zag)**alpha)
    #     else:
    #         beta = beta[0]
    #         gamma = gamma[0]
    #     sig2R = sig2z/(1 - beta)
    #     sig2phi = sig2R*(1 - gamma)

    # if mbh > 0:
    mbh += 1e-10
    sig2R_bh, sig2z_bh, sig2phi_bh, v2phi_bh = intrinsic_moments_cyl_bh(
        R, z, dens_lum, sigma_lum, qintr_lum, mbh, beta, gamma)
    sig2R += sig2R_bh
    sig2z += sig2z_bh
    sig2phi += sig2phi_bh
    v2phi += v2phi_bh

    return sig2R, sig2z, sig2phi, v2phi, nu

##############################################################################
def integand_tan_dth_pot(u, R2, z2, dens_pot, sigma_pot, qintr_pot):
    """
    Returns the integrand of the tan(th)*d(pot)/dth derivative
    of the MGE potential at (r, th).
    This is equation (51) of Cappellari (2020, MNRAS, 494, 4819)
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C

    """
    # DE Change of variables for Chandrasekhar u-integral
    # jnp.arcsinh(jnp.log([rmin, rmax])*2/jnp.pi) -> [0, inf]
    x = jnp.exp(jnp.sinh(u)*jnp.pi/2)
    duds = x*jnp.cosh(u)*jnp.pi/2
    u = x[:, None]

    G = 0.004301    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    qu = qintr_pot**2 + u
    u1 = 1 + u
    d = 2*jnp.pi*G*dens_pot*qintr_pot*(qintr_pot**2 - 1)*R2
    e = jnp.exp(-0.5/sigma_pot**2*(R2/u1 + z2/qu))
    tan_dth_pot = d*e/(u1**2*qu**1.5)

    return duds*tan_dth_pot.sum(1)   # u.size

##############################################################################
def integrand_sph(s, t, r, th, dens_lum, sigma_lum, qintr_lum, beta, gamma,
                  logistic, dens_pot, sigma_pot, qintr_pot, component):
    """
    Solution of the spherically-aligned Jeans equations for an MGE
    from eq.(52)-(54) of Cappellari (2020, MNRAS, 494, 4819)
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C

    """
    dens_lum = dens_lum[:, None, None]
    s2_lum = (sigma_lum**2)[:, None, None]
    q2_lum = (qintr_lum**2)[:, None, None]

    dens_pot = dens_pot[None, :, None]
    s2_pot = (sigma_pot**2)[None, :, None]
    qintr_pot = qintr_pot[None, :, None]

    # DE Change of variables for Chandrasekhar u-integral (Sec.6.2 of Cappellari 2020)
    # jnp.arcsinh(jnp.log([rmin, rmax])*2/jnp.pi) -> [0, inf]
    x = jnp.exp(jnp.sinh(s)*jnp.pi/2)
    duds = x*jnp.cosh(s)*jnp.pi/2
    u = x[None, None, :]

    # TANH Change of variables for Jeans r-integral (Sec.6.2 of Cappellari 2020)
    # jnp.log([rmin, rmax]) -> [r, inf]
    drdt = jnp.exp(t)
    r1 = r + drdt[None, None, :]

    if logistic:        # Variable anisotropy, same for all Gaussians
        rab, beta0, betainf, alpha = beta
        fun = (r1/r)**(2*beta0)
        fun *= ((1 + (r1/rab)**alpha)/(1 + (r/rab)**alpha))**(2*(betainf - beta0)/alpha)
        beta = beta0 + (betainf - beta0)/(1 + (rab/r)**alpha)
    else:               # Constant anisotropy, different per Gaussian
        beta = jnp.where(jnp.ptp(beta), beta[:, None, None], beta[0])
        fun = (r1/r)**(2*beta)

    # Tracer Gaussians
    rs = fun*(r*jnp.sin(th))**2
    aa = -r1**2/(2*q2_lum*s2_lum)
    bb = (1 - q2_lum)/(2*q2_lum*s2_lum)*rs
    ex1 = dens_lum*jnp.exp(aa + bb)

    # Mass Gaussians
    G = 0.004301    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    qu = qintr_pot**2 + u
    u1 = 1 + u
    cc = -r1**2/(2*qu*s2_pot)
    dd = (u1 - qu)/(2*qu*s2_pot*u1)*rs
    cst = 2*jnp.pi*G*dens_pot*qintr_pot*r1
    ex2 = cst*jnp.exp(cc + dd)/(u1*qu**1.5)

    psi = fun*ex1*ex2

    if component == 'sig2r':
        integ = psi
    elif component == 'sig2th':
        integ = psi*(1 - beta)
    elif component == 'sig2phi':
        if logistic:    # Variable anisotropy, same for all Gaussians
            rag, gamma0, gammainf, alpha = gamma
            gamma = gamma0 + (gammainf - gamma0)/(1 + (rag/r)**alpha)
        else:           # Constant anisotropy, different per Gaussian
            gamma = jnp.where(jnp.ptp(gamma), gamma[:, None, None], gamma[0])  # Anisotropy gamma = 1 - (sig_phi/sig_R)**2
        integ = psi*(1 - gamma)
    elif component == 'v2phi':
        integ = psi*(1 - beta)*(1 + 2*(bb + dd))

    return duds*drdt*integ.sum((0, 1))    # u.size == t.size

##############################################################################
def integrand_sph_bh(t, r, th, dens_lum, sigma_lum, qintr_lum, beta, gamma, logistic, mbh, component):
    """
    Solution of the spherically-aligned Jeans equations for a black hole alone

    """
    # TANH Change of variables for Jeans r-integral (Sec.6.2 of Cappellari 2020)
    # jnp.log([rmin, rmax]) -> [r, inf]
    drdt = jnp.exp(t)
    r1 = r + drdt[:, None]

    if logistic:    # Variable anisotropy, same for all Gaussians
        rab, beta0, betainf, alpha = beta
        fun = (r1/r)**(2*beta0)
        fun *= ((1 + (r1/rab)**alpha)/(1 + (r/rab)**alpha))**(2*(betainf - beta0)/alpha)
        beta = beta0 + (betainf - beta0)/(1 + (rab/r)**alpha)
    else:           # Constant anisotropy, different per Gaussian
        fun = (r1/r)**(2*beta)

    # Tracer Gaussians
    s2_lum = sigma_lum**2
    q2_lum = qintr_lum**2
    rs = fun*(r*jnp.sin(th))**2
    aa = -r1**2/(2*q2_lum*s2_lum)
    bb = (1 - q2_lum)/(2*q2_lum*s2_lum)*rs
    ex1 = dens_lum*jnp.exp(aa + bb)

    G = 0.004301    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    psi = G*fun*ex1*mbh/r1**2

    if component == 'sig2r':
        integ = psi
    elif component == 'sig2th':
        integ = psi*(1 - beta)
    elif component == 'sig2phi':
        if logistic:    # Variable anisotropy, same for all Gaussians
            rag, gamma0, gammainf, alpha = gamma
            gamma = gamma0 + (gammainf - gamma0)/(1 + (rag/r)**alpha)
        integ = psi*(1 - gamma)
    elif component == 'v2phi':
        integ = psi*(1 - beta)*(1 + 2*bb)

    return drdt*integ.sum(1)

##############################################################################
def intrinsic_moments_sph(R, z,
                          dens_lum, sigma_lum, qintr_lum,
                          dens_pot, sigma_pot, qintr_pot,
                          mbh, beta, gamma, logistic, epsrel):
    """ Numerical quadratures of the Jeans solution """

    r = jnp.sqrt(R**2 + z**2)
    th = jnp.arctan2(R, z)  # Angle from symmetry axis z

    mds, mxs = jnp.median(sigma_lum), jnp.max(sigma_lum)
    xlim = jnp.arcsinh(jnp.log(jnp.array([1e-7*mds, 1e3*mxs]))*2/jnp.pi)
    ylim = jnp.log(jnp.array([1e-6*mds, 3*mxs]))
    args = [r, th, dens_lum, sigma_lum, qintr_lum, beta, gamma, logistic]
    args_mge = args + [dens_pot, sigma_pot, qintr_pot]
    nu = density(R, z, dens_lum, sigma_lum, qintr_lum).sum(-1)
    varying_beta = _has_component_variation(beta)
    varying_gamma = _has_component_variation(gamma)

    ranges_2d = jnp.stack([xlim, ylim])
    sig2r = nq.nquad(integrand_sph, ranges_2d, args=args_mge+['sig2r'])/nu
    v2phi = nq.nquad(integrand_sph, ranges_2d, args=args_mge+['v2phi'])/nu

    if mbh > 0:
        args_bh = args + [mbh]
        sig2r += jnp.squeeze(nq.quad(integrand_sph_bh, ylim, args=args_bh+['sig2r']))/nu
        v2phi += jnp.squeeze(nq.quad(integrand_sph_bh, ylim, args=args_bh+['v2phi']))/nu

    if (not logistic) and (varying_beta or varying_gamma):
        sig2th = nq.nquad(integrand_sph, ranges_2d, args=args_mge+['sig2th'])/nu
        sig2phi = nq.nquad(integrand_sph, ranges_2d, args=args_mge+['sig2phi'])/nu
        if mbh > 0:
            sig2th += jnp.squeeze(nq.quad(integrand_sph_bh, ylim, args=args_bh+['sig2th']))/nu
            sig2phi += jnp.squeeze(nq.quad(integrand_sph_bh, ylim, args=args_bh+['sig2phi']))/nu
    else:
        if logistic:
            rab, beta0, betainf, alpha = beta
            beta = beta0 + (betainf - beta0)/(1 + (rab/r)**alpha)
            rag, gamma0, gammainf, alpha = gamma
            gamma = gamma0 + (gammainf - gamma0)/(1 + (rag/r)**alpha)
        else:
            beta = beta[0]
            gamma = gamma[0]
        sig2th = sig2r*(1 - beta)
        sig2phi = sig2r*(1 - gamma)

    v2phi += jnp.squeeze(nq.quad(integand_tan_dth_pot, xlim,
                     args=(R**2, z**2, dens_pot, sigma_pot, qintr_pot)))

    return jnp.array([sig2r, sig2th, sig2phi, v2phi, nu])

##############################################################################
def intrinsic_moments(Rbin, zbin, dens_lum, sigma_lum, qintr_lum, dens_pot,
                      sigma_pot, qintr_pot, mbh, beta, gamma, logistic, nrad,
                      nang, epsrel, interp, proj_cyl, align):

    fun = intrinsic_moments_cyl if align == 'cyl' else intrinsic_moments_sph

    if nrad*nang > Rbin.size or (not interp):  # Just calculate values
        #print("Just compute values")
        RzArgs = jnp.array([Rbin,zbin]).T

        def funArgs(arg):
            return fun(arg[0],arg[1],
            dens_lum, sigma_lum, qintr_lum, dens_pot, sigma_pot, qintr_pot,
            mbh, beta, gamma, logistic, epsrel)
        
        vfunArgs = jax.vmap(funArgs,0,0)
        A = jnp.squeeze(vfunArgs(RzArgs)).T
        sig2r, sig2th, sig2phi, v2phi, nu = A
        
    else:
        #print("Perform interpolation")
        irp = mom_interp(Rbin, zbin, dens_lum, sigma_lum, qintr_lum, dens_pot,
                         sigma_pot, qintr_pot, mbh, beta, gamma, logistic, nrad,
                         nang, align=align)
        sig2r, sig2th, sig2phi, v2phi, nu = irp.get_moments(Rbin, zbin)

    # Project the velocity dispersion tensor to cylindrical coordinates
    if proj_cyl and (align == 'sph'):
        th = jnp.arctan2(Rbin, zbin)  # Angle from symmetry axis z
        sig2R = sig2th*jnp.cos(th)**2 + sig2r*jnp.sin(th)**2
        sig2z = sig2th*jnp.sin(th)**2 + sig2r*jnp.cos(th)**2
        sig2r, sig2th = sig2R, sig2z

    return jnp.array([sig2r, sig2th, sig2phi, v2phi, nu])

##############################################################################

class mom_interp:

    def __init__(self, xbin, ybin,
                 dens_lum, sigma_lum, qintr_lum,
                 dens_pot, sigma_pot, qintr_pot,
                 mbh, beta, gamma, logistic, nrad, nang, epsrel=1e-2,
                 rmin=None, rmax=None, align='cyl'):
        """
        Initializes model values on a grid for subsequent interpolation

        """
        fun = intrinsic_moments_cyl if align == 'cyl' else intrinsic_moments_sph

        # Define parameters of polar grid for interpolation
        w = sigma_lum < jnp.max(jnp.abs(xbin))  # Characteristic MGE axial ratio in observed range
        # self.qmed = jnp.median(qintr_lum) if w.sum(-1) < 3 else jnp.median(qintr_lum[w])
        ####################Change the if statement to use nanmedian plus jnp.select###############
        #This is a work for concreate boolean sums which numpyro hates
        self.qmed = jax.lax.select(w.sum() < 3, jnp.nanmedian(jnp.where(w, qintr_lum, jnp.nan)), jnp.median(qintr_lum))

        if rmin is None or rmax is None:
            rell2 = xbin**2 + (ybin/self.qmed)**2

        self.rmin = jnp.sqrt(jnp.min(rell2)) if rmin is None else rmin
        self.rmax = jnp.sqrt(jnp.max(rell2)) if rmax is None else rmax

        # Make linear grid in log of elliptical radius RAD and eccentric anomaly ECC
        rad =  jnp.geomspace(self.rmin, self.rmax, nrad)
        self.logRad = jnp.log(rad)
        self.ang = jnp.linspace(0, jnp.pi/2, nang)
        rellGrid, eccGrid = map(jnp.ravel, jnp.meshgrid(rad, self.ang))
        R = rellGrid*jnp.cos(eccGrid)
        z = rellGrid*jnp.sin(eccGrid)*self.qmed  # ecc=0 on equatorial plane

        RzArgs = jnp.array([R,z]).T
        def funArgs(arg):
            res = fun(arg[0],arg[1],
                dens_lum, sigma_lum, qintr_lum, dens_pot, sigma_pot, qintr_pot,
                mbh, beta, gamma, logistic, epsrel)
            return res
       
        vfunArgs = jax.vmap(funArgs,0,0)
        A = jnp.squeeze(vfunArgs(RzArgs)).T
        sig2r, sig2th, sig2phi, v2phi, nu = A
        

        self.sig2r = sig2r.reshape(nang, nrad)
        self.sig2th = sig2th.reshape(nang, nrad)
        self.sig2phi = sig2phi.reshape(nang, nrad)
        self.v2phi = v2phi.reshape(nang, nrad)
        self.dens_lum = dens_lum
        self.sigma_lum = sigma_lum
        self.qintr_lum = qintr_lum

##############################################################################

    def get_moments(self, R, z):
        """
        Fast linearly-interpolated model values at the set
        of (R, z) coordinates from pre-computed values.
        Interpolation of non-weighted kinematics for accuracy.
        The returned density is analytic, not interpolated.

        """
        r1 = 0.5*jnp.log((R**2 + (z/self.qmed)**2).clip(self.rmin**2))  # Log elliptical radius of input (R,z)
        e1 = jnp.arctan2(jnp.abs(z/self.qmed), jnp.abs(R))  # Eccentric anomaly of input (R,z)
        sig2r = bilinear_interpolate(self.logRad, self.ang, self.sig2r, r1, e1)
        sig2th = bilinear_interpolate(self.logRad, self.ang, self.sig2th, r1, e1)
        sig2phi = bilinear_interpolate(self.logRad, self.ang, self.sig2phi, r1, e1)
        v2phi = bilinear_interpolate(self.logRad, self.ang, self.v2phi, r1, e1)
        nu = density(R[:, None], z[:, None], self.dens_lum, self.sigma_lum, self.qintr_lum).sum(-1)

        return sig2r, sig2th, sig2phi, v2phi, nu

##############################################################################

class jam_axi_intr(object):
    """
    jam_axi_intr
    ============

    Purpose
    -------

    This procedure calculates all the intrinsic first and second velocity
    moments for an anisotropic axisymmetric galaxy model.

    This program is useful e.g. to model the kinematics of galaxies
    like our Milky Way, for which the intrinsic moments can be observed
    directly, or to compute starting conditions for N-body numerical
    simulations of galaxies.

    Two assumptions for the orientation of the velocity ellipsoid are supported:

    - The cylindrically-aligned ``(R, z, phi)`` solution was presented in
      `Cappellari (2008) <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_

    - The spherically-aligned ``(r, th, phi)`` solution was presented in
      `Cappellari (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C>`_

    Calling Sequence
    ----------------

    .. code-block:: python

        from jampy.jam_axi_intr import jam_axi_intr

        jam = jam_axi_intr(
                 dens_lum, sigma_lum, qintr_lum, dens_pot, sigma_pot, qintr_pot,
                 mbh, Rbin, zbin, align='cyl', beta=None, data=None, epsrel=1e-2,
                 errors=None, gamma=None, goodbins=None, interp=True,
                 logistic=False, ml=None, nang=10, nodots=False, nrad=20,
                 plot=True, proj_cyl=False, quiet=False)

        # The meaning of the output is different depending on `align`
        sig2R, sig2z, sig2phi, v2phi = jam.model  # with align='cyl'
        sig2r, sig2th, sig2phi, v2phi = jam.model  # with align='sph'

        jam.plot()   # Generate data/model comparison

    Input Parameters
    ----------------

    dens_lum: array_like with shape (n,)
        vector containing the peak value of the MGE Gaussians describing
        the intrinsic density of the tracer population for which the kinematics
        is derived.
        The units are arbitrary as they cancel out in the final results.
        Typical units are e.g. ``Lsun/pc^3`` (solar luminosities per ``parsec^3``)
    sigma_lum: array_like with shape (n,)
        vector containing the dispersion (sigma) in ``pc`` of the MGE
        Gaussians describing the galaxy kinematic-tracer population.
    qintr_lum: array_like with shape (n,)
        vector containing the intrinsic axial ratio (q) of the MGE
        Gaussians describing the galaxy kinematic-tracer population.
    surf_pot: array_like with shape (m,)
        vector containing the peak value of the MGE Gaussians
        describing the galaxy total-mass density in units of ``Msun/pc^3``
        (solar masses per ``parsec^3``). This is the MGE model from which the
        model gravitational potential is computed.
    sigma_pot: array_like with shape (m,)
        vector containing the dispersion in ``pc`` of the MGE
        Gaussians describing the galaxy total-mass density.
    qintr_pot: array_like with shape (m,)
        vector containing the intrinsic axial ratio of the MGE
        Gaussians describing the galaxy total-mass density.
    mbh: float
        Mass of a nuclear supermassive black hole in solar masses.
    Rbin: array_like with shape (p,)
        Vector with the ``R`` coordinates in ``pc`` of the bins (or pixels) at
        which one wants to compute the model predictions. This is the first
        cylindrical coordinate ``(R, z)`` with the galaxy center at ``(0,0)``.

        There is a singularity at ``(0, 0)`` which should be avoided by the user
        in the input coordinates.
    zbin: array_like with shape (p,)
        Vector with the ``z`` coordinates in ``pc`` of the bins (or pixels) at
        which one wants to compute the model predictions. This is the second
        cylindrical coordinate ``(R, z)``, with the z-axis coincident with the
        galaxy symmetry axis.

    Optional Keywords
    -----------------

    align: {'cyl', 'sph'} optional
        If ``align='cyl'`` the program computes the solution of the Jeans
        equations with cylindrically-aligned velocity ellipsoid, presented
        in `Cappellari (2008)`_. If ``align='sph'`` the spherically-aligned
        solution of `Cappellari (2020)`_ is returned.
    beta: array_like with shape (n,) or (4,)
        Vector with the axial anisotropy of the individual kinematic-tracer
        MGE Gaussians (Default: ``beta=jnp.zeros(n)``)::

            beta = 1 - (sigma_th/sigma_r)^2  # with align='sph'
            beta = 1 - (sigma_z/sigma_R)^2   # with align='cyl'

        When ``logistic=True`` the procedure assumes::

            beta = [r_a, beta_0, beta_inf, alpha]

        and the anisotropy of the whole JAM model varies as a logistic
        function of the logarithmic spherical radius (with ``align='sph'``) or
        of the logarithmic distance from the equatorial plane (with ``align='cyl'``)::

            beta(r) = beta_0 + (beta_inf - beta_0)/[1 + (r_a/r)^alpha]    # with align='sph'
            beta(z) = beta_0 + (beta_inf - beta_0)/[1 + (z_a/|z|)^alpha]  # with align='cyl'

        Here ``beta_0`` represents the anisotropy at ``r = 0``, ``beta_inf``
        is the anisotropy at ``r = inf`` and ``r_a`` is the anisotropy
        transition radius, with ``alpha`` controlling the sharpness of the
        transition. In the special case ``beta_0 = 0, beta_inf = 1, alpha = 2``
        the anisotropy variation reduces to the form by Osipkov & Merritt, but
        the extra parameters allow for much more realistic anisotropy profiles.
        See an application in `Simon, Cappellari & Hartke (2023)
        <https://ui.adsabs.harvard.edu/abs/2023arXiv230318229S>`_.
    data: array_like of shape (4, p), optional
        Four input vectors with the observed values of:

        - ``[sigR, sigz, sigphi, vrms_phi]`` in ``km/s``, when ``align='cyl'``
          (or ``align='sph'`` and ``proj_cyl=True``).

          ``vrms_phi`` is the square root of the velocity second moment in the
          tangential direction. If the velocities ``vphi_j`` are measured from
          individual stars then ``vrms_phi = sqrt(mean(vphi_j^2))``.
          One can also use the relation ``vrms_phi = sqrt(vphi^2 + sigphi^2)``,
          where ``vphi = mean(vphi_j)`` and ``sigphi = std(vphi_j)``

        - ``[sigr, sigth, sigphi, vrms_phi]`` in ``km/s``, when ``align='sph'``,
          where ``vrms_phi`` is defined above.

    epsrel: float, optional
        Relative error requested for the numerical quadratures, before
        interpolation (Default: ``epsrel=1e-2``).
    errors: array_like of shape (4, p), optional
        ``1sigma`` uncertainties on ``data``, in the same format (default 5 ``km/s``).
    gamma: array_like with shape (n,)
        Vector with the tangential anisotropy of the individual kinematic-tracer
        MGE Gaussians (Default: ``gamma=jnp.zeros(n)``)::

            gamma = 1 - (sigma_phi/sigma_r)^2  # with align='sph'
            gamma = 1 - (sigma_phi/sigma_R)^2  # with align='cyl'

        When ``logistic=True`` the procedure assumes::

            gamma = [r_a, gamma_0, gamma_inf, alpha]

        and the anisotropy of the whole JAM model varies as a logistic
        function of the logarithmic spherical radius (with ``align='sph'``) or
        of the logarithmic distance from the equatorial plane (with ``align='cyl'``)::

            gamma(r) = gamma_0 + (gamma_inf - gamma_0)/[1 + (r_a/r)^alpha]    # with align='sph'
            gamma(z) = gamma_0 + (gamma_inf - gamma_0)/[1 + (z_a/|z|)^alpha]  # with align='cyl'

        Here ``gamma_0`` represents the anisotropy at ``r = 0``, ``gamma_inf``
        is the anisotropy at ``r = inf`` and ``r_a`` is the anisotropy
        transition radius, with ``alpha`` controlling the sharpness of the
        transition. In the special case ``gamma_0 = 0, gamma_inf = 1, alpha = 2``
        the anisotropy variation reduces to the form by Osipkov & Merritt, but
        the extra parameters allow for much more realistic anisotropy profiles.
    goodbins: array_like with shape (4, p), optional
        Boolean vector of the same shape as ``data`` with values ``True``
        for the bins which have to be included in the fit (if requested) and
        ``chi^2`` calculation (Default: fit all bins).
    interp: bool, optional
        If ``interp=False`` no interpolation is performed and the model is
        computed at every set of input (R, z) coordinates.
        If ``interp=True`` (default), the model is interpolated if the number
        of requested input (R, z) coordinates is larger than ``nang*nrad``.
    logistic: bool, optional
        When ``logistic=True``, JAM interprets the anisotropy parameters
        ``beta`` and ``gamma`` as defining a 4-parameters logistic function.
        See the documentation of the anisotropy keywords for details.
        (Default ``logistic=False``)
    ml: float, optional
        Mass-to-light ratio M/L. If ``ml=None`` (default) the M/L is fitted to
        the data and the best-fitting value is returned in output.
        The ``mbh`` is also scaled and becomes ``mbh*ml``.
        If ``ml=1`` no scaling is applied to the model.
    nang: int, optional
        The number of linearly-spaced intervals in the eccentric anomaly at
        which the model is evaluated before interpolation (default: ``nang=10``).
    nodots: bool, optional
        Set to ``True`` to hide the dots indicating the centers of the bins in
        the two-dimensional map (default ``False``).
    nrad: int, optional
        The number of logarithmically spaced radial positions at which the
        model is evaluated before interpolation. One may want to increase this
        value if the model has to be evaluated over many orders of magnitude in
        radius (default: ``nrad=20``).
    plot: bool, optional
        If ``plot=True`` (default) and ``data is not None``, produce a plot of
        the data-model comparison at the end of the calculation.
    proj_cyl: bool, optional
        If ``align='sph'`` and ``proj_cyl=True``, the function projects the
        spherically-aligned moments to cylindrical coordinates and returns the
        ``[sig2R, sig2z, sig2phi, v2phi]`` components as in the case
        ``align='cyl'``. This is useful for a direct comparison of results with
        either the spherical or cylindrical alignment, as it allows one to fit
        the same data with both modelling assumptions.
    quiet: bool, optional
        If ``quiet=False`` (default), print the best-fitting M/L and chi2 at
        the end for the calculation.

    Output Parameters
    -----------------

    Returned as attributes of the ``jam_axi_intr`` class.

    .chi2: float
        Reduced chi^2 (chi^2/DOF) describing the quality of the fit::

            d = (data/errors)[goodbins]
            m = (model/errors)[goodbins]
            chi2 = ((d - m)**2).sum()/goodbins.sum()

    .flux: array_like  with shape (p,)
        Vector with the MGE luminosity density at each ``(R, z)`` location in
        ``Lsun/pc^3``, used to plot the isophotes on the model results.
    .ml: float
        Best fitting M/L. This value is fitted while ignoring ``sigphi`` and it
        is strictly independent of the adopted tangential anisotropy ``gamma``.
    .model: array_like with shape (4, p)
        - Contains ``[sig2R, sig2z, sig2phi, v2phi]`` with ``align='cyl'``

        - Contains ``[sig2r, sig2th, sig2phi, v2phi]`` with ``align='sph'``

        where the above quantities are defined as follows:

        sig2R (sig2r): array_like with shape (p,)
            squared intrinsic dispersion in ``(km/s)^2`` along the R (r)
            direction at each ``(R, z)`` location.

        sig2z (sig2th): array_like with shape (p,)
            squared intrinsic dispersion in ``(km/s)^2`` along the z (th)
            direction at each ``(R, z)`` location.

        sig2phi: array_like with shape (p,)
            squared intrinsic dispersion in ``(km/s)^2``  along the
            tangential ``phi`` direction at each ``(R, z)`` location.

        v2phi: array_like with shape (p,)
            the second velocity moment in ``(km/s)^2`` along the
            tangential ``phi`` direction at each ``(R, z)`` location.

        The mean velocity along the tangential direction can be computed as
        ``vphi = jnp.sqrt(v2phi - sig2phi)``

        NOTE: I return squared velocities instead of taking the square root,
        to allow for negative values (unphysical solutions).

    ###########################################################################
    """

    @staticmethod
    def get_kinematics( dens_lum, sigma_lum, qintr_lum, dens_pot, sigma_pot,
                 qintr_pot, mbh, Rbin, zbin, align='cyl', beta=None, data=None,
                 epsrel=1e-2, errors=None, gamma=None, goodbins=None,
                 interp=True, logistic=False, ml=None, nang=10,
                 nrad=20, proj_cyl=False, quiet=False):

        # assert align in ['sph', 'cyl'], "`align` must be 'sph' or 'cyl'"
        # assert (ml is None) or (ml > 0), "The input `ml` must be positive"
        if beta is None:
            beta = jnp.zeros_like(dens_lum)  # Anisotropy parameter beta = 1 - (sig_th/sig_r)**2  (beta=0-->circle)
        if gamma is None:  # Anisotropy parameter beta = 1 - (sig_th/sig_r)**2
            if logistic:
                gamma = [1, 0, 0, 1]    # [r_a, gamma_0, gamma_inf, alpha]
            else:
                gamma = jnp.zeros_like(beta)
        # assert (dens_lum.size == sigma_lum.size == qintr_lum.size) \
            #    and ((len(beta) == 4 and logistic) or (len(beta) == dens_lum.size)) \
            #    and (len(beta) == len(gamma)), "The luminous MGE components and anisotropies do not match"
        # assert dens_pot.size == sigma_pot.size == qintr_pot.size, "The total mass MGE components do not match"
        # assert Rbin.size == zbin.size, "Rbin and zbin do not match"

        if data is not None:
            # assert len(data) == 4, "`data` must contain four vectors"
            if errors is None:
                errors = jnp.full_like(data, 5)  # Constant 5 km/s errors
            if goodbins is None:
                goodbins = jnp.ones_like(data, dtype=bool)
            # else:
                # print("")
                # assert goodbins.dtype == bool, "goodbins must be a boolean vector"
                # assert jnp.any(goodbins), "goodbins must contain some True values"
            # assert Rbin.size == len(data[0]) == len(errors[0]) == len(goodbins[0]), \
                # "(data, errors, goodbins) and (Rbin, zbin) do not match"
        if goodbins is None:
            goodbins = jnp.ones_like(data, dtype=bool)
        # else:
            # assert goodbins.dtype == bool, "goodbins must be a boolean vector"
            # assert jnp.any(goodbins), "goodbins must contain some True values"

        t = clock()

        # model contains [sig2r, sig2th, sig2phi, v2phi]
        *model, nu = intrinsic_moments(Rbin, zbin, dens_lum, sigma_lum,
            qintr_lum, dens_pot, sigma_pot, qintr_pot, mbh, beta, gamma,
            logistic, nrad, nang, epsrel, interp, proj_cyl, align)

        if not quiet:
            print(f'jam_axi_intr_{align} elapsed time (sec): {clock() - t:.2f}')

        if data is None:
            chi2 = None
            if ml is None:
                ml = 1.
        else:
            model0 = jnp.sqrt(jnp.clip(jnp.array(model), 0, None))   # sqrt([sig2r, sig2th, sig2phi, v2phi])

            # Exclude sig2phi from M/L calculation
            if (ml is None) or (ml <= 0):
                m = jnp.delete(model0, 2, 0).ravel()     # sqrt([sig2r, sig2th, v2phi])
                d = jnp.delete(data, 2, 0).ravel()
                e = jnp.delete(errors, 2, 0).ravel()
                ok = jnp.delete(goodbins, 2, 0).ravel()
                d, m = (d/e)[ok], (m/e)[ok]
                ml = ((d @ m)/(m @ m))**2   # eq. (51) of Cappellari (2008, MNRAS)

            # Use full data for chi2 calculation
            model0 *= jnp.sqrt(ml)
            d, m, e, ok = map(jnp.ravel, [data, model0, errors, goodbins])
            chi2 = (((d[ok] - m[ok])/e[ok])**2).sum(-1)/ok.sum(-1)

        # self.Rbin = Rbin
        # self.zbin = zbin
        # self.align = align
        # self.model = jnp.array(model)*ml  # model ~ V^2
        # self.data = data
        # self.errors = errors
        # self.flux = nu
        # self.ml = ml
        # self.chi2 = chi2
        # self.goodbins = goodbins
        return jnp.array(model),nu,ml,chi2

##############################################################################

    @staticmethod
    def plot(Rbin,zbin,align,model,data,errors,nu,ml,chi2,nodots=True):
        """ Data-model comparison for jam_axi_intr """

        plt.figure('jam')
        plt.clf()
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, num='jam', gridspec_kw={'hspace': 0})

        if align == 'cyl':
            txt = [r"$\sigma_R$", r"$\sigma_z$", r"$\sigma_\phi$", r"$V^{\rm rms}_\phi$"]
        else:
            txt = [r"$\sigma_r$", r"$\sigma_\theta$", r"$\sigma_\phi$", r"$V^{\rm rms}_\phi$"]

        ax[0, 0].set_title("Data")
        ax[0, 1].set_title(f"JAM$_{{\\rm {align}}}$ Model")

        for j, (d, m, t) in enumerate(zip(data, jnp.sqrt(model.clip(0)), txt)):
            plt.sca(ax[j, 0])
            mn, mx = jnp.percentile(d, jnp.array([1, 99]))
            plot_velfield(Rbin, zbin, d, vmin=mn, vmax=mx, flux=nu, nodots=nodots)

            plt.sca(ax[j, 1])
            plot_velfield(Rbin, zbin, m, vmin=mn, vmax=mx, flux=nu,
                          colorbar=1, label=t, nodots=nodots)

###############################################################################

def test_jam_axi_intr():


    def satoh_solution(R, z, beta):
            """
            Implements the analytic Jeans solution of eq.(68)-(69) from
            `Cappellari (2020, hereafter C20) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C>`_

            """
            G = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
            a = b = M = 1

            Q = np.sqrt(b**2 + z**2)
            S = np.sqrt(a**2 + 2*a*Q + R**2 + z**2)
            sig2z = G*M*Q*(a + 2*Q)/(3*Q*(a + 2*Q) + S**2)/(2*S)        # C20 eq.(68)
            v2phi = sig2z*(1 - 6*(R/S)**2)/(1 - beta) + G*M*R**2/S**3   # C20 eq.(69)

            return sig2z, v2phi

    import numpy as np

    rr = np.linspace(0, 16, 32)
    zz = np.linspace(0, 6, 12)
    Rbin, zbin = map(np.ravel, np.meshgrid(rr, zz))
    w = (Rbin == 0) & (zbin == 0)
    Rbin, zbin = Rbin[~w], zbin[~w]  # Remove central singularity

    lg_dens_lum, lg_sigma_lum, qintr_lum = np.loadtxt('./examples/cappellari2020_table1.txt').T
    dens_lum, sigma_lum = 10**lg_dens_lum, 10**lg_sigma_lum

    beta0 = -0.78
    sig2z, v2phi = satoh_solution(Rbin, zbin, beta0)

    gamma0 = 0.78
    beta = np.full_like(dens_lum, beta0)
    gamma = np.full_like(dens_lum, gamma0)

    dens_pot = dens_lum
    sigma_pot = sigma_lum
    qintr_pot = qintr_lum

    mbh = 0
    sig2R = sig2z/(1 - beta0)
    sig2phi = sig2R*(1 - gamma0)
    data = np.sqrt([sig2R, sig2z, sig2phi, v2phi])
    errors = np.full_like(data, 5.28e-5)  # make chi2/DOF = 1

    from functools import partial

    jam_obj = jam_axi_intr()
    jam_eval = partial(jam_obj.get_kinematics, sigma_lum=sigma_lum,  qintr_lum=qintr_lum, dens_pot=dens_pot, 
                    sigma_pot=sigma_pot, qintr_pot=qintr_pot,
                    mbh=mbh, Rbin=Rbin, zbin=zbin, beta=beta, gamma=gamma, align='cyl',
                    data=data, errors=errors, ml=None,quiet=True)
    model,nu,ml,chi2 = jam_eval(dens_lum)

    jam_obj.plot(Rbin,zbin,"cyl",model,data,errors,nu,ml,chi2)

    derivative_jam_eval = jax.jacfwd(jam_eval)

    # jam.plot(True)
    plt.show()

    der_jam = derivative_jam_eval(dens_lum)

    print(der_jam[2])
