"""
    Copyright (C) 2019-2023, Michele Cappellari

    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    http://purl.org/cappellari/software

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
from jax.tree_util import Partial as partial
import time
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy import special, signal, ndimage
from time import perf_counter as clock
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from jam_axi_intr import mom_interp
import nquad as nq

##############################################################################
def integrand_cyl_los(u1,
                      dens_lum, sigma_lum, q_lum,
                      dens_pot, sigma_pot, q_pot,
                      x1, y1, inc, beta, tensor):
    """
    This routine computes the integrand of Eq.(28) of Cappellari (2008; C08) for
    a model with constant anisotropy sigma_R**2 = b*sigma_z**2 and <V_R*V_z> = 0.
    The components of the proper motion dispersions tensor are calculated as
    described in note 5 of C08.
    See Cappellari (2012; C12 http://arxiv.org/abs/1211.7009)
    for explicit formulas for the proper motion tensor.
    The formulas are also given in Appendix A3 of Cappellari (2020, C20)
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C


    """
    dens_lum = dens_lum[:, None, None]
    sigma_lum = sigma_lum[:, None, None]
    q_lum = q_lum[:, None, None]
    beta = beta[:, None, None]

    dens_pot = dens_pot[None, :, None]
    sigma_pot = sigma_pot[None, :, None]
    q_pot = q_pot[None, :, None]

    u = u1[None, None, :]

    kani = 1./(1 - beta)  # Anisotropy ratio b = (sig_R/sig_z)**2
    ci = jnp.cos(inc)
    si = jnp.sin(inc)
    si2 = si**2
    ci2 = ci**2
    x2 = x1**2
    y2 = y1**2
    u2 = u**2

    s2_lum = sigma_lum**2
    q2_lum = q_lum**2
    e2_lum = 1 - q2_lum
    s2q2_lum = s2_lum*q2_lum

    s2_pot = sigma_pot**2
    e2_pot = 1 - q_pot**2

    # Double summation over (j, k) of eq.(28) for all values of integration variable u.
    # The triple loop in (j, k, u) is replaced by broadcast Numpy array operations.
    e2u2_pot = e2_pot*u2
    a = 0.5*(u2/s2_pot + 1/s2_lum)               # equation (29) in C08
    b = 0.5*(e2u2_pot*u2/(s2_pot*(1 - e2u2_pot)) + e2_lum/s2q2_lum) # equation (30) in C08
    c = e2_pot - s2q2_lum/s2_pot                  # equation (22) in C08
    d = 1 - kani*q2_lum - ((1 - kani)*c + e2_pot*kani)*u2  # equation (23) in C08
    e = a + b*ci2
    if tensor == 'xx':
        f = kani*s2q2_lum + d*((y1*ci*(a+b)/e)**2 + si2/(2*e)) # equation (4) in C12
    elif tensor == 'yy':
        f = s2q2_lum*(si2 + kani*ci2) + d*x2*ci2  # equation (5) in C12
    elif tensor == 'zz':
        f = s2q2_lum*(ci2 + kani*si2) + d*x2*si2  # z' LOS equation (28) in C08
    elif tensor == 'xy':
        f = -d*jnp.abs(x1*y1)*ci2*(a+b)/e          # equation (6) in C12
    elif tensor == 'xz':
        f = -d*jnp.abs(x1*y1)*si*ci*(a+b)/e         # -equation (7) in C12
    elif tensor == 'yz':
        f = -si*ci*(s2q2_lum*(1 - kani) - d*x2)   # -equation (8) in C12

    # arr has the dimensions (q_lum.size, q_pot.size, u.size)

    arr = q_pot*dens_pot*dens_lum*u2*f*jnp.exp(-a*(x2 + y2*(a + b)/e))/((1 - c*u2)*jnp.sqrt((1 - e2u2_pot)*e))
    #arr = q_pot*dens_pot*dens_lum*u2
    #print(dens_pot.max(),dens_lum.max(),f.max,jnp.exp(-a*(x2 + y2*(a + b)/e))/((1 - c*u2).max()) )

    G = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]

    return 4*jnp.pi**1.5*G*arr.sum((0, 1))

##############################################################################
def surf_v2los_cyl(x1, y1, inc,
                   dens_lum, sigma_lum, qintr_lum,
                   dens_pot, sigma_pot, qintr_pot,
                   beta, tensor):
    """
    This routine gives the projected weighted second moment Sigma*<V^2_los>

    """
    # sb_mu2 = jnp.empty_like(x1)
    # for j, (xj, yj) in enumerate(zip(x1, y1)):
    #     sb_mu2[j] = nq.quad(integrand_cyl_los, [0., 1.],
    #                        args=(dens_lum, sigma_lum, qintr_lum,
    #                               dens_pot, sigma_pot, qintr_pot,
    #                               xj, yj, inc, beta, tensor))
    def funArgs(Args):
        x,y = Args[0],Args[1]
        return nq.quad(integrand_cyl_los, [0., 1.],
                           args=(dens_lum, sigma_lum, qintr_lum,
                                  dens_pot, sigma_pot, qintr_pot,
                                  x, y, inc, beta, tensor))
    vfunArgs = jax.vmap(funArgs,0,0)

    sb_mu2 = jnp.squeeze( vfunArgs(jnp.array([x1,y1]).T) )

    return sb_mu2

##############################################################################
def vmom_proj(x1, y1, inc, mbh, beta, gamma, logistic,
              dens_lum, sigma_lum, qintr_lum,
              dens_pot, sigma_pot, qintr_pot,
              nrad, nang, nlos, epsrel, align, step):
    """
    This routine gives the projected first velocity moments
    and the second velocity moments tensor for a JAM model with
    either cylindrically or spherically-aligned velocity ellipsoid.
    The projection formulas given below are described in
    Sec.3 and the numerical quadrature in Sec.6.2 of
    Cappellari (2020, MNRAS, 494, 4819; hereafter C20)
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C

    """
    # TANH Change of variables for LOS integral (Sec.6.2 of Cappellari 2020)
    rmax = 3*jnp.max(sigma_lum)
    tmax = 8    # break is rmax/tmax
    t, dt = jnp.linspace(-tmax, tmax, nlos, retstep=True)
    scale = rmax/jnp.sinh(tmax)
    z1 = scale*jnp.sinh(t)
    dxdt = dt*scale*jnp.cosh(t)

    # Initialize moment values for interpolation
    irp = mom_interp(x1, y1,
                     dens_lum, sigma_lum, qintr_lum,
                     dens_pot, sigma_pot, qintr_pot,
                     mbh, beta, gamma, logistic, nrad, nang, epsrel,
                     rmin=step/jnp.sqrt(2), rmax=rmax, align=align)

    x = x1[:, None]
    y = z1*jnp.sin(inc) + y1[:, None]*jnp.cos(inc)                    # C20 eq.(29)
    z = z1*jnp.cos(inc) - y1[:, None]*jnp.sin(inc)
    R = jnp.sqrt(x**2 + y**2)
    r = jnp.sqrt(R**2 + z**2)
    cos_phi, sin_phi, sin_th, cos_th = x/R, y/R, R/r, z/r           # C20 eq.(30)

    mom = irp.get_moments(R.ravel(), z.ravel())
    sig2r, sig2th, sig2phi, v2phi, nu = jnp.reshape(jnp.array(mom), (5,) + z.shape)
    vphi = jnp.sqrt((v2phi - sig2phi).clip(0))   # Clip unphysical solutions
    diag = jnp.array([sig2r, sig2th, v2phi])
    zero = jnp.zeros_like(nu)
    one = jnp.ones_like(nu)
    
   
    if align == 'cyl':
        R = jnp.array([[cos_phi, zero, -sin_phi],
                      [sin_phi, zero, cos_phi],                     # C20 eq.(24)
                      [zero,    one,  zero]])                       # swap 2<->3 columns
    else:  # align == 'sph'
        R = jnp.array([[sin_th*cos_phi, cos_th*cos_phi, -sin_phi],
                      [sin_th*sin_phi, cos_th*sin_phi, cos_phi],    # C20 eq.(16)
                      [cos_th,         -sin_th,        zero]])

    S = jnp.array([[1,          0,            0],
                  [0, jnp.cos(inc), -jnp.sin(inc)],                   # C20 eq.(17)
                  [0, jnp.sin(inc), jnp.cos(inc)]])

    Q = jnp.einsum('ij,jkml->ikml', S, R)
    integ1 = vphi*Q[:, 2]                                           # C20 eq.(21)
    integ2 = jnp.einsum('jiml,kiml,iml->jkml', Q, Q, diag)           # C20 eq.(22)

    surf = nu @ dxdt                # DE quadrature
    nu_vlos = nu*integ1 @ dxdt      # DE quadrature
    nu_v2los = nu*integ2 @ dxdt     # DE quadrature
    vel = nu_vlos/surf
    vel2 = nu_v2los/surf

    
    return vel, vel2

##############################################################################
def mge_surf(x, y, surf, sigma, qobs):
    """ MGE surface brightness for a set of coordinates (x, y) """

    mge = surf*jnp.exp(-0.5/sigma**2*(x[..., None]**2 + (y[..., None]/qobs)**2))

    return mge.sum(-1)

##############################################################################

def bilinear_interpolate(xv, yv, im, xout, yout):
    """
    Interpolate the array `im` with values on a regular grid of coordinates
    `(xv, yv)` onto a new set of generic coordinates `(xout, yout)`.`
    The input array has size `im[ny, nx]` like `im = f(meshgrid(xv, yv))`.
    `xv` and `yv` are vectors of size `nx` and `ny` respectively.

    """
    ny, nx = im.shape
    # assert (nx, ny) == (xv.size, yv.size), "Input arrays dimensions do not match"

    xi = (nx - 1.)/(xv[-1] - xv[0])*(xout - xv[0])
    yi = (ny - 1.)/(yv[-1] - yv[0])*(yout - yv[0])

    return ndimage.map_coordinates(im.T, [xi, yi], order=1, mode='nearest')

##############################################################################
def rotate_points(x, y, ang):
    """
    Rotates points counter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013

    """
    theta = jnp.radians(ang)
    xNew = x*jnp.cos(theta) - y*jnp.sin(theta)
    yNew = x*jnp.sin(theta) + y*jnp.cos(theta)

    return xNew, yNew

##############################################################################

def psf_conv(x, y, inc_deg,
             surf_lum, sigma_lum, qobs_lum,
             surf_pot, sigma_pot, qobs_pot,
             mbh, beta, gamma, logistic, moment, align, sigmaPsf, normPsf,
             pixSize, pixAng, step, nrad, nang, nlos, epsrel, interp, analytic_los):
    """
    This routine gives the velocity moment after convolution with a PSF.
    The convolution is done using interpolation of the model on a
    polar grid, as described in Appendix A of Cappellari (2008, MNRAS, 390, 71)
    https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C

    """
    # Axisymmetric deprojection of both luminous and total mass.
    # See equation (12)-(14) of Cappellari (2008)
    inc = jnp.radians(inc_deg)
    qmin = 0.05   # Minimum desired intrinsic axial ratio
    qobsmin = jnp.sqrt(jnp.cos(inc)**2 + (qmin*jnp.sin(inc))**2)

    # assert jnp.all(qobs_lum >= qobsmin), f'Inclination too low q_lum < {qmin}'
    qintr_lum = jnp.sqrt(qobs_lum**2 - jnp.cos(inc)**2)/jnp.sin(inc)
    dens_lum = surf_lum*qobs_lum/(sigma_lum*qintr_lum*jnp.sqrt(2*jnp.pi))

    # assert jnp.all(qobs_pot >= qobsmin), f'Inclination too low q_pot < {qmin}'
    qintr_pot = jnp.sqrt(qobs_pot**2 - jnp.cos(inc)**2)/jnp.sin(inc)
    dens_pot = surf_pot*qobs_pot/(sigma_pot*qintr_pot*jnp.sqrt(2*jnp.pi))

    # Define parameters of polar grid for interpolation
    w = sigma_lum < jnp.max(jnp.abs(x))  # Characteristic MGE axial ratio in observed range
    #qmed = jnp.median(qobs_lum) if w.sum() < 3 else jnp.median(qobs_lum[w])
    ####################Change the if statement to use nanmedian plus jnp.select###############
    #This is a work for concreate boolean sums which numpyro hates
    qmed = jax.lax.select(w.sum() < 3, jnp.nanmedian(jnp.where(w, qobs_lum, jnp.nan)), jnp.median(qobs_lum))
    rell = jnp.sqrt(x**2 + (y/qmed)**2)  # Elliptical radius of input (x, y)

    # psf_convolution = (jnp.max(sigmaPsf) > 0) and (pixSize > 0)
    psf_convolution  = True

   
    if not interp or ((nrad*nang > x.size) and (not psf_convolution)):  # Just calculate values

        # assert jnp.all((x != 0) | (y != 0)), "One must avoid the singularity at `(xbin, ybin) = (0, 0)`"

        x_pol = x
        y_pol = y
        step = jnp.min(rell)  # Minimum radius

    else:  # Interpolate values on polar grid

        # Kernel step is 1/4 of largest between sigma(min) and 1/2 pixel side.
        # Kernel half size is the sum of 3*sigma(max) and 1/2 pixel diagonal.

        if psf_convolution:         # PSF convolution
            #if step == 0:
            step = jnp.min(sigmaPsf)/4
            mx = 3*jnp.max(sigmaPsf) + pixSize/jnp.sqrt(2)
        else:                       # No convolution
            step = jnp.min(rell)     # Minimum radius
            mx = 0

        # Make linear grid in log of elliptical radius RAD and eccentric anomaly ANG
        # See Appendix A of Cappellari (2008)
        rmax = jnp.max(rell) + mx  # Major axis of ellipse containing all data + convolution
        rad = jnp.geomspace(step/jnp.sqrt(2), rmax, nrad)  # Linear grid in jnp.log(rell)
        ang = jnp.linspace(0, jnp.pi/2, nang)  # Linear grid in eccentric anomaly
        radGrid, angGrid = map(jnp.ravel, jnp.meshgrid(rad, ang))
        x_pol = radGrid*jnp.cos(angGrid)
        y_pol = radGrid*jnp.sin(angGrid)*qmed
    
    # The model computation is only performed on the polar grid
    # which is then used to interpolate the values at any other location
    if analytic_los:
        # Analytic line-of-sight integral
        sb_mu2 = surf_v2los_cyl(x_pol, y_pol, inc,
                                dens_lum, sigma_lum, qintr_lum,
                                dens_pot, sigma_pot, qintr_pot, beta, moment)
        model = sb_mu2/mge_surf(x_pol, y_pol, surf_lum, sigma_lum, qobs_lum)
        vel = vel2 = None
    else:
        
        # Numeric line-of-sight integral
        vel, vel2 = vmom_proj(x_pol, y_pol, inc, mbh, beta, gamma, logistic,
                              dens_lum, sigma_lum, qintr_lum,
                              dens_pot, sigma_pot, qintr_pot,
                              nrad, nang, nlos, epsrel, align, step)
        
        model = {'xx': vel2[0, 0],
                 'yy': vel2[1, 1],
                 'zz': vel2[2, 2],
                 'xy': vel2[0, 1],
                 'xz': vel2[0, 2],
                 'yz': vel2[1, 2],
                 'x': vel[0],
                 'y': vel[1],
                 'z': vel[2]}[moment]   # match-case before Python 3.10
      
    t0 = time.time()
    if interp and psf_convolution:  # PSF convolution

        # nx = int(jnp.ceil(rmax/step))
        # ny = int(jnp.ceil(rmax*qmed/step))
        nx = 500
        ny = 500
        #print(nx, ny)
        x1 = jnp.linspace(0.5 - nx, nx - 0.5, 2*nx)*step
        y1 = jnp.linspace(0.5 - ny, ny - 0.5, 2*ny)*step
        x_car, y_car = jnp.meshgrid(x1, y1)  # Cartesian grid for convolution
        mge_car = mge_surf(x_car, y_car, surf_lum, sigma_lum, qobs_lum)
        
        # Interpolate moment over cartesian grid.
        # Interpolating "nu_v2/surf" instead of "nu_v2" or "jnp.log(nu_v2)" reduces interpolation error.
        r1 = 0.5*jnp.log(x_car**2 + (y_car/qmed)**2)  # Log elliptical radius of cartesian grid
        e1 = jnp.arctan2(jnp.abs(y_car/qmed), jnp.abs(x_car))    # Eccentric anomaly of cartesian grid
        model_car = mge_car*bilinear_interpolate(jnp.log(rad), ang, model.reshape(nang, nrad), r1, e1)
        # Calculation was done in positive quadrant: use symmetries
        if moment in ['xy', 'xz']:
            model_car *= jnp.sign(x_car*y_car)
        elif moment in ['y', 'z']:
            model_car *= jnp.sign(x_car)
        elif moment == 'x':
            model_car *= jnp.sign(y_car)
        
        #nk = int(jnp.ceil(mx/step))
        nk = 100
        #print(nk)
        kgrid = jnp.linspace(-nk, nk, 2*nk + 1)*step
        xgrid, ygrid = jnp.meshgrid(kgrid, kgrid)  # Kernel is square
        if pixAng != 0:
            xgrid, ygrid = rotate_points(xgrid, ygrid, pixAng)

        # Compute kernel with equation (A6) of Cappellari (2008).
        # Normalization is irrelevant here as it cancels out.
        dx = pixSize/2
        sp = jnp.sqrt(2)*sigmaPsf
        xg, yg = xgrid[..., None], ygrid[..., None]
        kernel = normPsf*(special.erf((dx - xg)/sp) + special.erf((dx + xg)/sp)) \
                        *(special.erf((dx - yg)/sp) + special.erf((dx + yg)/sp))
        kernel = kernel.sum(-1)   # Sum over PSF components

        # Seeing and aperture convolution with equation (A3) of Cappellari (2008)
        m1, m2 = signal.fftconvolve(jnp.array([model_car, mge_car]), kernel[None, ...], mode='same')
        # w = m2 > 0   # Allow for rounding errors
        # #muCar = jnp.zeros_like(m1)
        # #muCar[w] = m1[w]/m2[w]
        # muCar = jnp.where(w,m1/m2,0)
        mask = (m2 > 0).astype(m2.dtype)            # 0/1 掩码（常量参与乘法，不引分支）
        eps  = jnp.maximum(jnp.finfo(m2.dtype).eps, 1e-12)  # 比 tiny 更合适，避免过大比值
        den  = jnp.maximum(m2, eps)                  # 分母永不为 0
        muCar = (m1 / den) * mask                    # 无效像素置 0，形状全程静态



        # Interpolate convolved image at observed apertures.
        # Aperture integration was already included in the kernel.
        mu = bilinear_interpolate(x1, y1, muCar, x, y)
        
    else:  # No PSF convolution

        if not interp or (nrad*nang > x.size):      # Just returns values
            mu = model
        else:                      # Interpolate values
            r1 = 0.5*jnp.log(x**2 + (y/qmed)**2) # Log elliptical radius of input (x,y)
            e1 = jnp.arctan2(jnp.abs(y/qmed), jnp.abs(x))    # Eccentric anomaly of input (x,y)
            mu = bilinear_interpolate(jnp.log(rad), ang, model.reshape(nang, nrad), r1, e1)

            # Calculation was done in positive quadrant: use symmetries
            if moment in ('xy', 'xz'):
                mu *= jnp.sign(x*y)
            elif moment in ('y', 'z'):
                mu *= jnp.sign(x)
            elif moment == 'x':
                mu *= jnp.sign(y)

    return mu, psf_convolution, vel, vel2

##############################################################################

class jam_axi_proj:
    """
    jam_axi_proj
    ============

    Purpose
    -------

    This procedure calculates a prediction for all the projected first or second
    velocity moments for an anisotropic (three-integral) axisymmetric galaxy model.

    Any of the three components of the first velocity moment or any of the six
    components of the symmetric velocity dispersion tensor are supported.
    These include the line-of-sight velocities and the components of the proper motion.

    Two assumptions for the orientation of the velocity ellipsoid are supported:

    - The cylindrically-aligned ``(R, z, phi)`` solution was presented in
      `Cappellari (2008) <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_

    - The spherically-aligned ``(r, th, phi)`` solution was presented in
      `Cappellari (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C>`_

    Calling Sequence
    ----------------

    .. code-block:: python

        from jampy.jam_axi_proj import jam_axi_proj

        jam = jam_axi_proj(
                 surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                 inc, mbh, distance, xbin, ybin, align='cyl', analytic_los=True,
                 beta=None, data=None, epsrel=1e-2, errors=None, flux_obs=None,
                 gamma=None, goodbins=None, interp=True, kappa=None,
                 logistic=False, ml=None, moment='zz', nang=10, nlos=1500,
                 nodots=False, normpsf=1., nrad=20, pixang=0., pixsize=0.,
                 plot=True, quiet=False, rbh=0.01, sigmapsf=0., step=0.,
                 vmax=None, vmin=None)

        vrms = jam.model  # with moment='zz' the output is the LOS Vrms

        jam.plot()   # Generate data/model comparison when data is given

    See more examples in the ``jampy/examples`` folder inside
    `site-packages <https://stackoverflow.com/a/46071447>`_.

    Input Parameters
    ----------------

    surf_lum: array_like with shape (n,)
        peak surface values of the `Multi-Gaussian Expansion
        <https://pypi.org/project/mgefit/>`_ (MGE) Gaussians describing the
        surface brightness of the tracer population for which the kinematics
        is derived.

        The units are arbitrary as they cancel out in the final results.

        EXAMPLE: when one obtains the kinematics from optical spectroscopy,
        surf_lum contains the galaxy optical surface brightness, which has
        typical units of ``Lsun/pc^2`` (solar luminosities per ``parsec^2``).
    sigma_lum: array_like with shape (n,)
        dispersion (sigma) in arcseconds of the MGE Gaussians describing the
        distribution of the kinematic-tracer population.
    qobs_lum: array_like with shape (n,)
        observed axial ratio (q') of the MGE Gaussians describing the
        distribution of the kinematic-tracer population.
    surf_pot: array_like with shape (m,)
        peak value of the MGE Gaussians describing the galaxy total-mass
        surface density in units of ``Msun/pc^2`` (solar masses per ``parsec^2``).
        This is the MGE model from which the model gravitational potential is
        computed.

        EXAMPLE: with a self-consistent model, one has the same Gaussians
        for both the kinematic-tracer and the gravitational potential.
        This implies ``surf_pot = surf_lum``, ``sigma_pot = sigma_lum`` and
        ``qobs_pot = qobs_lum``. The global M/L of the model is fitted by the
        routine when passing the ``data`` and ``errors`` keywords with the
        observed kinematics.
    sigma_pot: array_like with shape (m,)
        dispersion in arcseconds of the MGE Gaussians describing the galaxy
        total-mass surface density.
    qobs_pot: array_like with shape (m,)
        observed axial ratio of the MGE Gaussians describing the galaxy
        total-mass surface density.
    inc: float
        inclination in degrees between the line-of-sight and the galaxy symmetry
        axis (0 being face-on and 90 edge-on).
    mbh: float
        Mass of a nuclear supermassive black hole in solar masses.

        IMPORTANT: The model predictions are computed assuming ``surf_pot``
        gives the total mass. In the self-consistent case, one has
        ``surf_pot = surf_lum`` and if requested (keyword ``ml``) the program
        can scale the output ``model`` to best fit the data. The scaling is
        equivalent to multiplying *both* ``surf_pot`` and ``mbh`` by a factor M/L.
        To avoid mistakes, the actual ``mbh`` used by the output model is
        printed on the screen.
    distance: float
        the distance of the galaxy in ``Mpc``. When the distance is derived 
        from redshift one should use the angular diameter distance ``D_A`` here.
    xbin: array_like with shape (p,)
        X coordinates in arcseconds of the bins (or pixels) at which one wants
        to compute the model predictions. The X-axis is assumed to coincide with
        the galaxy projected major axis. The galaxy center is at ``(0,0)``.
        
        In general the coordinates ``(xbin, ybin)`` have to be rotated to bring 
        the galaxy major axis on the X-axis, before calling ``jam_axi_proj``.

        When no PSF/pixel convolution is performed (``sigmapsf=0`` or
        ``pixsize=0``) there is a singularity at ``(0,0)`` which must be
        avoided by the user in the input coordinates.
    ybin: array_like with shape (p,)
        Y coordinates in arcseconds of the bins (or pixels) at which one wants
        to compute the model predictions. The Y-axis is assumed to coincide with
        the projected galaxy symmetry axis.

    Optional Keywords
    -----------------

    align: {'cyl', 'sph'}, optional.
        Assumed alignment for the velocity ellipsoid during the solution of
        the Jeans equations.

        - ``align='cyl'`` assumes a cylindrically-aligned velocity ellipsoid
          using the solution of `Cappellari (2008)`_

        - ``align='sph'`` assumes a spherically-aligned velocity ellipsoid
          using the solution of `Cappellari (2020)`_

    analytic_los: bool, optional
        This is ``True`` (default) if the line-of-sight integral is performed
        analytically and ``False`` if it is done via numerical quadrature.

        An analytic integral is only possible with ``align='cyl'`` and only for
        the second velocity moments. For this reason, when comparing the two
        second-moment solutions with ``align='cyl'`` and ``align='sph'``, it
        may be preferable to set ``analytic_los=False`` to ensure that
        numerical interpolation error is exactly the same in both cases.

        When ``align='sph'``, or when the user requests a first velocity
        moment, this keyword is automatically set to ``False``.
    beta: array_like with shape (n,) or (4,)
        Radial anisotropy of the individual kinematic-tracer MGE Gaussians
        (Default: ``beta=jnp.zeros(n)``)::

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
    data: array_like with shape (p,), optional
        observed first or second velocity moment used to fit the model.

        EXAMPLE: In the common case where one has only line-of-sight velocities
        the second moment is given by::

            Vrms = jnp.sqrt(velBin**2 + sigBin**2)

        at the coordinates positions given by the vectors ``xbin`` and ``ybin``.

        If ``data`` is set and ``ml`` is negative or ``None``, then the model
        is fitted to the data, otherwise, the adopted ``ml`` is used and just
        the ``chi**2`` is returned.
    epsrel: float, optional
        Relative error requested for the numerical computation of the intrinsic
        moments (before line-of-sight quadrature). (Default: ``epsrel=1e-2``)
    errors: array_like with shape (p,), optional
        1sigma uncertainty associated with the ``data`` measurements.

        EXAMPLE: In the case where the data are given by the
        ``Vrms = jnp.sqrt(velBin**2 + sigBin**2)``, from the error propagation::

            errors = jnp.sqrt((dVel*velBin)**2 + (dSig*sigBin)**2)/Vrms,

        where ``velBin`` and ``sigBin`` are the velocity and dispersion in each
        bin and ``dVel`` and ``dSig`` are the corresponding 1sigma uncertainties.
        (Default: constant ``errors = 0.05*jnp.median(data)``)
    flux_obs: array_like with shape (p,), optional
        Optional mean surface brightness of each bin for plotting.
    gamma: array_like with shape (n,)
        tangential anisotropy of the individual kinematic-tracer MGE Gaussians
        (Default: ``gamma=jnp.zeros(n)``)::

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

        IMPORTANT: ``gamma`` only affects the projected first velocity moments.
        The projected second moments are rigorously independent of ``gamma``.
    goodbins: array_like with shape (p,)
        Boolean vector with values ``True`` for the bins/spaxels which have to
        be included in the fit (if requested) and in the ``chi**2`` calculation.
        (Default: fit all bins).
    interp: bool, optional
        This keyword is for advanced use only! Set ``interp=False`` to force 
        no-interpolation on the sky plane. In this way ``jam.vel`` and 
        ``jam.vel2`` contain all the first and second velocity moments at the 
        input coordinates ``(xbin, ybin)``, without PSF convolution.
        By default ``interp=True`` and one should generally not change this.

        IMPORTANT: If ``sigmapsf=0`` or ``pixsize=0`` or ``interp=False`` then
        PSF convolution is not performed.

        This keyword is mainly useful for testing against analytic results or
        to compute all moments, including proper motions,  simultaneously.
    kappa: float, optional
        When ``kappa=None`` (default) the first velocity moments are scaled in
        such a way that the projected angular momentum of the data and model is
        the same [equation 52 of `Cappellari (2008)`_].
        When ``kappa=1`` the model first velocity moments are output without
        any scaling.
    logistic: bool, optional
        When ``logistic=True``, JAM interprets the anisotropy parameters
        ``beta`` and ``gamma`` as defining a 4-parameters logistic function.
        See the documentation of the anisotropy keywords for details.
        (Default ``logistic=False``)
    ml: float, optional
        Mass-to-light ratio (M/L) to multiply the values given by ``surf_pot``.
        Setting this keyword is completely equivalent to multiplying the
        output ``model`` by ``jnp.sqrt(M/L)`` after the fit. This implies that
        the BH mass is also scaled and becomes ``mbh*ml``.

        If ``ml=None`` (default) the M/L is fitted from the data and the
        best-fitting M/L is returned in output. The BH mass of the model is
        also scaled and becomes ``mbh*ml``.
    moment: {'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'}, optional
        String specifying the component of the velocity first or second moments
        requested by the user in output. All values ar in ``km/s``.

        - ``moment='x'`` gives the first moment ``<V_x'>`` of the proper motion
          in the direction orthogonal to the projected symmetry axis.

        - ``moment='y'`` gives the first moment ``<V_y'>`` of the proper motion
          in the direction parallel to the projected symmetry axis.

        - ``moment='z'`` gives the first moment ``Vlos = <V_z'>`` of the
          line-of-sight velocity.

        - ``moment='xx'`` gives ``sqrt<V_x'^2>`` of the component of the proper
          motion dispersion tensor in the direction orthogonal to the projected
          symmetry axis.

        - ``moment='yy'`` gives ``sqrt<V_y'^2>`` of the component of the proper
          motion dispersion tensor in the direction parallel to the projected
          symmetry axis.

        - ``moment='zz'`` (default) gives the usual line-of-sight
          ``Vrms = sqrt<V_z'^2>``.

        - ``moment='xy'`` gives the mixed component ``<V_x'V_y'>`` of the proper
          motion dispersion tensor.

        - ``moment='xz'`` gives the mixed component ``<V_x'V_z'>`` of the proper
          motion dispersion tensor.

        - ``moment='yz'`` gives the mixed component ``<V_y'V_z'>`` of the proper
          motion dispersion tensor.
    nang: int, optional
        The number of linearly-spaced intervals in the eccentric anomaly at
        which the model is evaluated before interpolation and PSF convolution.
        (default: ``nang=10``)
    nlos: int (optional)
        Number of values used for the numerical line-of-sight quadrature.
        (default ``nlos=1500``)
    nodots: bool, optional
        Set to ``True`` to hide the dots indicating the centers of the bins in
        the linearly-interpolated two-dimensional map (default ``False``).
    normpsf: array_like with shape (q,)
        fraction of the total PSF flux contained in the circular Gaussians
        describing the PSF of the kinematic observations.
        The PSF will be used for seeing convolution of the model kinematics.
        It has to be ``jnp.sum(normpsf) = 1``.
    nrad: int, optional
        The number of logarithmically spaced radial positions at which the
        model is evaluated before interpolation and PSF convolution. One may
        want to increase this value if the model has to be evaluated over many
        orders of magnitude in radius (default: ``nrad=20``).
    pixang: float, optional
        Angle between the observed spaxels and the galaxy major axis X.
        This angle only rotates the spaxels around their centers, *not* the
        whole coordinate system ``(xbin, ybin)``, which must be rotated
        independently by the user before calling ``jam_axi_proj``. 
        Using the keyword is generally unnecessary.
    pixsize: float, optional
        Size in arcseconds of the (square) spatial elements at which the
        kinematics is obtained. This may correspond to the side of the spaxel
        or lenslets of an integral-field spectrograph. This size is used to
        compute the kernel for the seeing and aperture convolution.

        IMPORTANT: If ``sigmapsf=0`` or ``pixsize=0`` or ``interp=False`` then
        PSF convolution is not performed.
    plot: bool
        When ``data is not None`` setting this keyword produces a plot with the
        data/model comparison at the end of the calculation.
    quiet: bool
        Set this keyword to avoid printing values on the console.
    rbh: float, optional
        This keyword is ignored unless ``align='cyl'`` and `analytic_los=True`.
        In all other cases JAM assume a point-like central black hole.
        This scalar gives the sigma in arcsec of the Gaussian approximating the
        central black hole of mass MBH [See Section 3.1.2 of `Cappellari (2008)`_]
        The gravitational potential is indistinguishable from a point source
        for ``radii > 2*rbh``, so the default ``rbh=0.01`` arcsec is appropriate
        in most current situations.

        When using different units as input, e.g. pc instead of arcsec, one
        should check that ``rbh`` is not too many order of magnitude smaller
        than the spatial resolution of the data.
    sigmapsf: array_like with shape (q,)
        dispersion in arcseconds of the circular Gaussians describing the PSF
        of the kinematic observations.

        IMPORTANT: If ``sigmapsf=0`` or ``pixsize=0`` or ``interp=False`` then
        PSF convolution is not performed.

        IMPORTANT: PSF convolution is done by creating a 2D image, with pixels
        size given by ``step=max(sigmapsf, pixsize/2)/4``, and convolving it
        with the PSF + aperture. If the input radii are very large compared
        to ``step``, the 2D image may require a too large amount of memory.
        If this is the case one may compute the model predictions at small radii
        with a first call to ``jam_axi_proj`` with PSF convolution, and the
        model predictions at large radii with a second call to ``jam_axi_proj``
        without PSF convolution.
    step: float, optional
        Spatial step for the model calculation and PSF convolution in arcsec.
        This value is automatically computed by default as
        ``step=max(sigmapsf,pixsize/2)/4``. It is assumed that when ``pixsize``
        or ``sigmapsf`` are large, high-resolution calculations are not needed. In
        some cases, however, e.g. to accurately estimate the central Vrms in a
        very cuspy galaxy inside a large aperture, one may want to override the
        default value to force smaller spatial pixels using this keyword.
    vmax: float, optional
        Maximum value of the ``data`` to plot.
    vmin: float, optional
        Minimum value of the ``data`` to plot.

    Output Parameters
    -----------------

    Stored as attributes of the ``jam_axi_proj`` class.

    .chi2: float
        Reduced ``chi**2``, namely per degree of freedom,  describing the 
        quality of the fit::

            d, m = (data/errors)[goodbins], (model/errors)[goodbins]
            chi2 = ((d - m)**2).sum()/goodbins.sum()

        When no data are given in input, this is returned as ``jnp.nan``.
    .flux: array_like with shape (p,)
        PSF-convolved MGE surface brightness of each bin in ``Lsun/pc^2``,
        used to plot the isophotes of the kinematic-tracer on the model results.
    .kappa: float
        Ratio by which the model was scaled to fit the observed velocity
        [defined by equation 52 of `Cappellari (2008)`_]
    .ml: float
        Best fitting M/L by which the mass was scaled to fit the observed moments.
    .model: array_like with shape (p,)
        Model predictions for the selected velocity moments for each input bin
        ``(xbin, ybin)``. This attribute is the main output from the program.

        Any of the six components of the symmetric proper motion dispersion
        tensor ``{'xx', 'yy', 'zz', 'xy', 'xz', 'yz'}``, or any of the three 
        first velocity moments ``{'x', 'y', 'z'}``` can be returned in output.
        The desired model output is selected using the ``moment`` keyword.
        See the ``moment`` documentation for details.
    .vel: array_like with shape (3, p)
        This attribute generally contains an intermediate result of the
        calculation and should not be used. Instead, the output kinematic
        model predictions are contained in the ``.model`` attribute.

        However, for advanced use only, when setting ``interp=False`` and
        ``analytic_los=False``, this attribute contains the first velocity
        moments for all the x, y and z components, *not* PSF convolved, at the
        sky coordinates ``(xbin, ybin)``.
    .vel2: array_like with shape (3, 3, p)
        This attribute generally contains an intermediate result of the
        calculation and should not be used. Instead, the output kinematic
        model predictions are contained in the ``.model`` attribute.

        However, for advanced use only, when setting ``interp=False`` and
        ``analytic_los=False``, this attribute contains the full 3x3 second
        velocity moment tensor, *not* PSF convolved, at the sky coordinates
        ``(xbin, ybin)``.

    ###########################################################################
    """
    
    @staticmethod
    def get_kinematics( surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot,
                 qobs_pot, inc, mbh, distance, xbin, ybin, align='cyl',
                 analytic_los=True, beta=None, data=None, epsrel=1e-2,
                 errors=None, flux_obs=None, gamma=None, goodbins=None,
                 interp=True, kappa=None, logistic=False, ml=None, moment='zz',
                 nang=10, nlos=1500, nodots=False, normpsf=1., nrad=20,
                 pixang=0., pixsize=0., quiet=False, rbh=0.01,
                 sigmapsf=0., step=0.):

        str1 = ['x', 'y', 'z']
        str2 =  ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        # assert moment in str1 + str2, f"`moment` must be one of {str1 + str2}"
        # assert align in ['sph', 'cyl'], "`align` must be 'sph' or 'cyl'"
        # assert (ml is None) or (ml > 0), "The input `ml` must be positive"
        if beta is None:
            beta = jnp.zeros_like(surf_lum)  # Anisotropy parameter beta = 1 - (sig_th/sig_r)**2
        if gamma is None:  # Anisotropy parameter beta = 1 - (sig_th/sig_r)**2
            if logistic:
                gamma = [1, 0, 0, 1]
            else:
                gamma = jnp.zeros_like(beta)
        # assert (surf_lum.size == sigma_lum.size == qobs_lum.size) \
            #    and ((len(beta) == 4 and logistic) or (len(beta) == surf_lum.size)) \
            #    and (len(beta) == len(gamma)), "The luminous MGE components and anisotropies do not match"
        # assert surf_pot.size == sigma_pot.size == qobs_pot.size, "The total-mass MGE components do not match"
        # assert xbin.size == ybin.size, "`xbin` and `ybin` do not match"
        if (not interp) or (moment in str1) or (align == 'sph') or logistic:
            analytic_los = False
        if data is not None:
            if errors is None:
                if moment in str2:
                    errors = jnp.full_like(data, jnp.median(data)*0.05)  # Constant ~5% errors
                else:
                    errors = jnp.full_like(data, 5.)  # Constant 5 km/s errors
            if goodbins is None:
                goodbins = jnp.ones_like(data, dtype=bool)
            # else:
                # assert goodbins.dtype == bool, "goodbins must be a boolean vector"
                # assert jnp.any(goodbins), "goodbins must contain some True values"
            # assert xbin.size == data.size == errors.size == goodbins.size, \
                # "(rms, erms, goodbins) and (xbin, ybin) do not match"

        sigmapsf = jnp.atleast_1d(sigmapsf)
        normpsf = jnp.atleast_1d(normpsf)
        # assert sigmapsf.size == normpsf.size, "sigmaPSF and normPSF do not match"
        # assert round(jnp.sum(normpsf), 2) == 1, "PSF not normalized"

        # Convert all distances to pc
        pc = distance*jnp.pi/0.648  # Factor to convert arcsec --> pc (with distance in Mpc)
        surf_lum_pc = surf_lum
        surf_pot_pc = surf_pot
        sigma_lum_pc = sigma_lum*pc
        sigma_pot_pc = sigma_pot*pc
        xbin_pc = xbin*pc
        ybin_pc = ybin*pc
        pixSize_pc = pixsize*pc
        sigmaPsf_pc = sigmapsf*pc
        step_pc = step*pc

        # Assumes beta = [r_a, beta_0, beta_inf, alpha]
        #        gamma = [r_a, gamma_0, gamma_inf, alpha]
        if logistic:
            beta = beta.copy()
            gamma = jnp.array(gamma.copy(), dtype=jnp.float64)
            beta = beta.at[0].set(beta[0]*pc)  
            gamma = gamma.at[0].set(gamma[0]*pc)  

        # Add a Gaussian with small sigma and the same total mass as the BH.
        # The Gaussian provides an excellent representation of the second moments
        # of a point-like mass, to 1% accuracy out to a radius 2*sigmaBH.
        # The error increases to 14% at 1*sigmaBH, independently of the BH mass.
        if mbh > 0 and analytic_los:
            tmp = jnp.concatenate([sigmapsf, jnp.array([pixsize]), sigma_lum])
            # assert rbh > 0.01*jnp.min(tmp[tmp > 0]), "`rbh` is too small"
            sigmaBH_pc = rbh*pc # Adopt for the BH just a very small size
            surfBH_pc = mbh/(2*jnp.pi*sigmaBH_pc**2)
            surf_pot_pc = jnp.append(surfBH_pc, surf_pot_pc) # Add Gaussian to potential only!
            sigma_pot_pc = jnp.append(sigmaBH_pc, sigma_pot_pc)
            qobs_pot = jnp.append(1., qobs_pot)  # Make sure vectors do not have extra dimensions

        qobs_lum = qobs_lum.clip(0, 0.999999)
        qobs_pot = qobs_pot.clip(0, 0.999999)

        t = clock()
        model, psfConvolution, vel, vel2 = psf_conv(
            xbin_pc, ybin_pc, inc,
            surf_lum_pc, sigma_lum_pc, qobs_lum,
            surf_pot_pc, sigma_pot_pc, qobs_pot,
            mbh, beta, gamma, logistic, moment, align, sigmaPsf_pc, normpsf,
            pixSize_pc, pixang, step_pc, nrad, nang, nlos, epsrel,
            interp, analytic_los)

        if moment in str2[:3]:
            model = jnp.sqrt(model.clip(0))  # sqrt and clip to allow for rounding errors

        # Analytic convolution of the MGE model with an MGE circular PSF
        # using Equations (4,5) of Cappellari (2002, MNRAS, 333, 400).
        # Broadcast triple loop over (n_MGE, n_PSF, n_bins)
        sigmaX2 = sigma_lum**2 + sigmapsf[:, None]**2
        sigmaY2 = (sigma_lum*qobs_lum)**2 + sigmapsf[:, None]**2
        surf_conv = surf_lum_pc*qobs_lum*sigma_lum**2*normpsf[:, None]/jnp.sqrt(sigmaX2*sigmaY2)
        flux = surf_conv[..., None]*jnp.exp(-0.5*(xbin**2/sigmaX2[..., None] + ybin**2/sigmaY2[..., None]))
        flux = flux.sum((0, 1))  # PSF-convolved Lsun/pc^2

        if data is None:

            chi2 = jnp.nan
            if moment in str2[:3]:
                if ml is None:
                    ml = kappa = 1.0
                else:
                    kappa = 1.0
                    model *= jnp.sqrt(ml)
            else:
                if kappa is None:
                    ml = kappa = 1.0
                else:
                    ml = 1.0
                    model *= kappa

        else:

            d, m = (data/errors)[goodbins], (model/errors)[goodbins]

            if moment in str2[:3]:

                if ml is None:
                    ml = ((d @ m)/(m @ m))**2   # eq. (51) of Cappellari (2008, MNRAS)

                scale, kappa = jnp.sqrt(ml), 1.0

            else:

                if kappa is None:

                    # Scale by having the same angular momentum in the model and
                    # in the galaxy with eq. (52) of Cappellari (2008, MNRAS)
                    kappa = jnp.abs(data*xbin)[goodbins].sum()/jnp.abs(model*xbin)[goodbins].sum()

                    # Measure the scaling one would have from a standard chi^2 fit of the V field.
                    # This value is only used to get proper sense of rotation for the model.
                    kappa1 = (d @ m)/(m @ m)  # eq. (51) of Cappellari (2008, MNRAS) not squared
                    kappa *= jnp.sign(kappa1)

                scale, ml = kappa, 1.0

            model *= scale
            m *= scale
            chi2 = ((d - m)**2).sum()/goodbins.sum()

        if not quiet:
            print(f'jam_axi_proj_{align}_{moment} (analytic_los={analytic_los}) '
                  f'elapsed time sec: {clock() - t:.2f}')
            if (not psfConvolution) or (not interp):
                txt = "No PSF/pixel convolution:"
                if jnp.max(sigmapsf) == 0:
                    txt += " sigmapsf == 0;"
                if pixsize == 0:
                    txt += " pixsize == 0;"
                if not interp:
                    txt += " interp == False;"
                print(txt)
            print(f'inc={inc:#.3g}; beta[1]={beta[1]:#.2g}; kappa={kappa:#.3g}; '
                  f'M/L={ml:#.3g}; BH={mbh*ml:#.2g}; chi2/DOF={chi2:#.3g}')
            mass = 2*jnp.pi*surf_pot_pc*qobs_pot*sigma_pot_pc**2
            print(f'Total mass MGE (MSun): {(mass*ml).sum():#.4g}')

        return model, chi2, flux, ml, vel, vel2

##############################################################################
    
    @staticmethod
    def plot(goodbins,moment,model,xbin,ybin,data,align,flux_obs, nodots=False):

        ok = goodbins
        str1 = ['x', 'y', 'z']
        sym = 1 if moment in str1 else 2
        #data1 = self.data.copy()  # Only symmetrize good bins
        #data1[ok] = symmetrize_velfield(self.xbin[ok], self.ybin[ok], data1[ok], sym=sym)
        data1 = jnp.where(ok,symmetrize_velfield( xbin,  ybin,  data, sym=sym), data)

        if  moment in str1 + ['xy', 'xz']:
            vmax = jnp.percentile(jnp.abs(data1[ok]), 99)
            vmin = -vmax
        else:
            vmin, vmax = jnp.percentile(data1[ok], jnp.array([0.5, 99.5]) )

        plt.clf()
        plt.subplot(121)
        plot_velfield( xbin,  ybin, data1, vmin=vmin, vmax=vmax, flux= flux_obs, nodots=nodots)
        plt.title(f"Input V$_{{{ moment}}}$ moment")

        plt.subplot(122)
        model = jnp.where(jnp.isnan( model),0,model)
        plot_velfield(xbin, ybin, jnp.clip(model,-1e3,1e3), vmin=vmin, vmax=vmax, flux=flux_obs, nodots=nodots)
        plt.plot(xbin[~ok], ybin[~ok], 'ok', mec='white')
        plt.title(f"JAM$_{{\\rm {align}}}$ model")
        plt.tick_params(labelleft=False)
        plt.subplots_adjust(wspace=0.03)

##############################################################################

