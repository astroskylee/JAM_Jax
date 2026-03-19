import jax
from jax import jit
import jax.numpy as jnp
from scipy.special import roots_legendre
from jax.tree_util import Partial


from jax import config
config.update("jax_enable_x64", True)

def def_nth_order_func_spec_quad(func,n=20):
    # scipy.quad written in jax

    xval, weights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    weights = weights.reshape(-1, 1)

    @jit
    def integrate( a, b, args=()):
        # Integrate function with args from a to b

        aux = jnp.apply_along_axis(func,1,0.5*((b - a) * xval + (b + a)),*args)
        res = 0.5*(b-a)*jnp.sum( weights*aux, axis=0 )
        
        return res
    
    return integrate

def def_nth_order_quad(n=20):
    #scipy.quad written in jax

    xval, weights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    weights = weights.reshape(-1, 1)
    
    def integrate(func, ranges, args=()):
        #Integrate function with args from a to b
        a,b= ranges[0], ranges[1]
        vafunc = jax.vmap(lambda x : func(x,*args), 0, 0)
        aux = vafunc( 0.5*((b - a) * xval + (b + a)) )
        #aux = jnp.apply_along_axis(func,1,0.5*((b - a) * xval + (b + a)),*args)
        res = 0.5*(b-a)*jnp.sum( weights*aux, axis=0 )
        
        return res
    
    return integrate
    
def def_segmented_quad(n=20, segments=8, log_L=3.0):
    """
    Fixed-order GL on pre-defined sub-intervals of [a, b].

    Segmentation is log-spaced and clustered near the lower bound ('log0').
    """
    xval, weights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)       # (n, 1)
    wr = weights.reshape(-1)         # (n,)
    segments = int(segments)
    if segments < 1:
        raise ValueError("segments must be >= 1")

    t = jnp.logspace(-log_L, 0.0, segments + 1)
    edges = (t - 10.0**(-log_L)) / (1.0 - 10.0**(-log_L))

    edges = edges.at[0].set(0.0)
    edges = edges.at[-1].set(1.0)

    def integrate(func, ranges, args=()):
        # Integrate function with args from a to b
        a, b = ranges[0], ranges[1]
        left = edges[:-1]
        right = edges[1:]

        # Map [0,1] segment edges to [a,b]
        l = a + (b - a) * left
        r = a + (b - a) * right

        # GL nodes on each segment: (segments, n)
        x = 0.5 * ((r - l)[:, None] * xval.T + (r + l)[:, None])
        xflat = x.reshape(-1, 1)

        vafunc = jax.vmap(lambda u: func(u, *args), 0, 0)
        aux = vafunc(xflat)  # (segments*n, ...)
        aux = aux.reshape((segments, xval.shape[0]) + aux.shape[1:])

        weighted = jnp.tensordot(aux, wr, axes=([1], [0]))  # (segments, ...)
        scale = 0.5 * (r - l)
        if weighted.ndim == 1:
            seg_int = weighted * scale
        else:
            seg_int = weighted * scale.reshape((segments,) + (1,) * (weighted.ndim - 1))

        return jnp.sum(seg_int, axis=0)

    return integrate

def _segmentation_edges(segments=8, spacing="log", log_L=3.0):
    segments = int(segments)
    if segments < 1:
        raise ValueError("segments must be >= 1")

    if spacing == "log":
        t = jnp.logspace(-log_L, 0.0, segments + 1)
        edges = (t - 10.0**(-log_L)) / (1.0 - 10.0**(-log_L))
    elif spacing == "uniform":
        edges = jnp.linspace(0.0, 1.0, segments + 1)
    else:
        raise ValueError("spacing must be 'log' or 'uniform'")

    edges = edges.at[0].set(0.0)
    edges = edges.at[-1].set(1.0)
    return edges

def def_segmented_firstdim_nquad(n=20, segments=8, spacing="log", log_L=3.0):
    """
    Fixed-order tensor-product GL where only the first integration dimension
    is subdivided into pre-defined segments.

    This is intended for the sph JAM path, where the first dimension maps the
    Chandrasekhar u-integral and typically benefits most from extra resolution
    near the lower bound.
    """
    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    edges = _segmentation_edges(segments=segments, spacing=spacing, log_L=log_L)

    def integrate(func, ranges, args=()):
        args = tuple(args)
        ranges = jnp.asarray(ranges)
        ndim = len(ranges)

        a0, b0 = ranges[0, 0], ranges[0, 1]
        left = edges[:-1]
        right = edges[1:]
        l0 = a0 + (b0 - a0) * left
        r0 = a0 + (b0 - a0) * right

        x0 = 0.5 * ((r0 - l0)[:, None] * xval.T + (r0 + l0)[:, None]).reshape(-1)
        w0 = (0.5 * (r0 - l0)[:, None] * legweights[None, :]).reshape(-1)

        spaces = [x0]
        weights = [w0]

        if ndim > 1:
            other_ranges = ranges[1:]
            other_spaces = (
                ((other_ranges[:, 1] - other_ranges[:, 0])[jnp.newaxis, :] * xval
                 + (other_ranges[:, 1] + other_ranges[:, 0])[jnp.newaxis, :]) * 0.5
            ).T
            other_weights = (
                0.5 * (other_ranges[:, 1] - other_ranges[:, 0])[:, jnp.newaxis]
                * jnp.tile(jnp.array([legweights]), (ndim - 1, 1))
            )

            spaces.extend(list(other_spaces))
            weights.extend(list(other_weights))

        meshes = jnp.meshgrid(*spaces, indexing="ij")
        weight_meshes = jnp.meshgrid(*weights, indexing="ij")
        flat_meshes = tuple(mesh.ravel() for mesh in meshes)
        argsfull = flat_meshes + args
        funceval = func(*argsfull)
        weights_prod = jnp.prod(jnp.array(weight_meshes), axis=0).ravel()
        res = jnp.sum(weights_prod * funceval, axis=-1)

        return res

    return integrate


# Default segmented strategy selected from 50x50 benchmark:
# max|ΔVrms| <= 0.11 km/s with lower runtime than uniform segmentation.
DEFAULT_QUAD_NAME = "logspace_segmentation"
DEFAULT_N = 20
DEFAULT_SEGMENTS = 8
DEFAULT_LOG_L = 3.0

quad = def_segmented_quad(n=DEFAULT_N, segments=DEFAULT_SEGMENTS, log_L=DEFAULT_LOG_L)

# Keep the sph 2D integral on the scanned default:
# segment only the first dimension, use log spacing, and keep 8 segments.
# This is the best residual/runtime trade-off among the tested sph settings.
DEFAULT_NQUAD_NAME = "first_dim_logspace_segmentation"
DEFAULT_NQUAD_SEGMENTS = 8
DEFAULT_NQUAD_SPACING = "log"
DEFAULT_NQUAD_LOG_L = 3.0

print(
    f"[jaxpy.nquad] default quadrature: {DEFAULT_QUAD_NAME}, "
    f"seg={DEFAULT_SEGMENTS}, n={DEFAULT_N}, log_L={DEFAULT_LOG_L}"
)

def def_nth_order_func_spec_nquad(func,n=20):
    #scipy.quad written in jax

    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    
    @jit
    def integrate(ranges, args=()):
        #Integrate function with args from a to b
        args = tuple(args)

        Ndim = len(ranges)
        spaces = ( ((ranges[:,1]-ranges[:,0])[jnp.newaxis,:] * xval  +  (ranges[:,1]+ranges[:,0])[jnp.newaxis,:] ) *0.5 ).T
        meshes = jnp.meshgrid(*spaces)
        weights = 0.5*(ranges[:,1]-ranges[:,0])[:,jnp.newaxis] * jnp.tile(jnp.array([legweights]),(Ndim,1))
        weightsmesh = jnp.meshgrid(*weights)
        flat_meshes = tuple(mesh.ravel() for mesh in meshes)
        argsfull = flat_meshes + args
        funceval = func(*argsfull)
        weightsmeshprod = jnp.prod(jnp.array(weightsmesh),axis=0).ravel()
        res = jnp.sum(weightsmeshprod*funceval, axis=-1)
        
        return res
    
    return integrate

def def_nth_order_range_spec_nquad(ranges,n=20):
    #scipy.quad written in jax

    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    Ndim = len(ranges)
    spaces = ( ((ranges[:,1]-ranges[:,0])[jnp.newaxis,:] * xval  +  (ranges[:,1] + ranges[:,0])[jnp.newaxis,:] ) * 0.5 ).T
    meshes = jnp.meshgrid(*spaces)
    weights = 0.5*(ranges[:,1]-ranges[:,0])[:,jnp.newaxis] * jnp.tile(jnp.array([legweights]), (Ndim,1))
    weightsmesh = jnp.meshgrid(*weights)
    weightsmeshprod = jnp.prod(jnp.array(weightsmesh),axis=0)
        
    def integrate(func, args=()):
        #Integrate function with args from a to b
        args = tuple(args)

        flat_meshes = tuple(mesh.ravel() for mesh in meshes)
        argsfull = flat_meshes + args
        funceval = func(*argsfull)
        res = jnp.sum(weightsmeshprod.ravel()*funceval, axis=-1)
        
        return res
    
    return integrate

def def_nth_order_nquad(n=20):
    #scipy.quad written in jax

    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    
    def integrate(func, ranges, args=()):
        #Integrate function with args from a to b
        args = tuple(args)

        Ndim = len(ranges)
        spaces = ( ((ranges[:,1]-ranges[:,0])[jnp.newaxis,:] * xval  +  (ranges[:,1]+ranges[:,0])[jnp.newaxis,:] ) *0.5 ).T
        meshes = jnp.meshgrid(*spaces)
        weights = 0.5*(ranges[:,1]-ranges[:,0])[:,jnp.newaxis] * jnp.tile(jnp.array([legweights]),(Ndim,1))
        weightsmesh = jnp.meshgrid(*weights)
        flat_meshes = tuple(mesh.ravel() for mesh in meshes)
        argsfull = flat_meshes + args
        funceval = func(*argsfull)
        weightsmeshprod = jnp.prod(jnp.array(weightsmesh),axis=0).ravel()
        res = jnp.sum(weightsmeshprod*funceval, axis=-1)
        
        return res
    
    return integrate

pure_gl_nquad = def_nth_order_nquad(n=DEFAULT_N)
nquad = def_segmented_firstdim_nquad(
    n=DEFAULT_N,
    segments=DEFAULT_NQUAD_SEGMENTS,
    spacing=DEFAULT_NQUAD_SPACING,
    log_L=DEFAULT_NQUAD_LOG_L,
)

print(
    f"[jaxpy.nquad] default sph nquad: {DEFAULT_NQUAD_NAME}, "
    f"seg={DEFAULT_NQUAD_SEGMENTS}, spacing={DEFAULT_NQUAD_SPACING}, "
    f"n={DEFAULT_N}, log_L={DEFAULT_NQUAD_LOG_L}"
)
