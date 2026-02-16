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
    
quad = def_nth_order_quad(n=20)

def def_nth_order_func_spec_nquad(func,n=20):
    #scipy.quad written in jax

    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    
    @jit
    def integrate(ranges, args=()):
        #Integrate function with args from a to b

        Ndim = len(ranges)
        spaces = ( ((ranges[:,1]-ranges[:,0])[jnp.newaxis,:] * xval  +  (ranges[:,1]+ranges[:,0])[jnp.newaxis,:] ) *0.5 ).T
        meshes = jnp.meshgrid(*spaces)
        weights = 0.5*(ranges[:,1]-ranges[:,0])[:,jnp.newaxis] * jnp.tile(jnp.array([legweights]),(Ndim,1))
        weightsmesh = jnp.meshgrid(*weights)
        argsfull = tuple(meshes) + args
        funceval = func(*argsfull)
        weightsmeshprod = jnp.prod(jnp.array(weightsmesh),axis=0)
        res = jnp.sum(weightsmeshprod*funceval )
        
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

        argsfull = tuple(meshes) + args
        funceval = func(*argsfull)
        res = jnp.sum(weightsmeshprod*funceval)
        
        return res
    
    return integrate

def def_nth_order_nquad(n=20):
    #scipy.quad written in jax

    xval, legweights = map(jnp.array, roots_legendre(n))
    xval = xval.reshape(-1, 1)
    
    def integrate(func, ranges, args=()):
        #Integrate function with args from a to b

        Ndim = len(ranges)
        spaces = ( ((ranges[:,1]-ranges[:,0])[jnp.newaxis,:] * xval  +  (ranges[:,1]+ranges[:,0])[jnp.newaxis,:] ) *0.5 ).T
        meshes = jnp.meshgrid(*spaces)
        weights = 0.5*(ranges[:,1]-ranges[:,0])[:,jnp.newaxis] * jnp.tile(jnp.array([legweights]),(Ndim,1))
        weightsmesh = jnp.meshgrid(*weights)
        argsfull = tuple(meshes) + args
        funceval = func(*argsfull)
        weightsmeshprod = jnp.prod(jnp.array(weightsmesh),axis=0)
        res = jnp.sum(weightsmeshprod*funceval )
        
        return res
    
    return integrate

nquad = def_nth_order_nquad(n=20)
