from numpy import linalg,zeros,ones,hstack,asarray
import itertools

def as_tall(x):
    """ Turns a row vector into a column vector """
    return x.reshape(x.shape + (1,))

def basis_vector(n, i):
    """ Return an array like [0, 0, ..., 1, ..., 0, 0]

    >>> from multipolyfit.core import basis_vector
    >>> basis_vector(3, 1)
    array([0, 1, 0])
    >>> basis_vector(5, 4)
    array([0, 0, 0, 0, 1])
    """
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

y = [[1],[2],[3]]
xs = [[2,3,4],[4,5,6],[6,7,8]]
y = asarray(y).squeeze()
rows = y.shape[0]
xs = asarray(xs)
num_covariates = xs.shape[1]
xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))


generators = [basis_vector(num_covariates+1, i)
                        for i in range(num_covariates+1)]

# All combinations of degrees
deg = 2
powers = map(sum, itertools.combinations_with_replacement(generators, deg))

# Raise data to specified degree pattern, stack in order
A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))

beta = linalg.lstsq(A, y)[0]