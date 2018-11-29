def dp_mean(data_vec, epsilon=1.0, delta=0, seed=0, axis=None):
    """
    Computes a differentially-private arithmetic mean along the specified axis.

    Returns the private average of the array elements. The average is taken over
    the flattened array by default, otherwise over the specified axis.
    Uses the Laplace Mechanism by default. If delta is specified, uses the Gaussian
    mechanism.

    Parameters
    ----------
    data_vec : ndarray
        A numeric tensor such that each element is bounded in the range [0, 1]
    epsilon: double, optional
        The privacy budget (default=1.0)
    delta: double, optional
        The privacy failure probability (default=0.001)
    seed: int, optional
        A seed for the random generator
    axis : int, optional
        Axis along which the means are computed. The default is to compute the mean
        of the flattened array.

    Returns
    -------
    mean: ndarray
        Returns a new array containing the mean values

    Examples
    --------
        >>> import numpy as np
        >>> x = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        >>> dp_mean(x)
        array([ 0.46284092])
        >>> dp_mean(x, axis=0)
        array([ 0.31284092,  0.47034795,  0.5287595 ,  0.61175675])
        >>> dp_mean(x, axis=1)
        array([ 0.26284092,  0.72034795])

    """

    import numpy as np

    assert (0.0 <= data_vec).all() and (data_vec <= 1.0).all(), \
        'ERROR: Input vector should have bounded entries in [0,1].'
    assert epsilon > 0.0, 'ERROR: Epsilon should be positive.'
    assert 0.0 <= delta <= 1.0, 'ERROR: Delta should be bounded in [0,1].'

    rnd = np.random.RandomState(seed)
    n = data_vec.size
    f = np.mean(data_vec, axis=axis)
    if delta == 0:
        # Laplace Mechanism
        noise = rnd.laplace(loc=0, scale=1/float(n*epsilon), size=f.size)
    else:
        # Gaussian Mechanism
        sigma = (1.0/(n*epsilon))*np.sqrt(2*np.log(1.25/delta))
        noise = rnd.normal(0.0, sigma, size=f.size)
    f += noise

    return f
        
def dp_var(data_vec, epsilon=1.0, delta=0, seed=0, axis=None):
    """
    Computes a differentially-private variance along the specified axis.

    Returns the private variance of the array elements. The variance is taken over
    the flattened array by default, otherwise over the specified axis.
    Uses the Laplace Mechanism by default. If delta is specified, uses the Gaussian
    mechanism.

    Parameters
    ----------
    data_vec : ndarray
        A numeric tensor such that each element is bounded in the range [0, 1]
    epsilon: double, optional
        The privacy budget (default=1.0)
    delta: double, optional
        The privacy failure probability (default=0.001)
    seed: int, optional
        A seed for the random generator
    axis : int, optional
        Axis along which the means are computed. The default is to compute the mean
        of the flattened array.

    Returns
    -------
    mean: ndarray
        Returns a new array containing the mean values

    Examples
    --------
        >>> import numpy as np
        >>> rnd = np.random.RandomState(1234)
        >>> x = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        >>> dp_var(x)
        array([ 0.0862074])
        >>> dp_mean(x, axis=0)
        array([ 0.31284092,  0.47034795,  0.5287595 ,  0.61175675])
        >>> dp_mean(x, axis=1)
        array([ 0.26284092,  0.72034795])

    """
    import numpy as np

    assert (0.0 <= data_vec).all() and (data_vec <= 1.0).all(), \
        'ERROR: Input vector should have bounded entries in [0,1].'
    assert epsilon > 0.0, 'ERROR: Epsilon should be positive.'
    assert 0.0 <= delta <= 1.0, 'ERROR: Delta should be bounded in [0,1].'

    rnd = np.random.RandomState(seed)
    var = np.var(data_vec, axis=axis)
    n = (data_vec.size / var.size)
    delf = 3.0 * (1.0-1.0/n)/n

    if delta == 0:
        noise = rnd.laplace(loc=0, scale=delf/epsilon, size=var.size)
    else:
        sigma = (3.0/(n*epsilon))*(1-1.0/n)*np.sqrt(2*np.log(1.25/delta))
        noise = rnd.normal(0.0, sigma, size=var.size)

    var += noise
    return var
        
def dp_hist ( data, num_bins=10, epsilon=1.0, delta=0.1, histtype = 'continuous' ):
    """
    This function provides a differentially-private estimate of the histogram of a vector.

    Input:

      data = data vector
      num_bins = number of bins for the histogram, default is 10
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1
      histtype = a string indicating which type of histogram is desired ('continuous', or 'discrete'),
                 by default, histtype = 'continuous'

    Note that for discrete histogram, the user input "num_bins" is ignored.

    Output:

      dp_hist_counts = number of items in each bins
      bin_edges = location of bin edges

    Example:

      # >>> import numpy as np
      # >>> import dp_stats as dps
      # >>> x = np.random.rand(10)
      # >>> x_mu = dps.dp_hist( x, 5, 1.0, 0.1, 'continuous' )
      (array([ 0.81163273, -1.18836727, -1.18836727,  0.81163273, -0.18836727]), array([ 0.1832111 ,  0.33342489,  0.48363868,  0.63385247,  0.78406625,
    0.93428004]))

    """

    import numpy as np

    if len(data) != np.size(data):
        print('ERROR: Input should be a vector.')
        return
    elif type(num_bins) != int:
        print('ERROR: Number of bins should be an integer')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        if histtype == 'discrete':
            num_bins = len( np.unique(data) )
        hist_counts = [0] * num_bins
        data_min = min(data)
        data_max = max(data)
        bin_edges = np.linspace(data_min, data_max, num_bins+1)
        interval = (data_max - data_min) + 0.000000000001
        
        for kk in data:
            loc = (kk - data_min) / interval
            index = int(loc * num_bins)
            hist_counts[index] += 1.0

        if delta==0:
            noise = np.random.laplace(loc = 0, scale = 1.0/epsilon, size = (1,len(hist_counts)))
        else:
            sigma = (1.0/epsilon)*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, len(hist_counts))

        hist_array=np.asarray(hist_counts)
        noise_array=np.asarray(noise)
        dp_hist_counts = hist_array+noise_array

        return ( dp_hist_counts, bin_edges )


## todo: cumulative

## todo: groupby
