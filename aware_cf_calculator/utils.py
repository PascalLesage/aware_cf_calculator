import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def get_stats(samples):
    return {
        "2.5%ile": np.percentile(samples, 2.5),
        "Q1": np.percentile(samples, 25),
        "Q3": np.percentile(samples, 75),
        "97.5%ile": np.percentile(samples, 97.5),
        "sd": np.std(samples),
        "mean": np.mean(samples),
        "median": np.median(samples)
    }


def OVL_two_random_arr(arr1, arr2, number_bins="Silverman's rule"):
    """Calculate the shared area of two distribution density curves as defined from sample arrays."""
    assert number_bins == "Silverman's rule" or isinstance(number_bins, int), "number_bins is not proprely defined"
    # Determine the range over which the integration will occur
    min_value = np.min((arr1.min(), arr2.min()))
    max_value = np.min((arr1.max(), arr2.max()))
    # Determine the bin width
    # If the default is used, Silverman's rule will be used for the longest array
    if number_bins == "Silverman's rule":
        if len(arr1) >= len(arr2):
            arr = arr1
        else:
            arr = arr2
        bin_width = 1.06 * len(arr) ** -0.2 * np.std(arr)
        number_bins = int((max_value - min_value) // bin_width)
    else:

        bin_width = (max_value - min_value) / number_bins
    # For each bin, find min frequency
    lower_bound = min_value  # Lower bound of the first bin is the min_value of both arrays
    min_arr = np.empty(number_bins)  # Array that will collect the min frequency in each bin
    for b in range(number_bins):
        higher_bound = lower_bound + bin_width  # Set the higher bound for the bin
        # Determine the share of samples in the interval
        freq_arr1 = np.ma.masked_where((arr1 < lower_bound) | (arr1 >= higher_bound), arr1).count() / len(arr1)
        freq_arr2 = np.ma.masked_where((arr2 < lower_bound) | (arr2 >= higher_bound), arr2).count() / len(arr2)
        # Conserve the lower frequency
        min_arr[b] = np.min((freq_arr1, freq_arr2))
        lower_bound = higher_bound  # To move to the next range
    return min_arr.sum()


def OVL_arr_fitted(arr, fitted, number_bins="Silverman's rule"):
    """Calculate the shared area of an array a PDF."""
    assert number_bins == "Silverman's rule" or isinstance(number_bins, int), "number_bins is not proprely defined"

    # Determine the range over which the integration will occur
    min_value = arr.min()
    max_value = arr.max()

    # Determine the bin width
    # If the default is used, Silverman's rule will be used for the longest array
    if number_bins == "Silverman's rule":
        bin_width = 1.06 * len(arr) ** -0.2 * np.std(arr)
        number_bins = int((max_value - min_value) // bin_width)
    else:
        bin_width = (max_value - min_value) / number_bins

    # For each bin, find min frequency
    lower_bound = min_value  # Lower bound of the first bin is the min_value of both arrays
    min_arr = np.empty(number_bins)  # Array that will collect the min frequency in each bin
    for b in range(number_bins):
        higher_bound = lower_bound + bin_width  # Set the higher bound for the bin
        # Determine the share of samples in the interval
        freq_arr = np.ma.masked_where((arr < lower_bound) | (arr >= higher_bound), arr).count() / len(arr)
        freq_dist = fitted.cdf(higher_bound) - fitted.cdf(lower_bound)
        # Conserve the lower frequency
        min_arr[b] = np.min((freq_arr, freq_dist))
        lower_bound = higher_bound  # To move to the next range
    return min_arr.sum()


def plot_arr_fitted(arr, fitted, number_bins="Silverman's rule", x_axis_label=None, title=None):
    assert number_bins == "Silverman's rule" or isinstance(number_bins, int), "number_bins is not proprely defined"

    # Determine the range over which the integration will occur
    min_value = np.min(arr)
    max_value = np.max(arr)

    # Determine the bin width
    # If the default is used, Silverman's rule will be used for the longest array
    if number_bins == "Silverman's rule":
        bin_width = 1.06 * len(arr) ** -0.2 * np.std(arr)
        number_bins = int((max_value - min_value) // bin_width)
    else:
        bin_width = (max_value - min_value) / number_bins

    fig, ax = plt.subplots(1, 1)
    interval = (np.min(arr) - np.max(arr)) / number_bins
    x = np.linspace(min_value + interval / 2, max_value - interval / 2, number_bins)
    ax.plot(x, fitted.pdf(x), 'r-', lw=2, alpha=0.6, label='fitted')
    ax.hist(arr, density=True, bins=number_bins)
    if x_axis_label:
        plt.xlabel(x_axis_label)
    if title:
        fig.suptitle(title)
    plt.ylabel("density")
    return fig

def trim_arr(arr, bottom_trim=0, top_trim=0):
    """Trim values from top and bottom, based on percentile"""
    assert 0<= top_trim <100, "Top trim should be between 0 and 100"
    assert 0 <= bottom_trim <= 100, "Bottom trim should be between 0 and 100"
    assert bottom_trim < 100-top_trim, "This combination of bottom and top trim would leave no values, if you didn't want to work, you could have stayed home"
    if len(set(arr))==1:
        return arr
    else:
        return arr[(arr>=np.percentile(arr, bottom_trim))&(arr<=np.percentile(arr, 100-top_trim))]


def no_uncertainty(arr):
    if len(set(arr))==1:
        return (arr[0], 1, arr[0])
    else:
        return "Not static", 0, "Not static"

def no_uncertainty_trunc_min_2_5(arr):
    arr = trim_arr(arr, 0, 2.5)
    if len(set(arr))==1:
        return (arr[0], 1, arr[0])
    else:
        return "Not static with min trimmed", 0, "Not static with min trimmed"

def no_uncertainty_trunc_max_2_5(arr):
    arr = trim_arr(arr, 0, 2.5)
    if len(set(arr))==1:
        return (set(arr)[0], 1, set(arr)[0])
    else:
        return "Not static  with max trimmed", 0, "Not static with max trimmed"

def no_uncertainty_trunc_min_max_5(arr):
    arr = trim_arr(arr, 2.5, 2.5)
    if len(set(arr))==1:
        return (arr[0], 1, arr[0])
    else:
        return "Not static with trimmed", 0, "Not static with trimmed"

def get_lognormal_fitted_all_free(arr):
    shape, loc, scale = sp.stats.lognorm.fit(arr)
    fitted = sp.stats.lognorm(s=shape, loc=loc, scale=scale)
    return (shape, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_lognormal_fitted_loc_is_0(arr):
    shape, loc, scale = sp.stats.lognorm.fit(arr, floc=0)
    fitted = sp.stats.lognorm(s=shape, loc=loc, scale=scale)
    return (shape, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_lognormal_fitted_loc_is_0_scale_is_median(arr):
    shape, loc, scale = sp.stats.lognorm.fit(arr, floc=0, fscale=np.median(arr))
    fitted = sp.stats.lognorm(s=shape, loc=loc, scale=scale)
    return (shape, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_triangular_fitted_all_free(arr):
    # The triangular distribution can be represented with an up-sloping
    # line from loc to (loc + c*scale) and then downsloping
    # for (loc + c*scale) to (loc + scale).
    # loc = min
    # scale = max - min
    # c = (mode - min)/(max - min)
    c, loc, scale = sp.stats.triang.fit(arr)
    fitted = sp.stats.triang(c=c, loc=loc, scale=scale)
    return (c, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_triangular_fitted_min_max(arr):
    # The triangular distribution can be represented with an up-sloping
    # line from loc to (loc + c*scale) and then downsloping
    # for (loc + c*scale) to (loc + scale).
    # loc = min
    # scale = max - min
    # c = (mode - min)/(max - min)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    c, loc, scale = sp.stats.triang.fit(arr, floc=arr_min, fscale=arr_max-arr_min)
    fitted = sp.stats.triang(c=c, loc=loc, scale=scale)
    return (c, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_triangular_fitted_min_max_trunc_5(arr):
    # The triangular distribution can be represented with an up-sloping
    # line from loc to (loc + c*scale) and then downsloping
    # for (loc + c*scale) to (loc + scale).
    # loc = min
    # scale = max - min
    # c = (mode - min)/(max - min)
    new_arr = trim_arr(arr, 2.5, 2.5)
    c, loc, scale = sp.stats.triang.fit(new_arr)
    fitted = sp.stats.triang(c=c, loc=loc, scale=scale)
    return (c, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_triangular_fitted_min_trunc_2_5(arr):
    # The triangular distribution can be represented with an up-sloping
    # line from loc to (loc + c*scale) and then downsloping
    # for (loc + c*scale) to (loc + scale).
    # loc = min
    # scale = max - min
    # c = (mode - min)/(max - min)
    new_arr = trim_arr(arr, 2.5, 0)
    c, loc, scale = sp.stats.triang.fit(new_arr)
    fitted = sp.stats.triang(c=c, loc=loc, scale=scale)
    return (c, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_triangular_fitted_max_trunc_2_5(arr):
    # The triangular distribution can be represented with an up-sloping
    # line from loc to (loc + c*scale) and then downsloping
    # for (loc + c*scale) to (loc + scale).
    # loc = min
    # scale = max - min
    # c = (mode - min)/(max - min)
    new_arr = trim_arr(arr, 0, 2.5)
    c, loc, scale = sp.stats.triang.fit(new_arr)
    fitted = sp.stats.triang(c=c, loc=loc, scale=scale)
    return (c, loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_uniform_fitted(arr):
    loc, scale = sp.stats.uniform.fit(arr)
    fitted = sp.stats.uniform(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_uniform_fitted_min_trunc_2_5(arr):
    new_arr = trim_arr(arr, 2.5, 0)
    loc, scale = sp.stats.uniform.fit(new_arr)
    fitted = sp.stats.uniform(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_uniform_fitted_max_trunc_2_5(arr):
    new_arr = trim_arr(arr, 0, 2.5)
    loc, scale = sp.stats.uniform.fit(new_arr)
    fitted = sp.stats.uniform(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_uniform_fitted_trunc_min_max_trunc_5(arr):
    new_arr = trim_arr(arr, 2.5, 2.5)
    loc, scale = sp.stats.uniform.fit(new_arr)
    fitted = sp.stats.uniform(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_normal_fitted(arr):
    loc, scale = sp.stats.norm.fit(arr)
    fitted = sp.stats.norm(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_normal_fitted_trunc_min_2_5(arr):
    new_arr = trim_arr(arr, 2.5, 0)
    loc, scale = sp.stats.norm.fit(new_arr)
    fitted = sp.stats.norm(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_normal_fitted_trunc_max_2_5(arr):
    new_arr = trim_arr(arr, 0, 2.5)
    loc, scale = sp.stats.norm.fit(new_arr)
    fitted = sp.stats.norm(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_normal_fitted_trunc_min_max_5(arr):
    new_arr = trim_arr(arr, 2.5, 2.5)
    loc, scale = sp.stats.norm.fit(new_arr)
    fitted = sp.stats.norm(loc=loc, scale=scale)
    return (loc, scale), OVL_arr_fitted(arr, fitted), fitted

def get_beta_all_free(arr):
    arr_mean = arr.mean()
    arr_min=arr.min()
    arr_max=arr.max()
    arr_var = np.var(arr)
    a=(arr_mean-arr_min)/(arr_max-arr_mean)*((arr_max-arr_mean)*(arr_mean-arr_min)/arr_var-1)
    b=(1-(arr_mean-arr_min)/(arr_max-arr_min))*((arr_max-arr_mean)*(arr_mean-arr_min)/arr_var-1)
    fitted = sp.stats.beta(a=a, b=b, loc=arr_min, scale=arr_max)
    return  (a, b, arr_min, arr_max), OVL_arr_fitted(arr, fitted), fitted

def get_beta_fixed_extremes(arr):
    arr_mean = arr.mean()
    arr_min=0
    arr_max=100
    arr_var = np.var(arr)
    a=(arr_mean-arr_min)/(arr_max-arr_mean)*((arr_max-arr_mean)*(arr_mean-arr_min)/arr_var-1)
    b=(1-(arr_mean-arr_min)/(arr_max-arr_min))*((arr_max-arr_mean)*(arr_mean-arr_min)/arr_var-1)
    fitted = sp.stats.beta(a=a, b=b, loc=arr_min, scale=arr_max)
    return  (a, b, arr_min, arr_max), OVL_arr_fitted(arr, fitted), fitted

def get_function_short_name(f):
    f_name = f.__str__()
    f_name = f_name[10:]
    f_name = f_name[0:f_name.find(' at')]
    return f_name


def try_all_stats_return_best(arr):
    all_fitting_strategies = [
        get_lognormal_fitted_all_free,
        get_lognormal_fitted_loc_is_0,
        get_lognormal_fitted_loc_is_0_scale_is_median,
        get_triangular_fitted_all_free,
        get_triangular_fitted_min_max,
        get_triangular_fitted_min_max_trunc_5,
        get_triangular_fitted_min_trunc_2_5,
        get_triangular_fitted_max_trunc_2_5,
        get_uniform_fitted,
        get_uniform_fitted_min_trunc_2_5,
        get_uniform_fitted_max_trunc_2_5,
        get_uniform_fitted_trunc_min_max_trunc_5,
        get_normal_fitted,
        get_normal_fitted_trunc_min_2_5,
        get_normal_fitted_trunc_max_2_5,
        get_normal_fitted_trunc_min_max_5,
        get_beta_all_free,
        get_beta_fixed_extremes,
        no_uncertainty,
        no_uncertainty_trunc_min_2_5,
        no_uncertainty_trunc_max_2_5,
        no_uncertainty_trunc_min_max_5,
    ]

    results = {}

    for strategy in all_fitting_strategies:
        try:
            params, ovl, fitted = strategy(arr)
            results[get_function_short_name(strategy)] = {
                'parameters': params,
                'ovl': ovl,
                'fitted': fitted
            }
        except Exception as err:
            results[get_function_short_name(strategy)] = {
                'parameters': "error",
                'ovl': 0,
                'fitted': str(err)
            }
    df = pd.DataFrame(results).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'fitting_function'}, inplace=True)
    if df[df['fitting_function'] == 'no_uncertainty'].loc['ovl'] == 1:
        return arr[0], 'no_uncertainty', 1, df[df['fitting_function'] == 'no_uncertainty'].iloc[0, :]
    elif df[df['fitting_function'] == 'no_uncertainty_trunc_min_2_5'].loc['ovl'] == 1:
        return arr[0], 'no_uncertainty_trunc_min_2_5', 1, df[df[
                                                                 'fitting_function'] == 'no_uncertainty_trunc_min_2_5'].iloc[
                                                          0, :]
    elif df[df['fitting_function'] == 'no_uncertainty_trunc_max_2_5'].loc['ovl'] == 1:
        return arr[0], 'no_uncertainty_trunc_max_2_5', 1, df[df[
                                                                 'fitting_function'] == 'no_uncertainty_trunc_max_2_5'].iloc[
                                                          0, :]
    elif df[df['fitting_function'] == 'no_uncertainty_trunc_min_max_5'].loc['ovl'] == 1:
        return arr[0], 'no_uncertainty_trunc_min_max_5', 1, df[df[
                                                                   'fitting_function'] == 'no_uncertainty_trunc_min_max_5'].iloc[
                                                            0, :]
    else:
        best_fit = df[df['ovl'] == df['ovl'].max()].iloc[0, 1]
        OVL = df[df['ovl'] == df['ovl'].max()].iloc[0, 2],
        best_fit_name = df[df['ovl'] == df['ovl'].max()].iloc[0, 0]
        return best_fit, best_fit_name, OVL, df[df['ovl'] == df['ovl'].max()].iloc[0, :]


def try_all_stats_return_all_OVL(arr):
    all_fitting_strategies = [
        get_lognormal_fitted_all_free,
        get_lognormal_fitted_loc_is_0,
        get_lognormal_fitted_loc_is_0_scale_is_median,
        get_triangular_fitted_all_free,
        get_triangular_fitted_min_max,
        get_triangular_fitted_min_max_trunc_5,
        get_triangular_fitted_min_trunc_2_5,
        get_triangular_fitted_max_trunc_2_5,
        get_uniform_fitted,
        get_uniform_fitted_min_trunc_2_5,
        get_uniform_fitted_max_trunc_2_5,
        get_uniform_fitted_trunc_min_max_trunc_5,
        get_normal_fitted,
        get_normal_fitted_trunc_min_2_5,
        get_normal_fitted_trunc_max_2_5,
        get_normal_fitted_trunc_min_max_5,
        get_beta_all_free,
        get_beta_fixed_extremes,
        no_uncertainty,
        no_uncertainty_trunc_min_2_5,
        no_uncertainty_trunc_max_2_5,
        no_uncertainty_trunc_min_max_5]

    results = {}

    for strategy in all_fitting_strategies:
        try:
            params, ovl, fitted = strategy(arr)
            results[get_function_short_name(strategy)] = {
                'parameters': params,
                'ovl': ovl,
                'fitted': fitted
            }
        except Exception as err:
            results[get_function_short_name(strategy)] = {
                'parameters': "error",
                'ovl': 0,
                'fitted': str(err)
            }
    df = pd.DataFrame(results).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'fitting_function'}, inplace=True)
    return df