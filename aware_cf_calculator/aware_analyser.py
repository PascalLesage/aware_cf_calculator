from .aware_stochastic import AwareStochastic
from .aware_static import AwareStatic
from .utils import get_stats, try_all_stats_return_all_OVL
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import os
import copy
import pyprind
from scipy.stats import gaussian_kde, rankdata, norm, spearmanr
import datetime


class AwareAnalyser(AwareStochastic):
    """ Class used for analysis of Aware samples

        Supported result types (passed as string):
            average_unknown_cf
            average_agri_cf
            average_non_agri_cf
            monthly_cf_all
            AMD_world_over_AMD_i

    """

    def __init__(self, base_dir_path, sim_name="all_random"):
        """ Class to assemble methods to analyze AwareStochastic results"""

        sim_dir = Path(base_dir_path) / "stochastic_results" / sim_name
        assert sim_dir.is_dir(), "No samples found for simulation {} at path {}".format(
                sim_name, base_dir_path
            )

        log_fp = sim_dir / "log_file.pickle"
        assert log_fp.is_file(), "No log file found for simulation {}".format(sim_name)

        with open(log_fp, 'rb') as f:
            log = pickle.load(f)

        AwareStochastic.__init__(
            self, base_dir_path=base_dir_path, sim_name=sim_name,
            consider_certain=log['consider_certain'],
            iterations=log['iterations']
        )

        self.valid_result_types = [
            'average_unknown_cf',
            'average_agri_cf',
            'average_non_agri_cf',
            'monthly_cf_all',
            'AMD_world_over_AMD_i'
        ]
        self.result_types_with_months = [
            'monthly_cf_all',
            'AMD_world_over_AMD_i'
        ]
        self.result_type_name_dict = {
            'average_unknown_cf': 'Average CF, unspecified',
            'average_agri_cf': 'Average CF, agricultural use',
            'average_non_agri_cf': 'Average CF, non-agricultural use',
            'monthly_cf_all': 'CF for month',
            'AMD_world_over_AMD_i': 'AMD world over AMD'
        }

    def check_result_type(self, result_type):
        """ Check that requested result type is part of valid result types"""
        if result_type not in self.valid_result_types:
            raise ValueError(
                "Result type {} not understood, accepted result types are: \
                {}".format(result_type, self.valid_result_types))

    def check_basins(self, basins):
        """ Check and, if necessary, modify basin list"""

        if basins == "all":
            basins = self.basins
        else:
            if not isinstance(basins, list):
                raise ValueError("Argument `basins` should be a list of basins")
            invalid_basins = [b for b in basins if b not in self.basins]
            if invalid_basins:
                raise ValueError("Not all basins were recognized: {}".format(invalid_basins))
        return basins

    def check_months(self, months):
        """ Check and, if necessary, modify month list"""
        if months == 'all':
            months = self.months
        else:
            assert isinstance(months, list), 'Months should be a list of months, is {}'.format(months)
            invalid_months = [month for month in months if month not in self.months]
            if invalid_months:
                raise ValueError('Invalid month list. Got {}, valid names are {}'.format(invalid_months, self.months))
        return months

    def plot_all_hists_given_basin_and_month(self, basin, month, lower_bound, upper_bound,
                                             return_graph=False, save_graph=True):
        """ Graph distribution of parameters and of results for given basin/month"""


        # Set up figure
        fig, axarr = plt.subplots(7, 3)
        fig.subplots_adjust(hspace=1, wspace=0.4)
        fig.set_size_inches(8, 14.5)

        # Get and plot given variable samples
        input_arrs = {}
        #variables = copy.copy(self.given_variable_names)
        variables = [
                "avail_delta_wo_model_uncertainty",
                "avail_net_wo_model_uncertainty",
                "model_uncertainty",
                "avail_delta",
                "avail_net",
                "domestic",
                "electricity",
                "irrigation",
                "livestock",
                "manufacturing",
                "pastor",
        ]

        for i, v in enumerate(variables):
            ax = plt.subplot(7, 3, i + 1)
            arr = self.get_arr_sampled_data(v, basin, month)
            if arr is not None:
                input_arrs[v] = arr
                ax.hist(arr, density=False, histtype="step", bins=100)
                ax.yaxis.set_visible(False)
                if v != 'pastor':
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            else:
                input_arrs[v] = np.zeros(shape=self.iterations) # This should probably be det values, not 0s
                ax.yaxis.set_visible(False)
            if v == 'avail_delta_wo_model_uncertainty':
                ax.set_title("avail_delta\nwo_model_uncertainty")
            elif v == 'avail_net_wo_model_uncertainty':
                ax.set_title("avail_net\nwo_model_uncertainty")
            else:
                ax.set_title(v)
            if v in ['irrigation', 'avail_delta', 'avail_net', 'avail_delta_wo_model_uncertainty', 'avail_net_wo_model_uncertainty']:
                ax.set_xlabel('[m3/month]')
            elif v in ['electricity', 'domestic', 'livestock', 'manufacturing']:
                ax.set_xlabel('[m3/year]')
            elif v in ['pastor', 'model_uncertainty']:
                ax.set_xlabel('[unitless]')
            else:
                print("Couldn't find unit for ", v)

        # Calculate intermediate variables
        HWC = input_arrs['irrigation'] + input_arrs['domestic'] / 12 \
              + input_arrs['electricity'] / 12 + input_arrs['livestock'] / 12 \
              + input_arrs['manufacturing'] / 12
        avail_before_human_consumption = HWC + input_arrs['avail_net']
        pristine = avail_before_human_consumption + input_arrs['avail_delta']
        EWR = pristine * input_arrs['pastor']
        AMD = avail_before_human_consumption - EWR - HWC
        AMD_per_m2 = AMD/self.det_data['area'].loc[basin, 'area']

        # Plot intermediate variables
        # HWC
        ax = plt.subplot(7, 3, 12)
        ax.hist(HWC, density=False, histtype="step", bins=50)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('[m3/month]')
        ax.yaxis.set_visible(False)
        ax.set_title("HWC")

        # Avail before human consumption
        ax = plt.subplot(7, 3, 13)
        ax.hist(avail_before_human_consumption, density=False, histtype="step", bins=50)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('[m3/month]')
        ax.yaxis.set_visible(False)
        ax.set_title("Availability before HC")

        # Pristine
        ax = plt.subplot(7, 3, 14)
        ax.hist(pristine, density=False, histtype="step", bins=50)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.set_visible(False)
        ax.set_xlabel('[m3/month]')
        ax.set_title("Pristine")

        # EWR
        ax = plt.subplot(7, 3, 15)
        ax.hist(EWR, density=False, histtype="step", bins=50)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.set_visible(False)
        ax.set_xlabel('[m3/month]')
        ax.set_title("EWR")

        # AMD
        ax = plt.subplot(7, 3, 16)
        ax.hist(AMD_per_m2, density=False, histtype="step", bins=50)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.set_visible(False)
        ax.set_xlabel('[(m3/month)/m2]')
        ax.set_title("AMD per m2")

        # AMD_world_over_AMD_i
        ax = plt.subplot(7, 3, 17)
        ax.hist(
            np.load(
                self.aggregated_samples / "AMD_world_over_AMD_i" / "AMD_world_over_AMD_i_{}_{}.npy".format(
                    basin, month)),
            density=False, histtype="step", bins=50
        )
        ax.set_title("AMD_world_over_AMD_i")
        ax.yaxis.set_visible(False)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('[unitless]')

        # CF for month
        ax = plt.subplot(7, 3, 18)
        ax.hist(
            np.load(
                self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_")/ "monthly" / "cf_{}_{}.npy".format(
                    basin, month)),
            density=False, histtype="step", bins=50
        )
        ax.yaxis.set_visible(False)
        ax.set_title("CF for month")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('[unitless]')

        # CF unknown
        ax = plt.subplot(7, 3, 19)
        ax.hist(
            np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_unknown" / "cf_average_unknown_{}.npy".format(
                    basin)),
            density=False, histtype="step", bins=50
        )
        ax.set_title("CF average unknown")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.set_visible(False)
        ax.set_xlabel('[unitless]')

        # CF agri
        ax = plt.subplot(7, 3, 20)
        ax.hist(
            np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_agri" / "cf_average_agri_{}.npy".format(
                        basin)
            ),
            density=False, histtype="step", bins=50
        )
        ax.set_title("CF average agri")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.set_visible(False)
        ax.set_xlabel('[unitless]')

        # CF non_agri
        ax = plt.subplot(7, 3, 21)
        ax.hist(
            np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_non_agri" / "cf_average_non_agri_{}.npy".format(
                    basin)
            ),
            density=False, histtype="step", bins=50
        )
        ax.set_title("CF average non agri")
        ax.yaxis.set_visible(False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xlabel('[unitless]')

        fig.suptitle('Basin {}, {}'.format(basin, month), y=1.01, fontsize=12, weight='bold')
        save_dir = self.graphs_dir / "all_hists_given_basin_and_month"
        save_dir.mkdir(exist_ok=True)
        if save_graph:
            fig.savefig(save_dir / "all_hists_{}_{}.jpeg".format(basin, month), bbox_inches="tight")
        if return_graph:
            return fig


    def get_all_arrays(self, basin, month, lower_bound, upper_bound,
                                             return_df=True, save_xls=False):
        """Get arrays for all values used in calcs for given basin and month"""

        # Get and plot given variable samples
        arrs_dict = {}
        variables = copy.copy(self.given_variable_names)
        for i, v in enumerate(variables):
            if v!='area':
                arr = self.get_arr_sampled_data(v, basin, month)
            else:
                with open(self.filtered_pickles / 'area.pickle', 'rb') as f:
                    area_series = pickle.load(f)
                area = area_series.loc[basin, 'area']
                arr = np.ones(shape=self.iterations) * area
            if arr is not None:
                arrs_dict[v] = arr
            else:
                arrs_dict[v] = np.zeros(shape=self.iterations)

        # Calculate intermediate variables
        arrs_dict['HWC'] = arrs_dict['irrigation'] + arrs_dict['domestic'] / 12 \
              + arrs_dict['electricity'] / 12 + arrs_dict['livestock'] / 12 \
              + arrs_dict['manufacturing'] / 12
        arrs_dict['avail_before_human_consumption'] = arrs_dict['HWC'] + arrs_dict['avail_net']
        arrs_dict['pristine'] = arrs_dict['avail_before_human_consumption'] + arrs_dict['avail_delta']
        arrs_dict['EWR'] = arrs_dict['pristine'] * arrs_dict['pastor']
        arrs_dict['AMD'] = arrs_dict['avail_before_human_consumption'] - arrs_dict['EWR'] - arrs_dict['HWC']
        arrs_dict['AMD_per_m2'] = arrs_dict['AMD']/area
        arrs_dict['AMD_world'] = np.load(self.aggregated_samples/'AMD_world'/'AMD_world.npy')
        arrs_dict['AMD_world_over_AMD_i'] = np.load(
                self.aggregated_samples / "AMD_world_over_AMD_i" / "AMD_world_over_AMD_i_{}_{}.npy".format(
                    basin, month))
        arrs_dict['CF for month'] = np.load(
                self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_")/ "monthly" / "cf_{}_{}.npy".format(
                    basin, month))

        arrs_dict['CF_unspecified'] = np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_unknown" / "cf_average_unknown_{}.npy".format(
                    basin))

        arrs_dict['CF agri'] = np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_agri" / "cf_average_agri_{}.npy".format(
                        basin)
            )

        arrs_dict['CF agri'] = np.load(
                self.cf_dir / "{}_{}".format(
                    lower_bound, upper_bound).replace(".", "_") / "average_non_agri" / "cf_average_non_agri_{}.npy".format(
                    basin)
            )
        df = pd.DataFrame.from_dict(arrs_dict)
        df['area'] = area
        save_dir = self.sim_dir/"all_samples_examples"
        save_dir.mkdir(exist_ok=True)
        if save_xls:
            df.to_excel(save_dir/"all_arrs_{}_{}.xlsx".format(basin, month))
        if return_df:
            return df
        else:
            return None

    def get_stochastic_array(self, result_type, basin, month=None,
                             lower_bound=None, upper_bound=None,
                             iterations_limiter=None, dtype=np.float32):
        """ Return array of precalculated results for given basin and month"""

        if result_type not in self.valid_result_types:
            raise ValueError("Result type {} not valid, valid types are {}".format(
                result_type, self.valid_result_types
            ))

        if iterations_limiter:
            if iterations_limiter >= self.iterations:
                print("iterations_limiter >= number of iterations available, using all samples")
                iterations = self.iterations
            else:
                iterations = iterations_limiter
        else:
            iterations = self.iterations

        if result_type == 'average_unknown_cf':
            assert all([lower_bound, upper_bound]), "Must specify lower and upper bound"
            result_dir = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_unknown"
            file_base = "cf_average_unknown_{}.npy"

        elif result_type == 'average_agri_cf':
            assert all([lower_bound, upper_bound]), "Must specify lower and upper bound"
            result_dir = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_agri"
            file_base = "cf_average_agri_{}.npy"

        elif result_type == 'average_non_agri_cf':
            assert all([lower_bound, upper_bound]), "Must specify lower and upper bound"
            result_dir = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_non_agri"
            file_base = "cf_average_non_agri_{}.npy"

        elif result_type == 'monthly_cf_all':
            assert all([lower_bound, upper_bound]), "Must specify lower and upper bound"
            assert month in self.months, \
                "Month must be specified and must be in {}, was passed as {}".format(self.months, month)
            result_dir = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "monthly"
            file_base = "cf_{}_{}.npy"

        elif result_type == 'AMD_world_over_AMD_i':
            assert month in self.months, \
                "Month must be specified and must be in {}, was passed as {}".format(self.months, month)
            result_dir = self.aggregated_samples / "AMD_world_over_AMD_i"
            file_base = "AMD_world_over_AMD_i_{}_{}.npy"

        else:
            raise NotImplemented("Result type {} not understood, accepted result types are: "
                                 "average_unknown_cf, average_agri_cf, average_non_agri, monthly_cf_all and AMD_world_over_AMD_i")

        assert result_dir.is_dir(), \
            "No results found at {}, generate results first".format(
                result_dir
            )

        if result_type in self.result_types_with_months:
            file = result_dir / file_base.format(basin, month)
        else:
            file = result_dir / file_base.format(basin)

        return np.load(file)[0:iterations].astype(dtype)

    def build_stochastic_result_df(self, result_type, basins="all", months="all",
                                   lower_bound=0.1, upper_bound=100,
                                   iterations_limiter=None, dtype=np.float32):
        """ Concatenate all samples for a given type of results in a tidy DataFrame

        Supported result types (passed as string):
            average_unknown_cf average cfs, unknown --> bounds necessary
            average_agri_cf average cfs, agri --> bounds necessary
            average_non_agri_cf average cfs, agri --> bounds necessary
            monthly_cf_all  --> bounds necessary
            AMD_world_over_AMD_i --> bounds not necessary

        """

        self.check_result_type(result_type)
        basins = self.check_basins(basins)

        if result_type in self.result_types_with_months:
            months = self.check_months(months)
            print("Looking into months:", months)
            samples_df_list = []

            for i, (basin, month) in pyprind.prog_bar(list(enumerate(
                    itertools.product(
                        basins,
                        months
                    ))
                )):
                arr = self.get_stochastic_array(
                    result_type=result_type, basin=basin, month=month,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    iterations_limiter=iterations_limiter, dtype=dtype
                )

                df = pd.DataFrame()
                df['sample'] = arr
                df['iteration'] = np.arange(arr.size)
                df['BAS34S_ID'] = basin
                df['month'] = month
                samples_df_list.append(df)

        else:
            samples_df_list = []

            for basin in pyprind.prog_bar(basins):
                arr = self.get_stochastic_array(
                    result_type=result_type, basin=basin,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    iterations_limiter=iterations_limiter
                )
                df = pd.DataFrame()
                df['sample'] = arr
                df['iteration'] = np.arange(arr.size)
                df['BAS34S_ID'] = basin
                samples_df_list.append(df)
        result = pd.concat(samples_df_list)
        result.reset_index(inplace=True, drop=True)
        return result

    def pair_violins_indicator_distribution(self, result_type, basins='all', months='all',
                                            lower_bound=0.1, upper_bound=100, dtype=np.float32,
                                            include_median=False, save_fig=True, return_fig=False):
        """ Plot distribution of static and stochastic results side by side"""
        self.check_result_type(result_type)
        basins = self.check_basins(basins)

        if result_type in self.result_types_with_months:
            months = self.check_months(months)

        if result_type == 'average_unknown_cf':
            title = "Average CF (unknown use)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_unknown.pickle").loc[basins]

            stochastic_values = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None, dtype=dtype)

        elif result_type == 'average_agri_cf':
            title = "Average CF (agri)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_agri.pickle").loc[basins]
            stochastic_values = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None, dtype=dtype)

        elif result_type == 'average_non_agri_cf':
            title = "Average CF (non agri)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_non_agri.pickle").loc[basins]
            stochastic_values = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None, dtype=dtype)

        elif result_type == 'monthly_cf_all':
            title = "All monthly CFs"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_monthly.pickle").loc[basins, months]
            stochastic_values = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None, dtype=dtype)

        elif result_type == 'AMD_world_over_AMD_i':
            title = "All monthly AMD world / AMD i"
            y_label = "AMD world / AMD i (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs"
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "AMD_world_over_AMD_i.pickle").loc[basins, months]
            stochastic_values = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None, dtype=dtype)

        else:
            raise NotImplemented("Result type {} not understood, accepted result types are: "
                                 "average_unknown_cf, average_agri_cf, average_non_agri, monthly_cf_all and AMD_world_over_AMD_i")



        sns.set_style("whitegrid")
        fig, (ax_static, ax_stochastic) = plt.subplots(1, 2, sharey=True)
        fig.suptitle(title)
        sns.violinplot(ax=ax_static, y=static_values, cut=0, inner=None, orient='v', color="lightgrey")
        if include_median:
            median_static = static_values.median()
            yposlist_static = median_static
            xposlist = 0
            ax_static.text(xposlist, yposlist_static, "Median={:.3}".format(median_static))
        ax_static.set_title("Static ({} values)".format(static_values.values.flatten().size))
        sns.violinplot(ax=ax_stochastic, y=stochastic_values['sample'].values, inner=None, cut=0, orient="v", color="lightgrey")
        if include_median:
            median_stochastic = np.median(stochastic_values['sample'].values)
            yposlist_stochastic = median_stochastic
            xposlist = 0
            ax_stochastic.text(xposlist, yposlist_stochastic, "Median={:.3}".format(median_stochastic))
        ax_static.set_ylabel(y_label)
        ax_stochastic.set_title("Stochastic ({} values)".format(stochastic_values.shape[0]))
        ax_stochastic.set_ylabel("")
        save_dir = self.graphs_dir / "violin_indicator_dispersion_with_median"
        save_dir.mkdir(exist_ok=True)
        if save_fig:
            fig.savefig(save_dir / "violin_indicator_dispersion_{}.jpeg".format(result_type), bbox_inches="tight")
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None

    def pair_boxplots_indicator_distribution(self, result_type, lower_bound=None, upper_bound=None):
        """ Plot distribution of static and stochastic results side by side

            Supported result types (passed as string):
                average_unknown_cf: average cfs, unknown --> bounds necessary
                average_agri_cf: average cfs, agri --> bounds necessary
                average_non_agri_cf: average cfs, agri --> bounds necessary
                monthly_cf_all:  --> bounds necessary
                AMD_world_over_AMD_i:
        """
        if result_type == 'average_unknown_cf':
            title = "Average CF (unspecified)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_unknown.pickle")
            stochastic_values = self.build_stochastic_result_df(result_type, lower_bound=lower_bound, upper_bound=upper_bound, iterations_limiter=None)

        elif result_type == 'average_agri_cf':
            title = "Average CF (agri)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_agri.pickle")
            stochastic_values = self.build_stochastic_result_df(result_type, lower_bound=lower_bound, upper_bound=upper_bound, iterations_limiter=None)
        elif result_type == 'average_non_agri_cf':
            title = "Average CF (non agri)"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_average_non_agri.pickle")
            stochastic_values = self.build_stochastic_result_df(result_type, lower_bound=lower_bound, upper_bound=upper_bound, iterations_limiter=None)
        elif result_type == 'monthly_cf_all':
            title = "All monthly CFs"
            y_label = "CF (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "cfs_monthly.pickle")
            stochastic_values = self.build_stochastic_result_df(result_type, lower_bound=lower_bound, upper_bound=upper_bound, iterations_limiter=None)
        elif result_type == 'AMD_world_over_AMD_i':
            title = "All monthly AMD world / AMD i"
            y_label = "AMD world / AMD i (unitless)"
            static_df_dirpath = self.det_results_dir / "dfs"
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_values = pd.read_pickle(static_df_dirpath / "AMD_world_over_AMD_i.pickle")
            stochastic_values = self.build_stochastic_result_df(result_type, lower_bound=lower_bound, upper_bound=upper_bound, iterations_limiter=None)
        else:
            raise NotImplemented("Result type {} not understood, accepted result types are: "
                                 "average_unknown_cf, average_agri_cf, average_non_agri_cf, monthly_cf_all and AMD_world_over_AMD_i")

        sns.set_style("whitegrid")
        fig, (ax_static, ax_stochastic) = plt.subplots(1, 2, sharey=True)
        fig.suptitle(title)
        sns.boxplot(ax=ax_static, y=static_values.values.flatten(), fliersize=1)
        ax_static.set_title("Static ({} values)".format(static_values.values.flatten().size))
        sns.boxplot(ax=ax_stochastic, y=stochastic_values['sample'].values, fliersize=1)
        ax_static.set_ylabel(y_label)
        ax_stochastic.set_title("Stochastic ({} values)".format(stochastic_values.shape[0]))
        ax_stochastic.set_ylabel("")

    def get_full_stats(self, result_type, basins="all", months='all',
                       lower_bound=0.1, upper_bound=100,
                       aggregate_by_static=False, augment_with_dispersion=True,
                       iterations_limiter=None, dtype=np.float32):
        """ get summary df for plotting results

            if aggregate_by_static, statistics will be calculated for all samples
            whose static counterpart are equal (used for e.g. "plot_all_variability")
        """
        self.check_result_type(result_type)
        basins = self.check_basins(basins)
        if result_type in self.result_types_with_months:
            months = self.check_months(months)

        if result_type == 'average_unknown_cf':
            assert all([lower_bound, upper_bound]), "lower bound and upper bound need to be specified"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_df = pd.read_pickle(static_df_dirpath / "cfs_average_unknown.pickle").loc[basins]
            stochastic_result_df = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                iterations_limiter=iterations_limiter,
                dtype=dtype
            )

        elif result_type == 'average_agri_cf':
            assert all([lower_bound, upper_bound]), "lower bound and upper bound need to be specified"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_df = pd.read_pickle(static_df_dirpath / "cfs_average_agri.pickle").loc[basins]
            stochastic_result_df = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=iterations_limiter,
                dtype=dtype
            )

        elif result_type == 'average_non_agri_cf':
            assert all([lower_bound, upper_bound]), "lower bound and upper bound need to be specified"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_df = pd.read_pickle(static_df_dirpath / "cfs_average_non_agri.pickle")
            stochastic_result_df = self.build_stochastic_result_df(
                result_type,
                basins=basins,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=iterations_limiter,
                dtype=dtype
            )

        elif result_type == 'monthly_cf_all':
            assert all([lower_bound, upper_bound]), "lower bound and upper bound need to be specified"
            static_df_dirpath = self.det_results_dir / "dfs" / "cfs_{}_{}".format(lower_bound, upper_bound).replace(".", "_")
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_df = pd.read_pickle(static_df_dirpath / "cfs_monthly.pickle").loc[basins, months]
            stochastic_result_df = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None,
                dtype=dtype
            )

        elif result_type == 'AMD_world_over_AMD_i':
            static_df_dirpath = self.det_results_dir / "dfs"
            assert static_df_dirpath.is_dir(), "No static results found at {}, generate results for these bounds first".format(
                static_df_dirpath)
            static_df = pd.read_pickle(static_df_dirpath / "AMD_world_over_AMD_i.pickle").loc[basins, months]
            stochastic_result_df = self.build_stochastic_result_df(
                result_type=result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                iterations_limiter=None,
                dtype=dtype
            )

        else:
            raise NotImplemented("Result type {} not understood, accepted result types are: "
                                 "average_unknown_cf, average_agri_cf, average_non_agri, monthly_cf_all and AMD_world_over_AMD_i")

        sers = []
        static_df = static_df.astype("float32")
        if not result_type in self.result_types_with_months:
            for basin in pyprind.prog_bar(basins):
                samples = stochastic_result_df[
                    stochastic_result_df['BAS34S_ID'] == basin]['sample'].values
                stats = pd.Series(get_stats(samples))
                stats["BAS34S_ID"] = int(basin)
                stats["static"] = static_df.loc[basin]
                sers.append(pd.Series(stats))
        else:
            for basin in pyprind.prog_bar(basins):
                for month in months:
                    samples = stochastic_result_df[
                        (stochastic_result_df['BAS34S_ID'] == basin)
                        &
                        (stochastic_result_df['month'] == month)
                        ]['sample'].values

                    stats = pd.Series(get_stats(samples))
                    stats["BAS34S_ID"] = int(basin)
                    stats["month"] = month
                    stats["static"] = static_df.loc[basin, month]
                    sers.append(stats)
        summary_df = pd.DataFrame(sers)
        summary_df.reset_index(inplace=True, drop=True)

        if aggregate_by_static:
            summary_df = self.agg_stats_by_static(result_type, summary_df,  stochastic_result_df)
        if augment_with_dispersion:
            summary_df = self.augment_stats_df_with_dispersion(summary_df)
        return summary_df

    def agg_stats_by_static(self, result_type, stat_df, stochastic_result_df):
        """ Get summary stats, aggregating samples for identical det values"""

        dfs_to_aggregate = []
        for static_value, df in stat_df.groupby("static"):
            if df.shape[0] == 1:
                dfs_to_aggregate.append(df)
            else:
                if result_type in self.result_types_with_months:
                    arrs = []
                    for basin, df_by_basin in df.groupby('BAS34S_ID'):
                        for month in df_by_basin['month'].unique():
                            arrs.append(stochastic_result_df[
                                            (stochastic_result_df['BAS34S_ID'] == basin)
                                            &
                                            (stochastic_result_df['month'] == month)
                                            ]['sample'].values)
                    selected_samples = np.concatenate(arrs)
                    stats = get_stats(selected_samples)
                    stats['BAS34S_ID'] = "multiple"
                    stats['month'] = "multiple"
                    stats['static'] = static_value
                    dfs_to_aggregate.append(pd.DataFrame.from_dict(stats, orient='index').T)
                else:
                    arrs = []
                    for basin, df_by_basin in df.groupby('BAS34S_ID'):
                        arrs.append(stochastic_result_df[
                                        (stochastic_result_df['BAS34S_ID'] == basin)
                                    ]['sample'].values)
                    selected_samples = np.concatenate(arrs)
                    stats = get_stats(selected_samples)
                    stats['BAS34S_ID'] = "multiple"
                    stats['static'] = static_value
                    dfs_to_aggregate.append(pd.DataFrame.from_dict(stats, orient='index').T)

        if len(dfs_to_aggregate) != stat_df.shape[0]:
            stat_df = pd.concat(dfs_to_aggregate, ignore_index=True, axis=0)
        return stat_df

    def plot_all_variability(self, result_type, stats_df=None, basins="all", months='all',
                             lower_bound=0.1, upper_bound=100, save_fig=True, return_fig=False):
        """ Plot all CFs and show IQR and IPR95"""
        if stats_df is None:
            stats_df = self.get_full_stats(
                result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                aggregate_by_static=True, iterations_limiter=None)

        # Plot
        fig = plt.figure(figsize=(25, 6))
        ax = plt.subplot(1, 1, 1)

        # Plot intervals
        ax.fill_between(
            x=stats_df['static'].tolist(),
            y1=stats_df['2.5%ile'].tolist(),
            y2=stats_df['97.5%ile'].tolist(),
            color={'xkcd:steel blue'},
            lw=0.001,
            label="Interprecentile range 95%"
        )
        ax.fill_between(
            x=stats_df['static'].tolist(),
            y1=stats_df['Q1'].tolist(),
            y2=stats_df['Q3'].tolist(),
            color={'xkcd:sky blue'},
            lw=0.002,
            label="Interquartile range"
        )
        # Plot static
        ax.plot(stats_df['static'], stats_df['static'], color='black', label="Static value", lw=2)

        # set plot titles
        title_dict = {
            'average_unknown_cf': "Average CFs, unspecified",
            'average_agri_cf': "Average CFs, agricultural",
            'average_non_agri_cf': "Average CFs, non-agricultural",
            'monthly_cf_all': "Monthly CFs",
            'AMD_world_over_AMD_i': "Availability Minus Demand ratio"
        }

        ax.set_ylabel("Stochastic results")
        ax.set_xlabel("Static results")
        ax.set_title(title_dict[result_type])
        ax.legend()

        if save_fig:
            fig.savefig(self.graphs_dir / "all_variability_{}.jpeg".format(self.result_type_name_dict[result_type].replace(" ", "_")), bbox_inches="tight")
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None

    @staticmethod
    def augment_stats_df_with_dispersion(stats_df):
        stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']
        stats_df['IQR/static'] = stats_df['IQR'] / stats_df['static']
        stats_df['IPR-95%'] = stats_df['97.5%ile'] - stats_df['2.5%ile']
        stats_df['IPR-95%/static'] = stats_df['IPR-95%'] / stats_df['static']
        stats_df['sd/static'] = stats_df['sd'] / stats_df['static']
        return stats_df


    def plot_dispersion_violins_normalized(self, result_type, summary_stats=None, basins="all", months='all',
                                lower_bound=0.1, upper_bound=100, save_fig=True, return_fig=False):
        """ From a summary dataframe, plot violin plots for three measures of dispersion"""
        if summary_stats is None:
            summary_stats = self.get_full_stats(
                result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                aggregate_by_static=False, iterations_limiter=None,
                augment_with_dispersion=True
            )

        fig, (ax_CV, ax_IQ, ax_IP) = plt.subplots(3, 1, sharex=True, figsize=(12, 5))
        IP = (summary_stats['97.5%ile'].values - summary_stats['2.5%ile'].values) / summary_stats['static']
        IQ = (summary_stats['Q3'].values - summary_stats['Q1'].values) / summary_stats['static']
        CV = summary_stats['sd'].values / summary_stats['static']
        sns.violinplot(ax=ax_IP, data=IP, orient="h", cut=0)
        sns.violinplot(ax=ax_IQ, data=IQ, orient="h", cut=0)
        sns.violinplot(ax=ax_CV, data=CV, orient="h", cut=0)
        ax_IP.set_title("Interpercentile range (95%)/static")
        ax_IQ.set_title("Interquartile range/static")
        ax_CV.set_title("Standard deviation/static")

        title_dict = {
            'average_unknown_cf': "Average CFs, unspecified ({} results)",
            'average_agri_cf': "Average CFs, agricultural ({} results)",
            'average_non_agri_cf': "Average CFs, non-agricultural ({} results)",
            'monthly_cf_all': "Monthly CFs ({} results)",
            'AMD_world_over_AMD_i': "Availability Minus Demand ratio ({} results)"
        }
        ax_IP.set_xlabel("Dispersion / static CF")
        fig.suptitle(title_dict[result_type].format(IP.size), va="bottom", fontsize=16)
        if save_fig:
            fig.savefig(self.graphs_dir / "dispersion_violins_{}.jpeg".format(result_type), bbox_inches="tight")
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None


    def plot_dispersion_violins_not_normalized(self, result_type, summary_stats=None, basins="all", months='all',
                                lower_bound=0.1, upper_bound=100, save_fig=True, return_fig=False):
        """ From a summary dataframe, plot violin plots for three measures of dispersion"""
        if summary_stats is None:
            summary_stats = self.get_full_stats(
                result_type,
                basins=basins, months=months,
                lower_bound=lower_bound, upper_bound=upper_bound,
                aggregate_by_static=False, iterations_limiter=None,
                augment_with_dispersion=True
            )

        fig, (ax_CV, ax_IQ, ax_IP) = plt.subplots(3, 1, sharex=True, figsize=(12, 5))
        IP = (summary_stats['97.5%ile'].values - summary_stats['2.5%ile'].values)
        IQ = (summary_stats['Q3'].values - summary_stats['Q1'].values)
        CV = summary_stats['sd'].values
        sns.violinplot(ax=ax_IP, data=IP, orient="h", cut=0)
        sns.violinplot(ax=ax_IQ, data=IQ, orient="h", cut=0)
        sns.violinplot(ax=ax_CV, data=CV, orient="h", cut=0)
        ax_IP.set_title("Interpercentile range (95%)")
        ax_IQ.set_title("Interquartile range")
        ax_CV.set_title("Standard deviation")

        title_dict = {
            'average_unknown_cf': "Average CFs, unspecified ({} results)",
            'average_agri_cf': "Average CFs, agricultural ({} results)",
            'average_non_agri_cf': "Average CFs, non-agricultural ({} results)",
            'monthly_cf_all': "Monthly CFs ({} results)",
            'AMD_world_over_AMD_i': "Availability Minus Demand ratio ({} results)"
        }
        ax_IP.set_xlabel("Measures of dispersion")
        fig.suptitle(title_dict[result_type].format(IP.size), va="bottom", fontsize=16)
        if save_fig:
            fig.savefig(self.graphs_dir / "dispersion_violins_unscaled{}.jpeg".format(result_type), bbox_inches="tight")
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None


    def get_sens_args(self, result_type, basin, month, lower_bound=0.1, upper_bound=100):
        """ Extract and format data to sent to GSA (Delta Moment-Independent Measure)"""

        if basin not in self.basins:
           raise ValueError('Basin {} not valid'.format(basin))
        if result_type not in self.result_types_with_months:
            raise ValueError("Only outputs at a monthly resolution are supported. {} is a yearly average".format(result_type))
        if month is None:
            raise ValueError('Month must be specified for {}'.format(result_type))
        if month not in self.months:
            raise ValueError('Month {} not a valid month name'.format(month))

        indices = {param: pickle.load(open(self.indices_dir / "{}.pickle".format(param), 'rb')) for param in self.sampled_variables}
        samples = {param: np.load(str(self.samples_dir / "{}.npy".format(param)), mmap_mode='r') for param in
                   self.sampled_variables}

        output = self.get_stochastic_array(
                    result_type=result_type, basin=basin, month=month,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    iterations_limiter=None
                )

        to_send_to_analysis = {}
        skipped = []

        for param in self.sampled_variables:
            param_one_d = not isinstance(indices[param][0], tuple)
            if param_one_d:
                try:
                    to_send_to_analysis[param] = samples[param][indices[param].index(basin), :] / 12
                except ValueError:
                    skipped.append(param)
            else:
                try:
                    to_send_to_analysis[param] = samples[param][indices[param].index((basin, month)), :]
                except ValueError:
                    skipped.append(param)
        to_send_to_analysis['global_AMD'] = np.load(
            str(self.aggregated_samples / 'AMD_world' / 'AMD_world.npy'),
            mmap_mode='r'
        )
        params = list(to_send_to_analysis.keys())
        array = np.stack([to_send_to_analysis[p] for p in params]).T
        return array, params, output, skipped

    def GSA_single_basin_month(self, result_type, basin, month, lower_bound=0.1, upper_bound=100, overwrite=False):
        """ Calculate GSA"""

        def analyze(X, Y, num_resamples=10, conf_level=0.95):
            """Perform Delta Moment-Independent Analysis on model outputs.

            Returns a dictionary with keys 'delta', 'delta_conf', 'S1', and 'S1_conf',
            where each entry is a list of size D (the number of parameters) containing
            the indices in the same order as the parameter file.

            Parameters
            ----------
            X: A NumPy array containing the model inputs. Parameters are columns, samples are rows.
            Y : A NumPy array containing the model outputs
            num_resamples : int
                The number of bootstrap resamples when computing confidence intervals (default 10)
            conf_level : float
                The confidence interval level (default 0.95)

            References
            ----------
            .. [1] Borgonovo, E. (2007). "A new uncertainty importance measure."
                   Reliability Engineering & System Safety, 92(6):771-784,
                   doi:10.1016/j.ress.2006.04.015.

            .. [2] Plischke, E., E. Borgonovo, and C. L. Smith (2013). "Global
                   sensitivity measures from given data." European Journal of
                   Operational Research, 226(3):536-550, doi:10.1016/j.ejor.2012.11.047.

            """

            D = X.shape[1]
            N = Y.size

            if not 0 < conf_level < 1:
                raise RuntimeError("Confidence level must be between 0-1.")

            # equal frequency partition
            M = min(np.ceil(N ** (2 / (7 + np.tanh((1500 - N) / 500)))), 48)
            m = np.linspace(0, N, M + 1)
            Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

            keys = ('delta', 'delta_conf', 'S1', 'S1_conf')
            S = dict((k, np.zeros(D)) for k in keys)

            for i in range(D):
                S['delta'][i], S['delta_conf'][i] = bias_reduced_delta(
                    Y, Ygrid, X[:, i], m, num_resamples, conf_level)
                S['S1'][i] = sobol_first(Y, X[:, i], m)
                S['S1_conf'][i] = sobol_first_conf(
                    Y, X[:, i], m, num_resamples, conf_level)

            return S

        # Plischke et al. 2013 estimator (eqn 26) for d_hat

        def calc_delta(Y, Ygrid, X, m):
            N = len(Y)
            fy = gaussian_kde(Y, bw_method='silverman')(Ygrid)
            xr = rankdata(X, method='ordinal')

            d_hat = 0
            for j in range(len(m) - 1):
                ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
                nm = len(ix)
                fyc = gaussian_kde(Y[ix], bw_method='silverman')(Ygrid)
                d_hat += (nm / (2 * N)) * np.trapz(np.abs(fy - fyc), Ygrid)

            return d_hat

        # Plischke et al. 2013 bias reduction technique (eqn 30)

        def bias_reduced_delta(Y, Ygrid, X, m, num_resamples, conf_level):
            d = np.zeros(num_resamples)
            d_hat = calc_delta(Y, Ygrid, X, m)

            for i in range(num_resamples):
                r = np.random.randint(len(Y), size=len(Y))
                d[i] = calc_delta(Y[r], Ygrid, X[r], m)

            d = 2 * d_hat - d
            return (d.mean(), norm.ppf(0.5 + conf_level / 2) * d.std(ddof=1))

        def sobol_first(Y, X, m):
            xr = rankdata(X, method='ordinal')
            Vi = 0
            N = len(Y)
            for j in range(len(m) - 1):
                ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
                nm = len(ix)
                Vi += (nm / N) * (Y[ix].mean() - Y.mean()) ** 2
            return Vi / np.var(Y)

        def sobol_first_conf(Y, X, m, num_resamples, conf_level):
            s = np.zeros(num_resamples)

            for i in range(num_resamples):
                r = np.random.randint(len(Y), size=len(Y))
                s[i] = sobol_first(Y[r], X[r], m)

            return norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)

        if result_type not in ['monthly_cf_all', 'AMD_world_over_AMD_i']:
            raise ValueError("Only outputs at a monthly resolution are supported. {} is a yearly average".format(result_type))

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        save_to_dir = self.GSA_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        save_to_dir.mkdir(parents=True, exist_ok=True)
        error_dir = save_to_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        already_done = os.listdir(save_to_dir) + os.listdir(error_dir)
        file_name = "GSA_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)

        if file_name in already_done and not overwrite:
            pass
        else:
            array, params, output, skipped = self.get_sens_args(
                result_type=result_type, basin=basin, month=month,
                lower_bound=lower_bound, upper_bound=upper_bound)
            try:
                S = analyze(array, output, num_resamples=10)
                S['params'] = params
                S['month'] = month
                S['basin'] = basin
                pd.DataFrame.from_dict(S).to_pickle(save_to_dir / file_name)
            except:
                info = {
                    "params": params,
                    "output": output
                }
                with open(os.path.join(error_dir, file_name), 'wb') as f:
                    pickle.dump(info, f)

    def GSA_multiple_basins_months(self, result_type, basins='all', months='all', lower_bound=0.1, upper_bound=100, overwrite=False):
        """ GSA on set of basins and months"""

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")
        basins = self.check_basins(basins)
        months = self.check_months(months)

        save_to_dir = self.GSA_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        save_to_dir.mkdir(parents=True, exist_ok=True)
        error_dir = save_to_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            already_done = os.listdir(save_to_dir) + os.listdir(error_dir)
            requested_basinmonths = [(basin, month) for basin, month in itertools.product(basins, months)]

            left_to_do = [
                (basin, month) for basin, month in requested_basinmonths
                if "GSA_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)
                not in already_done
            ]

            print("{} requested, of which {} were already done, calculating {} Rank-order correlation dfs".format(
                len(basins)*len(months), len(basins)*len(months)-len(left_to_do), len(left_to_do)
           ))

        else:
            left_to_do = [(basin, month) for basin, month in itertools.product(basins, months)]
            print("{} requested. Already existing results will be overwritten".format(len(left_to_do)))

        for basin, month in pyprind.prog_bar(left_to_do):
                self.GSA_single_basin_month(result_type, basin, month, lower_bound, upper_bound)

    def GSA_aggregate(self, result_type,  suffix=None, basins='all', months='all', lower_bound=0.1, upper_bound=100,
                      proceed_with_missing=False, return_result=False):
        """ Aggregate all GSA in one df"""

        basins = self.check_basins(basins)
        months = self.check_months(months)

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        save_to_dir = self.GSA_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        assert save_to_dir.is_dir(), "No results in expected location ({}), calculate results first".format(save_to_dir)
        error_dir = save_to_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        requested_basinmonths = [(basin, month) for basin, month in itertools.product(basins, months)]
        present = os.listdir(save_to_dir) + os.listdir(error_dir)
        missing = [
            (basin, month) for basin, month in requested_basinmonths
            if "GSA_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)
            not in present]

        if missing and not proceed_with_missing:
            raise ValueError("Some requested values have not been calculated")
        elif missing and proceed_with_missing:
            print("{} of the {} requested results are missing, will proceed anyways.".format(len(missing), len(requested_basinmonths)))
            requested_basinmonths = [f for f in requested_basinmonths if f not in missing]

        all_df_list = []
        errors = []
        for basin, month in pyprind.prog_bar(requested_basinmonths):
            filename = "GSA_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)
            if filename not in os.listdir(save_to_dir):
                errors.append((basin, month))
            else:
                all_df_list.append(pd.read_pickle(save_to_dir / filename))
        all_GSA = pd.concat(all_df_list, axis=0, ignore_index=True)
        for col in ['delta', 'delta_conf', 'S1', 'S1_conf']:
            all_GSA[col] = all_GSA[col].astype('float32')

        if suffix:
            name = "aggregated_df_{}.pickle".format(suffix)
        else:
            name = "aggregated_df.pickle"
        all_GSA.to_pickle(save_to_dir / name)
        print("{} basin-months were not aggregated because errors were encountered when calculating the global sensitivity: {}".format(len(errors), errors))

        if return_result:
            return all_GSA

    def spearman_rank_order_single_basin_month(
            self, result_type, basin, month, lower_bound=0.1, upper_bound=100,
            overwrite=False
    ):

        if result_type not in ['monthly_cf_all', 'AMD_world_over_AMD_i']:
            raise ValueError("Only outputs at a monthly resolution are supported. {} is a yearly average".format(result_type))

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        save_to_dir = self.spearman_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        save_to_dir.mkdir(parents=True, exist_ok=True)
        error_dir = save_to_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        already_done = os.listdir(save_to_dir) + os.listdir(error_dir)
        file_name = "SpearmanROC_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)

        if file_name in already_done and not overwrite:
            pass
        else:
            array, params, output, skipped = self.get_sens_args(
                result_type=result_type, basin=basin, month=month,
                lower_bound=lower_bound, upper_bound=upper_bound)
            df = pd.DataFrame(
                index=np.arange(len(params)),
                columns=['basin', 'month', 'parameter', 'rank-order correlation', 'p-value']
            )
            for i, param in enumerate(params):
                try:
                    df.loc[i, 'basin'] = basin
                    df.loc[i, 'month'] = month
                    roc, p = spearmanr(array[:, i], output)
                    df.loc[i, 'parameter'] = param
                    df.loc[i, 'rank-order correlation'] = roc
                    df.loc[i, 'p-value'] = p

                except Exception as err:
                    df.loc[i, 'basin'] = basin
                    df.loc[i, 'month'] = month
                    df.loc[i, 'parameter'] = param
                    df.loc[i, 'rank-order correlation'] = str(err)
                    df.loc[i, 'p-value'] = str(err)
                df.to_pickle(save_to_dir / file_name)

    def spearman_rank_order_multiple(self, result_type, basins='all', months='all', lower_bound=0.1, upper_bound=100, overwrite=False):
        """ Spearman ROC for selected basins and months saved to file"""

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")
        basins = self.check_basins(basins)
        months = self.check_months(months)

        save_to_dir = self.spearman_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        save_to_dir.mkdir(parents=True, exist_ok=True)
        error_dir = save_to_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            already_done = os.listdir(save_to_dir) + os.listdir(error_dir)
            requested_basinmonths = itertools.product(basins, months)

            left_to_do = [
                (basin, month) for basin, month in requested_basinmonths
                if "SpearmanROC_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text, upper_bound_text)
                not in already_done
            ]

            print("{} requested, of which {} were already done, calculating {} Rank-order correlation dfs".format(
                len(basins)*len(months), len(basins)*len(months)-len(left_to_do), len(left_to_do)
           ))

        else:
            left_to_do = [(basin, month) for basin, month in itertools.product(basins, months)]
            print("{} requested. Already existing results will be overwritten".format(len(left_to_do)))

        for basin, month in pyprind.prog_bar(left_to_do):
            self.spearman_rank_order_single_basin_month(
                result_type, basin, month, lower_bound, upper_bound)

    def spearman_rank_order_aggregate(self, result_type, suffix=None, basins='all', months='all', lower_bound=0.1, upper_bound=100, return_result=False):
        """ Aggregate all spearman ROC in one df"""

        basins = self.check_basins(basins)
        months = self.check_months(months)

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        save_to_dir = self.spearman_dir / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text)
        assert save_to_dir.is_dir(), "No results in expected location ({}), calculate results first".format(save_to_dir)

        available_file_names = [f.name for f in save_to_dir.iterdir() if f.is_file()]
        file_name_structure = "SpearmanROC_{}_{}_{}_{}_{}.pickle"

        requested_file_names = [file_name_structure.format(
            result_type, basin, month, lower_bound_text, upper_bound_text
        ) for basin, month in itertools.product(basins, months)]

        missing_files = [f for f in requested_file_names if f not in available_file_names]

        if missing_files:
            raise ValueError("{} of {} requested files missing, cannot aggregate".format(len(requested_file_names), len(missing_files)))

        all_df_list = []
        print("Aggregating {} files".format(len(requested_file_names)))
        for f in pyprind.prog_bar(requested_file_names):
            all_df_list.append(pd.read_pickle(save_to_dir / f))
        all_spearman_df = pd.concat(all_df_list, axis=0)
        all_spearman_df['rank-order correlation'] = all_spearman_df['rank-order correlation'].astype('float32')

        if suffix:
            name = "aggregated_df_{}.pickle".format(suffix)
        else:
            name = "aggregated_df.pickle"

        all_spearman_df.to_pickle(save_to_dir / name)

        if return_result:
            return all_spearman_df

    """#############################################################################"""

    def ovls_single_arr_to_disk(self, result_type, basin,
                                month=None, lower_bound=0.1, upper_bound=100,
                                overwrite=False):
        """Calculate OVL for various PDFs for given result"""
        self.check_result_type(result_type)
        assert basin in self.basins
        if result_type in self.result_types_with_months:
            assert month in self.months

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        save_to_dir = self.sim_dir / "fitting" / "all_ovls" / "{}_{}_{}".format(result_type, lower_bound_text, upper_bound_text) / "individual_series"
        save_to_dir.mkdir(parents=True, exist_ok=True)

        already_done = os.listdir(save_to_dir)

        if result_type in self.result_types_with_months:
            file_name = "ovls_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text,
                                                            upper_bound_text)
        else:
            file_name = "ovls_{}_{}_{}_{}.pickle".format(result_type, basin, lower_bound_text,
                                                            upper_bound_text)
        if file_name in already_done and not overwrite:
            pass
        else:
            arr = self.get_stochastic_array(result_type, basin, month, lower_bound, upper_bound)
            df = try_all_stats_return_all_OVL(arr)
            ser = pd.Series(index=df['fitting_function'], data=list(df['ovl']))
            ser.loc['basin'] = basin
            if result_type in self.result_types_with_months:
                ser.loc['month'] = month
            ser.name = file_name[0:-7]
            ser.to_pickle(save_to_dir / file_name)

    def ovls_single_arr_to_disk_multiple(self, result_type, basins='all', months='all', lower_bound=0.1, upper_bound=100, overwrite=False):
        """ Spearman ROC for selected basins and months saved to file"""

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")
        basins = self.check_basins(basins)
        months = self.check_months(months)

        save_to_dir = self.sim_dir / "fitting" / "all_ovls" / "{}_{}_{}".format(result_type, lower_bound_text,
                                                                  upper_bound_text) / "individual_series"
        save_to_dir.mkdir(parents=True, exist_ok=True)



        already_done = os.listdir(save_to_dir)

        if result_type in self.result_types_with_months:
            requested_basinmonths = itertools.product(basins, months)

            if not overwrite:
                left_to_do = [
                    (basin, month) for basin, month in requested_basinmonths
                    if "ovls_{}_{}_{}_{}_{}.pickle".format(result_type, basin, month, lower_bound_text,
                                                            upper_bound_text)
                       not in already_done
                ]

                print("{} requested, of which {} were already done, calculating {} sets of OVLs".format(
                    len(basins) * len(months), len(basins) * len(months) - len(left_to_do), len(left_to_do)
                ))
            else:
                left_to_do = [(basin, month) for basin, month in itertools.product(basins, months)]
                print("{} requested. Already existing results will be overwritten".format(len(left_to_do)))

            for basin, month in pyprind.prog_bar(left_to_do):
                self.ovls_single_arr_to_disk(
                    result_type, basin, month, lower_bound, upper_bound, overwrite=overwrite)

        else:
            if not overwrite:
                left_to_do = [
                    basin for basin in basins if "ovls_{}_{}_{}_{}.pickle".format(result_type, basin, lower_bound_text,
                                                           upper_bound_text)
                       not in already_done
                ]

                if not left_to_do:
                    print("All requested {} already done".format(result_type))
                    return None
                print("{} requested, of which {} were already done, calculating {} sets of OVLs".format(
                    len(basins), len(basins) - len(left_to_do), len(left_to_do)
                ))
            else:
                left_to_do = basins
                print("{} requested. Already existing results will be overwritten".format(len(left_to_do)))

            for basin in pyprind.prog_bar(left_to_do):
                self.ovls_single_arr_to_disk(
                    result_type, basin, None, lower_bound, upper_bound, overwrite=overwrite)

    def ovls_aggregate(self, result_type, suffix=None, basins='all', months='all', lower_bound=0.1, upper_bound=100, return_result=False):
        """ Aggregate all sets of OVL in one df"""

        basins = self.check_basins(basins)
        if result_type in self.result_types_with_months:
            months = self.check_months(months)
            file_name_structure = "ovls_{}_{}_{}_{}_{}.pickle"
        else:
            file_name_structure = "ovls_{}_{}_{}_{}.pickle"

        lower_bound_text = str(lower_bound).replace(".", "_")
        upper_bound_text = str(upper_bound).replace(".", "_")

        source_dir = self.sim_dir / "fitting" / "all_ovls" / "{}_{}_{}".format(result_type, lower_bound_text,
                                                                  upper_bound_text) / "individual_series"

        save_to_dir = self.sim_dir / "fitting" / "all_ovls" / "{}_{}_{}".format(result_type, lower_bound_text,
                                                                                upper_bound_text)
        assert source_dir.is_dir(), "No results in expected location ({}), calculate results first".format(source_dir)

        available_file_names = [f.name for f in source_dir.iterdir() if f.is_file()]

        if result_type in self.result_types_with_months:
            requested_file_names = [file_name_structure.format(
                result_type, basin, month, lower_bound_text, upper_bound_text
            ) for basin, month in itertools.product(basins, months)]
        else:
            requested_file_names = [file_name_structure.format(
                result_type, basin, lower_bound_text, upper_bound_text
            ) for basin in basins]

        missing_files = [f for f in requested_file_names if f not in available_file_names]

        if missing_files:
            raise ValueError("{} of {} requested files missing, cannot aggregate".format(len(requested_file_names), len(missing_files)))

        all_df_list = []
        print("Aggregating {} files".format(len(requested_file_names)))
        for f in pyprind.prog_bar(requested_file_names):
            all_df_list.append(pd.read_pickle(source_dir / f))
        all_ovl_df = pd.concat(all_df_list, axis=1).T
        all_ovl_df.fillna(0, inplace=True)

        if suffix:
            name = "aggregated_ovls_{}".format(suffix)
        else:
            name = "aggregated_df"

        all_ovl_df.to_pickle(save_to_dir / "{}.pickle".format(name))
        all_ovl_df.to_excel(save_to_dir / "{}.xlsx".format(name))

        if return_result:
            return all_ovl_df