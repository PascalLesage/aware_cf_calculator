from .aware_static import AwareStatic
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import copy
from stats_arrays import UncertaintyBase, MCRandomNumberGenerator
import pyprind
import os
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import warnings


class AwareStochastic(AwareStatic):
    """ Class used to generate samples, calculate cfs stochastically and analyse results

    Instantiation creates core attributes and generates samples if they do not already exist.

    Calculation of actual results relies on the following methods:
        AwareStochastic.calculate_AMD_samples:
            Calculate one set of results per iteration per requested variable,
            if results do not already exist.
        AwareStochastic.aggregate_results:
            Aggregate results over all iterations
        AwareStochastic.calculate_all_single_month_cfs_stochastic
            Calculate CFs for given lower_bound and upper_bound values
        AwareStochastic.calculate_cfs_average:



    """

    def __init__(self, base_dir_path, sim_name="all_random", consider_certain=[], iterations=1000):
        """ Creates core attributes and generates samples if they do not already exist.

        Subclasses AwareStatic class, which provides independent variables and
        deterministic results as attributes.

        ARGUMENTS
        ---------
        base_dir_path: string
            Path to the root folder where all processed data is to be saved.
            Passed to the AwareStatic subclass instantiation
        raw_data_fp: string
            Filepath to raw data Excel file, only necessary if the raw
            data has not yet been imported and saved as a pickled DataFrame.
            Passed to the AwareStatic subclass instantiation
        sim_name: string
            Name of simulation, used as directory path for results generated
            for an associated set of argument values
        consider_certain: list of strings
            List of independent variable names that will not be sampled and for
            which the static value will be used in calculations
        iterations: int
            Number of iterations to use in the Monte Carlo Simulations

        """
        AwareStatic.__init__(self, base_dir_path)
        self.consider_certain = consider_certain
        self.sim_name = sim_name
        self.specify_MC_paths()
        self.generate_base_log_file(consider_certain, iterations)
        self.iterations = self.log['iterations']
        self.check_if_samples_generated()
        if self.samples_generated:
            print("Samples already exist.")
        else:
            print("Samples need to be generated.")
            self.generate_samples()
        assert self.right_number_of_samples, 'wrong number of samples, change required iterations or delete existing samples'

    def generate_base_log_file(self, consider_certain, iterations):
        """ Generate, if required, a log file"""
        self.log_fp = self.sim_dir / "log_file.pickle"
        if not self.log_fp.is_file():
            self.log = {}
            self.log['first_instantiated_date'] = str(datetime.datetime.now())
            self.log['sim_name'] = self.sim_name
            if not consider_certain:
                self.log["consider_certain"] = []
            else:
                self.log["consider_certain"] = consider_certain
            if not iterations:
                self.log["iterations"] = 1000
            else:
                self.log["iterations"] = iterations
            with open(self.log_fp, "wb") as f:
                pickle.dump(self.log, f)
        else:
            with open(self.log_fp, "rb") as f:
                self.log = pickle.load(f)

    def update_log(self, k, v):
        """ Update log file with new information"""
        self.log[k] = v
        with open(self.log_fp, "wb") as f:
            pickle.dump(self.log, f)

    def specify_MC_paths(self):
        """ Create directories to store stochastic results"""
        # Samples
        self.samples_dir = self.base_dir / "independent_variable_samples" / "samples"
        self.samples_dir.mkdir(exist_ok=True, parents=True)
        # Sample indices
        self.indices_dir = self.base_dir / "independent_variable_samples" / "samples_indices"
        self.indices_dir.mkdir(exist_ok=True)
        # Base directory for stochastic results
        self.MC_dir = self.base_dir / "stochastic_results"
        self.MC_dir.mkdir(exist_ok=True)
        # Base directory for a given simulation
        # Note: simulations may differ in what variables are held constant
        self.sim_dir = self.MC_dir / self.sim_name
        self.sim_dir.mkdir(exist_ok=True)
        # Raw samples directory for given simulation
        self.raw_samples = self.sim_dir / "raw"
        self.raw_samples.mkdir(exist_ok=True)
        # Indices for given simulation
        self.total_indices = self.sim_dir / "indices"
        self.total_indices.mkdir(exist_ok=True)
        # Aggregated samples for given simulation
        self.aggregated_samples = self.sim_dir / "aggregated"
        self.aggregated_samples.mkdir(exist_ok=True)
        # GSA result for given simulation
        self.GSA_dir = self.sim_dir / "GSA"
        self.GSA_dir.mkdir(exist_ok=True)
        # Spearman Rank-Order Correlation
        self.spearman_dir = self.sim_dir / "SpearmanRankOrderCorr"
        self.spearman_dir.mkdir(exist_ok=True)

        # CF dir
        self.cf_dir = self.sim_dir / "cfs"
        self.cf_dir.mkdir(exist_ok=True)
        # Graphs
        self.graphs_dir = self.sim_dir / "graphs"
        self.graphs_dir.mkdir(exist_ok=True)


    def check_if_samples_generated(self):
        samples_exist = True
        right_number_of_iterations = True
        sampled_files_available = [f for f in self.samples_dir.iterdir()]
        indices_files_available = [f for f in self.indices_dir.iterdir()]
        required_variables = [f for f in self.given_variable_names if f not in ['area', 'uncertainty']]
        for variable in required_variables:
            samples_exist *= self.samples_dir / "{}.npy".format(variable) in sampled_files_available
            samples_exist *= self.indices_dir / "{}.pickle".format(variable) in indices_files_available
        if samples_exist:
            right_number_of_iterations *= np.load(self.samples_dir / "{}.npy".format(variable)).shape[1] == self.iterations
        else:
            pass
        self.samples_generated = samples_exist
        self.right_number_of_samples = right_number_of_iterations

    def generate_samples(self):
        """ Generate an array of sampled values for uncertain elements of given df

        Returns both a samples array and a list of indices that represent the
        basin or the basin/month for each sample array row.
        These can subsequently be used to replace values in a stacked version of
        dataframes used in calculations.
        """
        random_variables = [v for v in self.given_variable_names if v!='area']
        for variable in random_variables:
            with open(self.filtered_pickles / "{}.pickle".format(variable), 'rb') as f:
                det_df = pickle.load(f)

            if det_df.shape[1] == 12: # Variables with monthly resolution
                stacked_df = det_df.stack().reset_index()
                stacked_df.rename(columns={'level_1': 'month', 0: 'value'}, inplace=True)

            else: # Variables with no monthly resolution
                stacked_df = det_df.reset_index()
                stacked_df.rename(columns={variable: 'value'}, inplace=True)

            if variable not in ['pastor']:  # All uncertain parameters lognormally distributed except pastor
                uncertainty_df = pd.read_pickle(self.filtered_pickles / 'uncertainty.pickle')
                uncertainty = uncertainty_df[variable]
                stacked_df['GSD2'] = stacked_df['BAS34S_ID'].map(uncertainty)
                non_zero = stacked_df['value'] != 0
                uncertain = stacked_df['GSD2'] > 1
                lognormally_distributed = non_zero & uncertain
                df_only_uncertain = copy.deepcopy(stacked_df[lognormally_distributed])
                gsd2_to_scale = lambda x: np.log(np.sqrt(x))
                amount_to_loc = lambda x: np.log(np.absolute(x))
                df_only_uncertain['loc'] = df_only_uncertain['value'].apply(amount_to_loc)
                df_only_uncertain['scale'] = df_only_uncertain['GSD2'].apply(gsd2_to_scale)
                df_only_uncertain['negative'] = df_only_uncertain['value'] < 0
                df_only_uncertain['uncertainty_type'] = 2
                df_only_uncertain['uncertainty_type'] = df_only_uncertain['uncertainty_type'].astype(int)
                d_list = df_only_uncertain.to_dict(orient='records')
                stats_arrays_input_dicts = [
                    {
                        'loc': d['loc'],
                        'scale': d['scale'],
                        'negative': d['negative'],
                        'uncertainty_type': d['uncertainty_type']
                    } for d in d_list
                ]

            else:  # For pastor, which has a triangular distribution, Min=Mode, Max = 1.5*Mode
                include = stacked_df['value'] != 0
                df_only_uncertain = copy.deepcopy(stacked_df[include])

                df_only_uncertain['loc'] = df_only_uncertain['value']
                df_only_uncertain['minimum'] = df_only_uncertain['value']
                max_out_to_one = lambda x: 1.5 * x if 1.5 * x < 1 else 1
                df_only_uncertain['maximum'] = df_only_uncertain['loc'].apply(max_out_to_one)
                df_only_uncertain['uncertainty_type'] = 5
                df_only_uncertain['uncertainty_type'] = df_only_uncertain['uncertainty_type'].astype(int)
                d_list = df_only_uncertain.to_dict(orient='records')
                stats_arrays_input_dicts = [
                    {
                        'loc': d['loc'],
                        'minimum': d['minimum'],
                        'maximum': d['maximum'],
                        'uncertainty_type': d['uncertainty_type']
                    } for d in d_list
                ]
            uncertainty_base = UncertaintyBase.from_dicts(*stats_arrays_input_dicts)
            rng = MCRandomNumberGenerator(uncertainty_base)
            arr = np.zeros(shape=[df_only_uncertain.shape[0], self.iterations])
            for iteration in range(self.iterations):
                arr[:, iteration] = next(rng)

            if 'month' in df_only_uncertain.columns:
                indices_as_arr = np.array([df_only_uncertain['BAS34S_ID'], df_only_uncertain['month']]).T
                indices_as_list = [(indices_as_arr[i, 0], indices_as_arr[i, 1]) for i in range(indices_as_arr.shape[0])]
            else:
                indices_as_list = list(df_only_uncertain['BAS34S_ID'].values)

            np.save(self.samples_dir / "{}.npy".format(variable), arr)
            with open(self.indices_dir / "{}.pickle".format(variable), 'wb') as f:
                pickle.dump(indices_as_list, f)
            print("\t{} samples taken for {}".format(self.iterations, variable))
        self.samples_generated = True

    @staticmethod
    def update_df_with_sample(sample_index, sample_values, det_df):
        """ Update a df of dimension basin x month with sampled values

        Arguments:
            `sample_index`:
                list of tuples that are used to generate the index
                of series containing the sampled values
            `sample_values`:
                one-dimensional array of sampled values, of dimension
                and order equal to that of `sample_index` 
            `det_df`: DataFrame to be updated, populated with default, static data.

            Together, `sample_index` and `sample_values` are used to generate a `sample_series` that is used to update `det_df`

        Returns an updated DataFrame
        """
        sample_series = pd.Series(index=sample_index, data=sample_values)
        if det_df.shape[1]>1:
            updated = det_df.stack()
            updated.index.set_names(level=1, names='month', inplace=True)
            updated.update(sample_series)
            updated = updated.unstack(level='month')
        else:
            updated=copy.deepcopy(det_df)
            updated.iloc[:, 0].update(sample_series)
        return updated

    @staticmethod
    def update_series_with_sample(sample_index, sample_values, det_series):
        """ Update a Series of dimension basin with sampled values

        Arguments:
            `sample_index`:
                list of basin ids that are used to generate the index
                of series containing the sampled values
            `sample_values`:
                one-dimensional array of sampled values, of dimension
                and order equal to that of `sample_index` 
            `det_series`: Series to be updated, populated with default, static data.

            Together, `sample_index` and `sample_values` are used to generate a
            `sample_series` that is used to update `det_series`

        Returns an updated Series
        """

        sample_series = pd.Series(index=sample_index, data=sample_values)
        new_series = copy.deepcopy(det_series)
        new_series.update(sample_series)
        return new_series

    def prep_samples(self):
        self.samples_dict = {}
        self.samples_indices_dict = {}
        for variable in [f for f in self.given_variable_names
            if f != 'area'
            and f != 'uncertainty'
            and f not in self.consider_certain
            ]:
            self.samples_dict[variable] = np.load(str(self.samples_dir / "{}.npy".format(variable)), mmap_mode='r')
            with open(self.indices_dir / "{}.pickle".format(variable), 'rb') as f:
                indices_precursor = pickle.load(f)
            if isinstance(indices_precursor[0], tuple):
                self.samples_indices_dict[variable] = pd.MultiIndex.from_tuples(indices_precursor)
            else:
                self.samples_indices_dict[variable] = indices_precursor
        return None

    def update_data(self, i):
        # Update all dfs and series
        data_d_MC = {'area': self.det_data['area']}

        for variable in [
            f for f in self.given_variable_names
            if f != 'area']:
            if variable not in self.consider_certain:
                data_d_MC[variable] = AwareStochastic.update_df_with_sample(
                    self.samples_indices_dict[variable],
                    self.samples_dict[variable][:, i],
                    self.det_data[variable]
                )
            else:
                data_d_MC[variable] = self.det_data[variable]
        return data_d_MC

    def calculate_AMD_samples(self, results_to_save=['AMD_world', 'AMD_world_over_AMD_i', 'irrigation', 'HWC']):
        """ Calculate results for a given list of results.

        Results are stored as float (AMD_world) or numpy arrays (all others), with
         one result per iteration.
        These then need to be reassembled into arrays with all iterations in one array,
        see static method 'aggregate_results'.

        The default list of results is:
            AMD_world_over_AMD_i: array of dimension basins x month
            AMD_world: float
            irrigation: array of dimension basins x month
            HWC: array of dimension basins x month
        All results are stored in the MC.raw directory.
        Results already on file are not recalculated. To recalculate a given result, first delete
        the existing results.
        """
        # Load self.samples_dict and self.samples_indices_dict
        self.prep_samples()

        # Identify which iterations were already calculated for a given result type
        find_iteration = lambda s: int(s[::-1][0:s[::-1].find("_")].strip("ypn.")[::-1])
        already_calculated = {}
        for wanted_result in results_to_save:
            result_path = self.raw_samples / wanted_result
            result_path.mkdir(parents=True, exist_ok=True)
            already_calculated[wanted_result] = [find_iteration(result)
                                                 for result in os.listdir(result_path)]
        print("ready to start with ", wanted_result, len(already_calculated[wanted_result]), "already calculated")
        # Generate results
        for i in pyprind.prog_bar(range(self.iterations)):
            iteration_done = {wanted_result: i in already_calculated[wanted_result]
                              for wanted_result in results_to_save
                              }

            # Check if all results already exist for given iteration
            if not all([iteration_done[wanted_result] for wanted_result in results_to_save]):
                # If not, calculate results for given iteration
                data_d_MC = self.update_data(i)
                result_d = self.calc_AMDs(data_d_MC)
                # In first iteration, store if required the indices and columns of the result
                if i == 0:
                    for wanted_result in results_to_save:
                        # Get the result, be it in the result dict or data dict
                        try:
                            result = result_d[wanted_result]
                        except:
                            result = data_d_MC[wanted_result]
                        try:
                            indices = list(result.index)
                            index_file = self.total_indices / "{}_index.pickle".format(
                                    wanted_result
                                )
                            if not index_file in [f for f in self.total_indices.iterdir()]:
                                with open(index_file, 'wb') as f:
                                    pickle.dump(indices, f)
                        except:
                            # No index for this object, e.g. world_AMD
                            pass
                        try:
                            columns = list(result.columns)
                            column_file = self.total_indices / "{}_columns.pickle".format(
                                    wanted_result
                                )
                            if not column_file in [f for f in self.total_indices.iterdir()]:
                                with open(column_file, 'wb') as f:
                                    pickle.dump(columns, f)
                        except:
                            # No columns for this object, e.g. world_AMD
                            pass

                for wanted_result in results_to_save:
                    try:
                        arr = result_d[wanted_result]
                    except KeyError:
                        arr = data_d_MC[wanted_result]
                    np.save(
                        self.raw_samples / wanted_result / "{}_{}.npy".format(
                            wanted_result, i),
                        arr
                    )

    def aggregate_results(self, results_to_aggregate=['HWC', 'AMD_world', 'AMD_world_over_AMD_i', 'irrigation']
                          ):
        for result_type in results_to_aggregate:
            orig_dir = self.raw_samples / result_type
            all_files = os.listdir(orig_dir)
            first_file = np.load(str(orig_dir / all_files[0]), mmap_mode="r")
            if np.ndim(first_file)==0:
                self.aggregate_0dim_results(result_type)
            else:
                self.aggregate_2d_results(result_type)


    def aggregate_0dim_results(self, result_type):
        """ Aggregate float results for a given result type

        Typically used for world_AMD
        """
        orig_dir = self.raw_samples / result_type
        dest_dir = self.aggregated_samples / result_type
        dest_dir.mkdir(exist_ok=True)
        all_files_names = os.listdir(orig_dir)
        iterations = len(all_files_names)
        ordered_files = ["{}_{}.npy".format(result_type, i) for i in range(iterations)]
        assert all([f in all_files_names for f in ordered_files])
        expected_file = os.path.join(dest_dir, "{}.npy".format(result_type))
        found_files = [str(f) for f in dest_dir.iterdir()]
        if expected_file in found_files:
            print("{} already aggregated".format(result_type))
        else:
            print("Aggregating {}".format(result_type))
            if "{}.npy".format(result_type) in os.listdir(dest_dir):
                pass
            else:
                arr = np.zeros(shape=iterations)
                for i in range(iterations):
                    f = ordered_files[i]
                    arr[i] = np.load(os.path.join(orig_dir, f)).item()
                np.save(os.path.join(dest_dir, "{}.npy".format(result_type)), arr)

    def aggregate_2d_results(self, result_type):
        """ Convert n variable x 1 iteration to 1 variable x m iteration arrays"""

        # Get indices and columns
        with open(self.total_indices / "{}_index.pickle".format(result_type), 'rb') as f:
            indices = pickle.load(f)
        with open(self.total_indices / "{}_columns.pickle".format(result_type), 'rb') as f:
            columns = pickle.load(f)

        orig_dir = self.raw_samples / result_type
        dest_dir = self.aggregated_samples / result_type
        dest_dir.mkdir(exist_ok=True)
        all_files_in_dest_dir = os.listdir(dest_dir)
        expected_files = ["{}_{}_{}.npy".format(result_type, index, col)
                          for index in indices for col in columns]
        files_to_treat = [f for f in expected_files if f not in all_files_in_dest_dir]
        if len(files_to_treat)==0:
            print("{} already aggregated".format(result_type))
            pass
        else:
            print("{} to treat for {}".format(len(files_to_treat), result_type))
            all_files_names = os.listdir(orig_dir)
            iterations = len(all_files_names)
            ordered_files = ["{}_{}.npy".format(result_type, i) for i in range(iterations)]
            assert all([f in all_files_names for f in ordered_files])
            sample_dict = {file_name: np.load(str(orig_dir / file_name), mmap_mode='r')
                           for file_name in ordered_files
                           }
            for i in pyprind.prog_bar(range(len(indices))):
                index = indices[i]
                for j, col in enumerate(columns):
                    if "{}_{}_{}.npy".format(result_type, index, col) in all_files_in_dest_dir:
                        pass
                    else:
                        try:
                            arr = np.zeros(shape=iterations)
                            for iteration, f in enumerate(ordered_files):
                                arr[iteration] = sample_dict[f][i, j]
                            np.save(os.path.join(dest_dir, "{}_{}_{}.npy".format(result_type, index, col)), arr)
                        except Exception as err:
                            print("Failure with: ", i, j, str(err))

    def calculate_single_month_cf_stochastic(self, basin, month, lower_bound=0.1, upper_bound=100):
        """ Calculate characterization factors using the specified bounds."""

        AMD_world_over_AMD_i_arr = np.load(
            self.aggregated_samples / "AMD_world_over_AMD_i" / "AMD_world_over_AMD_i_{}_{}.npy".format(basin, month))
        cf_arr = AwareStatic.calc_aware_cf_arrays(AMD_world_over_AMD_i_arr, lower_bound, upper_bound)
        cf_folder = "{}_{}".format(lower_bound, upper_bound).replace(".", "_")
        month_cf_folder = self.cf_dir / cf_folder / "monthly"
        month_cf_folder.mkdir(exist_ok=True, parents=True)
        np.save(month_cf_folder / "cf_{}_{}.npy".format(basin, month), cf_arr)

    def calculate_all_single_month_cfs_stochastic(self, lower_bound, upper_bound, overwrite=False):
        """ Calculate all monthly cfs for specified lower and upper bounds"""

        # Get list of basins
        get_basin = lambda s: int(s[::-1][8:][::-1].replace('AMD_world_over_AMD_i_', ""))
        orig_dir = self.aggregated_samples / "AMD_world_over_AMD_i"
        dest_dir = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "monthly"
        for AMD in pyprind.prog_bar([f for f in orig_dir.iterdir()]):
            basin = get_basin(AMD.name)
            month = AMD.name[-7:-4]
            if (dest_dir / "cf_{}_{}.npy".format(basin, month)).is_file() and not overwrite:
                pass
            else:
                self.calculate_single_month_cf_stochastic(basin, month, lower_bound, upper_bound)

    def calculate_cfs_average(self, lower_bound, upper_bound, overwrite=False):
        """ Calculate average cfs for given bounds

        Monthly cfs must be calculated first,
        and irrigation/HWC data must be available.
        """
        monthly_dir = self.cf_dir / \
                      "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "monthly"
        assert monthly_dir.is_dir(), "Need to calculate monthly cfs first"
        assert len(os.listdir(monthly_dir)) != 0, "Need to calculate monthly cfs first"

        unknown_dest_dir = self.cf_dir / \
                           "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_unknown"
        unknown_dest_dir.mkdir(exist_ok=True)
        agri_dest_dir = self.cf_dir / \
                        "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_agri"
        agri_dest_dir.mkdir(exist_ok=True)
        non_agri_dest_dir = self.cf_dir / \
                            "{}_{}".format(lower_bound, upper_bound).replace(".", "_") / "average_non_agri"
        non_agri_dest_dir.mkdir(exist_ok=True)
        print("Will generate average cfs for {} basins".format(len(self.basins)))
        for basin in pyprind.prog_bar(self.basins):
            unknown_exists = (unknown_dest_dir / "cf_average_unknown_{}.npy".format(basin)).is_file()
            agri_exists = (agri_dest_dir/ "cf_average_agri_{}.npy".format(basin)).is_file()
            non_agri_exists = (non_agri_dest_dir/ "cf_average_non_agri_{}.npy".format(basin)).is_file()
            if all([unknown_exists, agri_exists, non_agri_exists]) and not overwrite:
                continue
            month_cfs = np.vstack(
                [
                    np.load(monthly_dir / "cf_{}_{}.npy".format(basin, month))
                    for month in self.months
                ]
            )
            irrigation_dir = self.aggregated_samples / "irrigation"
            irrigation = np.vstack(
                [
                    np.load(irrigation_dir / "irrigation_{}_{}.npy".format(basin, month))
                    for month in self.months
                ]
            )
            HWC_dir = self.aggregated_samples / "HWC"
            HWC = np.vstack(
                [
                    np.load(HWC_dir / "HWC_{}_{}.npy".format(basin, month))
                    for month in self.months
                ]
            )
            cfs_unknown = np.zeros(shape=(month_cfs.shape[1]))
            has_HWC = HWC.sum(axis=0) != 0
            cfs_unknown[has_HWC] = \
                (HWC[:, has_HWC] * month_cfs[:, has_HWC]).sum(axis=0) \
                / HWC[:, has_HWC].sum(axis=0)
            cfs_unknown[~has_HWC] = month_cfs[:, ~has_HWC].mean(axis=0)
            np.save(unknown_dest_dir / "cf_average_unknown_{}.npy".format(basin), cfs_unknown)

            cfs_agri = np.zeros(shape=(month_cfs.shape[1]))
            has_irrigation = irrigation.sum(axis=0) != 0
            cfs_agri[has_irrigation] = \
                (irrigation[:, has_irrigation] * month_cfs[:, has_irrigation]).sum(axis=0) \
                / irrigation[:, has_irrigation].sum(axis=0)
            cfs_agri[~has_irrigation] = month_cfs[:, ~has_irrigation].mean(axis=0)
            np.save(agri_dest_dir / "cf_average_agri_{}.npy".format(basin), cfs_agri)

            cfs_non_agri = month_cfs.mean(axis=0)
            np.save(non_agri_dest_dir / "cf_average_non_agri_{}.npy".format(basin), cfs_non_agri)

    def get_arr_sampled_data(self, name, basin, month=None):
        """ Load and return sample array from variable name, basin and, when relevant, month"""

        with open(Path(self.indices_dir)/"{}.pickle".format(name), "rb") as f:
            indices = pickle.load(f)
        try:
            if isinstance(indices[0], tuple):
                index = indices.index((basin, month))
            else:
                index = indices.index(basin)
            return np.load(str(Path(self.samples_dir)/"{}.npy".format(name)), mmap_mode='r')[index, :]
        except:
            if month:
                print("Could not get {} for basin {} and month {}".format(name, basin, month))
            else:
                print("Could not get {} for basin {}".format(name, basin))
            return None


    def summarize_multiple_variables(self, basin, month, iterations):
        """Get summary df for key input variables"""

        summary_df = pd.DataFrame(index=np.arange(iterations))
        for i, v in enumerate([
            "irrigation",
            "domestic",
            "electricity",
            "livestock",
            "manufacturing",
            "pastor",
            "avail_delta",
            "avail_net",
        ]):
            arr = self.get_arr_sampled_data(v, basin, month)
            if arr is not None:
                summary_df[v]=arr[0:iterations]
            else:
                summary_df[v]=np.zeros(shape=(iterations))
        print(summary_df)
        # Calcs
        summary_df['HWC'] = summary_df['irrigation'] + summary_df['domestic'] / 12 \
              + summary_df['electricity'] / 12 + summary_df['livestock'] / 12 \
              + summary_df['manufacturing'] / 12
        summary_df['avail_before_human_consumption'] = summary_df['HWC'] + summary_df['avail_net']
        summary_df['pristine'] = summary_df['avail_before_human_consumption'] + summary_df['avail_delta']
        summary_df['EWR'] = summary_df['pristine'] * summary_df['pastor']
        summary_df['AMD'] = summary_df['avail_before_human_consumption'] - summary_df['EWR'] - summary_df["HWC"]
        return summary_df.T

    def convert_dfs_to_csv(self, lower_bound, upper_bound):
        """ Convert numpy arrays of CFs to CSV"""

        cf_folder = self.cf_dir / "{}_{}".format(lower_bound, upper_bound).replace(".", "_")
        if not cf_folder.is_dir():
            warnings.warn("No CFs found in directory {}. No csv generated.".format(str(cf_folder)))
            return
        for cf_type in [
            "monthly",
            "average_unknown",
            "average_agri",
            "average_non_agri"
        ]:
            np_dir = cf_folder / cf_type
            if not np_dir.is_dir():
                warnings.warn("CFs in npy version do not exist for {}. No csv generated".format(cf_type))
                continue
            else:
                save_dir = self.sim_dir / "csv" / cf_type
                save_dir.mkdir(parents=True, exist_ok=True)
                for f in [f for f in np_dir.iterdir()]:
                    arr = np.load(f)
                    np.savetxt(save_dir / f.name.replace('npy', 'csv'), arr, delimiter=',')

