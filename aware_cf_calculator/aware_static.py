from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import copy
from warnings import warn


class AwareStatic:
    """ Class to calculate CFs for the Aware impact assessment method

    The object is instantiated by specifying:
        (1) a directory where all results and intermediate files are stored and
        (2) a filepath to a formatted Excel file with data on the independent
            (i.e. given) data. This filepath is only necessary if the data has
            not yet been imported.

    The independent variables are:
        Monthly irrigation
            Description: irrigation water, per month, per basin
            Unit: m3/month
            Location in Excel doc: Irrigation
            File name once imported: irrigation.pickle
            table shape: (11050, 12)

        Non-irrigation hwc: electricity, domestic, livestock, manufacturing
            Description: non-irrigation uses of water
            Unit: m3/year
            Location in Excel doc: hwc_non_irrigation
            File name once imported: electricity.pickle, domestic.pickle,
                livestock.pickle, manufacturing.pickle
            table shape: 3 x (11050,)

        avail_delta
            Description: Difference between "pristine" natural availability
                reported in PastorXNatAvail and natural availability calculated
                from "Actual availability as received from WaterGap - after
                human consumption" (Avail!W:AH) plus HWC.
                This should be added to calculated water availability to
                get the water availability used for the calculation of EWR
            Unit: m3/month
            Location in Excel doc: avail_delta
            File name once imported: avail_delta.pickle
            table shape: (11050, 12)

        avail_net
            Description: Actual availability as received from WaterGap - after human consumption
            Unit: m3/month
            Location in Excel doc: avail_net
            File name once imported: avail_net.pickle
            table shape: (11050, 12)

        pastor
            Description: fraction of PRISTINE water availability that should be reserved for environment
            Unit: unitless
            Location in Excel doc: pastor
            File name once imported: pastor.pickle
            table shape: (11050, 12)

        area
            Description: area
            Unit: m2
            Location in Excel doc: area
            File name once imported: area.pickle
            table shape: (11050,)

    The raw data is filtered and stored as more convenient Pandas DataFrames.
    """

    def __init__(self, base_dir_path, raw_data_fp=None):
        """ Instantiate the Aware CF calculator.

        Instantiation readies Aware object for calculation by making necessary
        directories, defining the given (independent) variables used in the
        model, and importing raw data from a formatted Excel file, storing
        pickled Pandas DataFrame for each independent variable.

        Instantiation will automatically try to import data if necessary.

        Instantiation will also set links to result files if data has previously
        been calculated and stored in the corresponding dir.


        ARGUMENTS
        ---------
        base_dir_path: string
            Path to the root folder where all processed data is to be saved.
        raw_data_fp: string
            Filepath to raw data Excel file, only necessary if the raw
            data has not yet been imported and saved as a pickled DataFrame.

        NOTABLE INVOKED METHODS
        ---------------
        self.specify_paths: Creates directories for storing data and results
            Specifically, it sets the following paths as attributes:
                self.base_dir: root directory for data
                self.raw_pickles = directory with dataframes as imported from
                    Excel
                self.filtered_pickles: directory with dataframes after fitering
                    out unwanted basins
                self.det_results: directory with Excel output of static model
                    results
        self.import_raw_files: Imports data from Excel and stores raw data in DataFrames
            Only invoked if files not already imported in self.raw_pickles
        self.filter_files: Removes unwanted basins from DataFrames
            Only invoked if files not already imported in self.raw_pickles
        self.set_static_attribute_dict: Makes data filepaths available via a self.det_data dictionary.
            For example, data for variable "domestic" is accessible via pd.read_pickle(self.det_data['domestic'])
        self.set_result_dict: Make result filepaths other than cfs available via a self.det_data dictionary.
            For example, data result type "AMD_world_over_AMD_i" is accessible via
            pd.read_pickle(self.det_results['AMD_world_over_AMD_i']).
            For cfs, links are contained in a nested dict with a key equal to
            (lower_bound+"_"+upper_bound).replace(".", "_").replace(" ", "_").
            For example, to get cfs_average_agri with lower and upper bounds equal to 0.1 and 100,
            respectively, use pd.read_pickle(self.det_results['0_1_100']['cfs_average_agri'])
        """
        self.given_variable_names = [
            "area",
            "avail_delta",
            "avail_net",
            "domestic",
            "electricity",
            "irrigation",
            "livestock",
            "manufacturing",
            "pastor",
        ]

        self.specify_paths(base_dir_path)
        self.check_if_xls_parsed_and_saved()
        if not self.xls_parsed_and_saved:
            print("Data needs to be imported.")
            assert raw_data_fp, \
                "Data has not been imported, and no path to an Excel file has been supplied"
            self.raw_data_file = Path(raw_data_fp)
            if not self.raw_data_file.parent == self.base_dir / "extracted":
                new_location = self.base_dir / "extracted" / "AWARE_base_data.xlsx"
                shutil.copyfile(str(self.raw_data_file), str(new_location))
                self.imported_raw_data_file = new_location
            self.import_raw_files()
            self.filter_files()
            self.xls_parsed_and_saved=True
        else:
            print("Data already imported")
            self.imported_raw_data_file = self.base_dir / "extracted" / "AWARE_base_data.xlsx"
            if not self.imported_raw_data_file.is_file():
                warn("The imported raw data file is not found in its expected location")
        self.set_static_attribute_dict()
        self.set_basins_and_months_attributes()
        self.check_results_exist()

    def check_results_exist(self):
        """ Check that results exist and set result pickle paths dict as attribute if they are."""

        df_dir = self.det_results_dir / "dfs"
        if not df_dir.is_dir():
            print("No results were calculated, calculate using `det_calcs` method")
            return
        else:
            required_non_cf_results = [
                'AMD_per_m2',
                'AMD_world',
                'AMD_world_over_AMD_i',
                'avail_before_human_consumption',
                'EWR',
                'HWC',
                'pristine',
                'total_AMD',
                'yearly_AMD_per_m2'
            ]
            required_cf_results = [
                'cfs_average_agri',
                'cfs_average_non_agri',
                'cfs_average_unknown',
                'cfs_monthly'
            ]

            available_non_cf_file_stems = [
                f.stem for f in df_dir.iterdir()
                if f.is_file()
            ]
            missing_non_cf_results = [
                result for result in required_non_cf_results
                if result not in available_non_cf_file_stems
            ]
            if missing_non_cf_results:
                print("Non-CF results not yet calculated: calculate using `det_calcs` method")
            else:
                print("Non-CF results already calculated")

            available_cf_sets = []
            available_cf_dirs = [
                f for f in df_dir.iterdir()
                if f.is_dir()
            ]
            for available_cf_dir in available_cf_dirs:
                available_cfs = [
                    f.stem for f in available_cf_dir.iterdir()
                    if f.is_file()
                ]

                missing_cf_results = [
                    cf_result for cf_result in required_cf_results
                    if cf_result not in available_cfs
                ]
                if missing_cf_results:
                    continue
                else:
                    available_cf_sets.append(available_cf_dir.name)
            if not available_cf_sets:
                print("No cfs were calculated, calculate using `det_calcs` method")
            else:
                print("CFs available for {}".format(available_cf_sets))

    def check_if_xls_parsed_and_saved(self):
        """ Check that Excel data has been parsed and saved."""

        ok = True
        unfiltered_files_available = [f for f in self.raw_pickles.iterdir()]
        filtered_files_available = [f for f in self.filtered_pickles.iterdir()]
        required_filtered_variables = self.given_variable_names
        required_unfiltered_variables = ['uncertainty', 'filters']
        for variable in required_filtered_variables:
            if self.filtered_pickles / "{}.pickle".format(variable) in filtered_files_available:
                pass
            else:
                ok *= False
        for variable in required_unfiltered_variables:
            if self.raw_pickles / "{}.pickle".format(variable) in unfiltered_files_available:
                pass
            else:
                ok *= False
        self.xls_parsed_and_saved = ok

    def prep_data_on_file(self):
        """ Get data ready for manipulation. """

        self.transfer_excel_data_to_pickled_df()
        self.filter_files()
        self.excel_parsed_and_df_saved = True

    def set_static_attribute_dict(self):
        """ Set data from pickled dfs as attributes for deterministic calculation """
        self.det_data = {}
        ########### SET PATHS NOT DFS
        for variable in self.given_variable_names:
            self.det_data[variable] = pd.read_pickle(self.filtered_pickles / "{}.pickle".format(variable))
        for variable in ['uncertainty', 'filters']:
            self.det_data[variable] = pd.read_pickle(self.raw_pickles / "{}.pickle".format(variable))

    def set_basins_and_months_attributes(self):
        df = pd.read_pickle(self.filtered_pickles / "avail_delta.pickle")
        self.basins = df.index.tolist()
        self.months = df.columns.tolist()

    def specify_paths(self, base_dir_path):
        """ Set paths to folders used to retrieve or store data"""

        self.base_dir = Path(base_dir_path)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_pickles = self.base_dir / "extracted" / "raw_pickles"
        self.raw_pickles.mkdir(exist_ok=True, parents=True)
        self.filtered_pickles = self.base_dir / "extracted" / "filtered_pickles"
        self.filtered_pickles.mkdir(exist_ok=True)
        self.det_results_dir = self.base_dir / "static_results"
        self.det_results_dir.mkdir(exist_ok=True, parents=True)

    def import_raw_files(self):
        """Import data from the Excel spreadsheet where all the data is contained"""

        assert self.raw_data_file.is_file(), "No file found at {}"
        assert self.raw_data_file.suffix == ".xlsx", "The raw data file should be an Excel"
        xls = pd.ExcelFile(self.raw_data_file)
        sheets_to_get = self.given_variable_names + ['uncertainty', 'filters', 'model_uncertainty']
        for variable_name in sheets_to_get:
            assert variable_name in xls.sheet_names, "Sheet {} missing".format(variable_name)

        print("\nExtracting data from Excel...")
        for variable in sheets_to_get:
            print("...{}".format(variable))
            df = pd.read_excel(self.raw_data_file, sheet_name=variable, index_col=0)
            df.to_pickle(self.raw_pickles/"{}.pickle".format(variable))

    def filter_files(self):
        """ Filter out basins for which pastor values sum to less than 0.01"""

        self.filters_df = pd.read_pickle(self.raw_pickles / "filters.pickle")
        print("\nFiltering out unwanted basins")
        self.filters_df['total_filter'] = self.filters_df.all(axis=1)
        variables_to_strip = self.given_variable_names + ['uncertainty', 'model_uncertainty']
        for v in variables_to_strip:
            print("...{}".format(v))
            unfiltered = pd.read_pickle(self.raw_pickles / "{}.pickle".format(v))
            filtered = unfiltered[self.filters_df['total_filter']]
            print("\t shape changed from {} to {}".format(unfiltered.shape, filtered.shape))
            filtered.to_pickle(self.filtered_pickles / "{}.pickle".format(v))
        df = pd.read_pickle(self.raw_pickles / "filters.pickle".format(v))
        df.to_pickle(self.filtered_pickles / "filters.pickle".format(v))

    def det_calcs(self, lower_bound=0.1, upper_bound=100,
                  dump_excel=False, dump_pickle=True):
        """ Calculate CFs and potentially dump them"""
        "Calculating..."

        self.det_results = self.calc_AMDs(self.det_data)
        self.det_results = AwareStatic.calc_aware_cf_monthly(
            result_d=self.det_results,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        self.det_results = AwareStatic.calc_aware_annual_unknown(
            self.det_results
        )
        self.det_results = AwareStatic.calc_aware_annual_agri(
            self.det_data,
            self.det_results
        )
        self.det_results = AwareStatic.calc_aware_annual_non_agri(
            self.det_results
        )
        if dump_excel:
            dump_dir = self.det_results_dir / "excel"
            dump_dir.mkdir(exist_ok=True)
            cf_dump_dir = dump_dir / "cfs_{}_{}".format(
                str(lower_bound).replace(".", "_"), str(upper_bound).replace(".", "_"))
            cf_dump_dir.mkdir(exist_ok=True)
            for name, results in self.det_results.items():
                if name[0:3]!="cfs":
                    if isinstance(results, float):
                        results = pd.Series(index=["world_AMD"], data=[results])
                    results.to_excel(dump_dir / "{}.xlsx".format(name))
                else:
                    results.to_excel(cf_dump_dir / "{}.xlsx".format(name))
        if dump_pickle:
            dump_dir = self.det_results_dir / "dfs"
            dump_dir.mkdir(exist_ok=True)
            cf_dump_dir = dump_dir / "cfs_{}_{}".format(
                str(lower_bound).replace(".", "_"),str(upper_bound).replace(".", "_"))
            cf_dump_dir.mkdir(exist_ok=True)
            for name, results in self.det_results.items():
                if name[0:3]!="cfs":
                    if isinstance(results, (pd.core.frame.DataFrame, pd.core.series.Series)):
                        results.to_pickle(dump_dir / "{}.pickle".format(name))
                    elif isinstance(results, float):
                        # For world_AMD
                        results = np.array(results)
                        np.save(str(dump_dir / name), results)
                else:
                    results.to_pickle(cf_dump_dir / "{}.pickle".format(name))

    @staticmethod
    def calc_HWC(data_d, result_d):
        non_irrigation_total = data_d['electricity']['electricity'] + \
                               data_d['manufacturing']['manufacturing'] + \
                               data_d['livestock']['livestock'] + \
                               data_d['domestic']['domestic']
        result_d['HWC'] = data_d['irrigation'].add(non_irrigation_total / 12, axis=0)
        return result_d

    @staticmethod
    def calc_avail_before_human_consumption(data_d, result_d):
        result_d['avail_before_human_consumption'] = result_d['HWC'].add(
            data_d['avail_net']
        )
        return result_d

    @staticmethod
    def calc_pristine(data_d, result_d):
        result_d['pristine'] = result_d['avail_before_human_consumption'].add(
            data_d['avail_delta']
        )
        return result_d

    @staticmethod
    def calc_EWR(data_d, result_d):
        result_d['EWR'] = result_d['pristine'].multiply(data_d['pastor'])
        return result_d

    @staticmethod
    def calc_total_AMD(result_d):
        total_consumption = result_d['EWR'].add(result_d['HWC'])
        result_d['total_AMD'] = result_d['avail_before_human_consumption'].add(
            -total_consumption
        )
        return result_d

    @staticmethod
    def calc_AMD_per_m2(data_d, result_d):
        result_d['AMD_per_m2'] = result_d['total_AMD'].divide(
            data_d['area']['area'], axis="index"
        )
        return result_d

    @staticmethod
    def calc_yearly_AMD_per_m2(result_d):
        has_HWC = result_d['HWC'].sum(axis=1) != 0
        yearly_AMD_per_m2 = pd.Series(index=result_d['HWC'].index)
        yearly_AMD_per_m2.loc[has_HWC] = result_d['AMD_per_m2'][has_HWC].multiply(
            result_d['HWC'][has_HWC]).sum(axis=1) / result_d['HWC'][has_HWC].sum(
            axis=1)
        yearly_AMD_per_m2.loc[-has_HWC] = result_d['AMD_per_m2'][-has_HWC].mean(axis=1)
        result_d['yearly_AMD_per_m2'] = yearly_AMD_per_m2
        return result_d

    @staticmethod
    def calc_global_average_AMD(result_d):
        result_d['AMD_world'] = (result_d['HWC'].sum(axis=1).multiply(
            result_d['yearly_AMD_per_m2'])
               ).sum() / result_d['HWC'].sum(axis=1).sum(axis=0)
        return result_d

    @staticmethod
    def calc_AMD_world_over_AMD_i(result_d):
        np.seterr(divide='ignore')
        AMD_world_over_AMD_i = result_d['AMD_world']/ result_d['AMD_per_m2']
        #AMD_world_over_AMD_i = AMD_world_over_AMD_i.replace(np.inf, np.nan)
        result_d['AMD_world_over_AMD_i'] = AMD_world_over_AMD_i
        return result_d

    @staticmethod
    def calc_AMDs(data_d):
        """ Wrapper function to calculate global average AMD and basin-specific AMD ratios, per month"""
        result_d = {}
        result_d = AwareStatic.calc_HWC(data_d, result_d)
        result_d = AwareStatic.calc_avail_before_human_consumption(data_d, result_d)
        result_d = AwareStatic.calc_pristine(data_d, result_d)
        result_d = AwareStatic.calc_EWR(data_d, result_d)
        result_d = AwareStatic.calc_total_AMD(result_d)
        result_d = AwareStatic.calc_AMD_per_m2(data_d, result_d)
        result_d = AwareStatic.calc_yearly_AMD_per_m2(result_d)
        result_d = AwareStatic.calc_global_average_AMD(result_d)
        result_d = AwareStatic.calc_AMD_world_over_AMD_i(result_d)
        return result_d

    @staticmethod
    def calc_aware_cf_monthly(result_d, lower_bound=0.1, upper_bound=100):
        """ Calculate characterization factors from AMD ratio DataFrame"""
        AMD_arr = copy.deepcopy(result_d['AMD_world_over_AMD_i'].values)
        cf_arr = AwareStatic.calc_aware_cf_arrays(AMD_arr, lower_bound, upper_bound)
        result_d['cfs_monthly'] = pd.DataFrame(
            index=result_d['AMD_world_over_AMD_i'].index,
            columns=result_d['AMD_world_over_AMD_i'].columns,
            data=cf_arr
        )
        return result_d

    @staticmethod
    def calc_aware_cf_arrays(AMD_world_over_AMD_i_arr, lower_bound, upper_bound):
        """ Calculate characterization factors from AMD ratio array (values in DataFrame)"""
        infs = np.isinf(AMD_world_over_AMD_i_arr)
        #AMD_world_over_AMD_i_arr[infs] = 100
        under_zero = (AMD_world_over_AMD_i_arr < 0)
        over_upper_bound = AMD_world_over_AMD_i_arr > upper_bound
        under_lower_bound = (AMD_world_over_AMD_i_arr < lower_bound) & ~infs & ~under_zero
        cfs = AMD_world_over_AMD_i_arr
        cfs[np.isinf(AMD_world_over_AMD_i_arr)] = upper_bound
        cfs[under_zero] = upper_bound
        cfs[under_lower_bound] = lower_bound
        cfs[over_upper_bound] = upper_bound
        return cfs

    @staticmethod
    def calc_aware_annual_unknown(result_d):
        """ Calculate the annual weighted average CFs per basin for situations
        where the type of water withdrawn is not known"""
        cfs_monthly = result_d['cfs_monthly']
        HWC = result_d['HWC']
        has_HWC = HWC.sum(axis=1) != 0  # Because regions without HWC are treated differently
        average_cfs_unknown = pd.Series(index=HWC.index)  # Blank result Series
        # HWC-weighted average if there is HWC
        average_cfs_unknown.loc[has_HWC] = cfs_monthly[has_HWC].multiply(HWC[has_HWC]).sum(axis=1) / HWC[has_HWC].sum(
            axis=1)
        # Simple average if there is no HWC
        average_cfs_unknown.loc[-has_HWC] = cfs_monthly[-has_HWC].mean(axis=1)
        result_d['cfs_average_unknown'] = average_cfs_unknown
        return result_d

    @staticmethod
    def calc_aware_annual_agri(data_d, result_d):
        """ Calculate the annual weighted average CFs per basin for situations
        where water is withdrawn for agriculture"""

        cfs_monthly = result_d['cfs_monthly']
        irrigation = data_d['irrigation']

        has_irrigation = irrigation.sum(axis=1) != 0  # Because regions without irrigation are treated differently
        average_cfs_agri = pd.Series(index=irrigation.index)  # Blank result Series
        # Irrigation-weighted average if there is irrigation
        average_cfs_agri.loc[has_irrigation] = cfs_monthly[has_irrigation].multiply(irrigation[has_irrigation]).sum(
            axis=1) / irrigation[has_irrigation].sum(axis=1)
        # Simple average if there is no HWC
        average_cfs_agri.loc[-has_irrigation] = cfs_monthly[-has_irrigation].mean(axis=1)
        result_d['cfs_average_agri'] = average_cfs_agri
        return result_d

    @staticmethod
    def calc_aware_annual_non_agri(result_d):
        """ Calculate the annual weighted average CFs per basin for situations
        where water is known not to be withdrawn for agriculture"""
        result_d['cfs_average_non_agri'] = result_d['cfs_monthly'].mean(axis=1)
        return result_d
