import glob
import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sktime.utils import load_data
from torch.utils.data import Dataset
from tqdm import tqdm

from options import Options
from utils import seed_everything, setup

logger = logging.getLogger("__main__")

regression_datasets = [
    "AustraliaRainfall",
    "HouseholdPowerConsumption1",
    "HouseholdPowerConsumption2",
    "BeijingPM25Quality",
    "BeijingPM10Quality",
    "Covid3Month",
    "LiveFuelMoistureContent",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "AppliancesEnergy",
    "BenzeneConcentration",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "BIDMC32RR",
    "BIDMC32HR",
    "BIDMC32SpO2",
    "IEEEPPG",
    "PPGDalia",
]


def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


# The following code is adapted from the python package sktime to read .ts file.
class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """

    pass


def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Loads data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
    """

    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_target_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_float = None
    previous_timestamp_was_int = None
    previous_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    target_labels = False

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            # print(".", end='')
            # Strip white space from start/end of line and change to lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "problemname tag requires an associated value"
                        )

                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "timestamps tag requires an associated Boolean value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise TsFileParseException("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise TsFileParseException(
                            "univariate tag requires an associated Boolean value"
                        )
                    # elif tokens[1] == "true":
                    #     univariate = True
                    # elif tokens[1] == "false":
                    #     univariate = False
                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "classlabel tag requires an associated Boolean value"
                        )

                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise TsFileParseException(
                            "if the classlabel tag is true then class values must be supplied"
                        )

                    has_class_labels_tag = True
                    # class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "targetlabel tag requires an associated Boolean value"
                        )

                    if tokens[1] == "true":
                        target_labels = True
                    elif tokens[1] == "false":
                        target_labels = False
                    else:
                        raise TsFileParseException("invalid targetLabel value")

                    has_target_labels_tag = True
                    class_val_list = []
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise TsFileParseException(
                            "data tag should not have an associated value"
                        )

                    if data_started and not metadata_started:
                        raise TsFileParseException("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    incomplete_regression_meta_data = (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_target_labels_tag
                        or not has_data_tag
                    )
                    incomplete_classification_meta_data = (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    )
                    if (
                        incomplete_regression_meta_data
                        and incomplete_classification_meta_data
                    ):
                        raise TsFileParseException(
                            "a full set of metadata has not been provided before the data"
                        )

                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False

                        timestamps_for_dimension = []
                        values_for_dimension = []

                        this_line_num_dimensions = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (
                                        this_line_num_dimensions + 1
                                    ):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(
                                        pd.Series()
                                    )
                                    this_line_num_dimensions += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamps_for_dimension = []
                                    values_for_dimension = []

                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and target_labels:
                                        class_val = line[char_num:].strip()

                                        # if class_val not in class_val_list:
                                        #     raise TsFileParseException(
                                        #         "the class value '" + class_val + "' on line " + str(
                                        #             line_num + 1) + " is not valid")

                                        class_val_list.append(float(class_val))
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within the next tuple

                                        if line[char_num] != "(" and not target_labels:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does not start with a '('"
                                            )

                                        char_num += 1
                                        tuple_data = ""

                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does not end with a ')'"
                                            )

                                        # Read in any spaces immediately after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                        last_comma_index = tuple_data.rfind(",")

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has no comma inside of it"
                                            )

                                        try:
                                            value = tuple_data[(last_comma_index + 1) :]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does not have a valid numeric value"
                                            )

                                        # Check the type of timestamp that we have

                                        timestamp = tuple_data[0:last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = float(timestamp)
                                                timestamp_is_float = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_float = False

                                        if (
                                            not timestamp_is_int
                                            and not timestamp_is_float
                                        ):
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                            and not timestamp_is_float
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )

                                        if (
                                            previous_timestamp_was_float is not None
                                            and previous_timestamp_was_float
                                            and not timestamp_is_float
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the timestamp format is inconsistent"
                                            )

                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the timestamp format is inconsistent"
                                            )

                                        if (
                                            previous_timestamp_was_timestamp is not None
                                            and previous_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dimensions + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the timestamp format is inconsistent"
                                            )

                                        # Store the values

                                        timestamps_for_dimension += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then we store the type of timestamp we had

                                        if (
                                            previous_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            previous_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                            previous_timestamp_was_float = False

                                        if (
                                            previous_timestamp_was_float is None
                                            and timestamp_is_float
                                        ):
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = True

                                        # See if we should add the data for this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dimensions + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamps_for_dimension = (
                                                    pd.DatetimeIndex(
                                                        timestamps_for_dimension
                                                    )
                                                )

                                            instance_list[
                                                this_line_num_dimensions
                                            ].append(
                                                pd.Series(
                                                    index=timestamps_for_dimension,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dimensions += 1

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension "
                                    + str(this_line_num_dimensions + 1)
                                    + " on line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that is not followed by another tuple"
                                )

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension "
                                    + str(this_line_num_dimensions + 1)
                                    + " on line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while it should list a class value"
                                )

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dimensions

                                if num_dimensions != this_line_num_dimensions:
                                    raise TsFileParseException(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the same number of dimensions as the previous line of data"
                                    )

                        # Check that we are not expecting some more data, and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dimensions + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is not followed by another tuple"
                            )

                        elif has_another_dimension and target_labels:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dimensions + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it should list a class value"
                            )

                        elif has_another_dimension and not target_labels:
                            if len(instance_list) < (this_line_num_dimensions + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dimensions].append(pd.Series())
                            this_line_num_dimensions += 1
                            num_dimensions = this_line_num_dimensions

                        # If this is the 1st line of data we have seen then note the dimensions

                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dimensions
                        ):
                            raise TsFileParseException(
                                "line "
                                + str(line_num + 1)
                                + " does not have the same number of dimensions as the previous line of data"
                            )

                        # Check if we should have class values, and if so that they are contained in those listed in the metadata

                        if target_labels and len(class_val_list) == 0:
                            raise TsFileParseException(
                                "the cases have no associated class values"
                            )
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if target_labels:
                                num_dimensions -= 1

                            for dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False

                        # See how many dimensions that the case whose data in represented in this line has
                        this_line_num_dimensions = len(dimensions)

                        if target_labels:
                            this_line_num_dimensions -= 1

                        # All dimensions should be included for all series, even if they are empty
                        if this_line_num_dimensions != num_dimensions:
                            raise TsFileParseException(
                                "inconsistent number of dimensions. Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dimensions)
                            )

                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series())

                        if target_labels:
                            class_val_list.append(
                                float(dimensions[num_dimensions].strip())
                            )

            line_num += 1

    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        complete_regression_meta_data = (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_target_labels_tag
            and has_data_tag
        )
        complete_classification_meta_data = (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        )

        if (
            metadata_started
            and not complete_regression_meta_data
            and not complete_classification_meta_data
        ):
            raise TsFileParseException("metadata incomplete")
        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")
        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if target_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise TsFileParseException("empty file")


def process_data(X, min_len, normalise=None):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param min_len:
    :param normalise:
    :return:
    """
    tmp = []
    for i in tqdm(range(len(X))):
        _x = X.iloc[i, :].copy(deep=True)

        # 1. find the maximum length of each dimension
        all_len = [len(y) for y in _x]
        max_len = max(all_len)

        # 2. adjust the length of each dimension
        _y = []
        for y in _x:
            # 2.1 fill missing values
            if y.isnull().any():
                y = y.interpolate(method="linear", limit_direction="both")

            # 2.2. if length of each dimension is different, uniformly scale the shorter ones to the max length
            if len(y) < max_len:
                y = uniform_scaling(y, max_len)
            _y.append(y)
        _y = np.array(np.transpose(_y))

        # 3. adjust the length of the series, chop of the longer series
        _y = _y[:min_len, :]

        # 4. normalise the series
        if normalise == "standard":
            scaler = StandardScaler().fit(_y)
            _y = scaler.transform(_y)
        if normalise == "minmax":
            scaler = MinMaxScaler().fit(_y)
            _y = scaler.transform(_y)

        tmp.append(_y)
    X = np.array(tmp)
    return X


def split_dataset(
    data_indices,
    validation_method,
    n_splits,
    validation_ratio,
    test_set_ratio=0,
    test_indices=None,
    random_seed=1337,
    labels=None,
):
    """
    Splits dataset (i.e. the global datasets indices) into a test set and a training/validation set.
    The training/validation set is used to produce `n_splits` different configurations/splits of indices.

    Returns:
        test_indices: numpy array containing the global datasets indices corresponding to the test set
            (empty if test_set_ratio is 0 or None)
        train_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's training set
        val_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's validation set
    """

    # Set aside test set, if explicitly defined
    if test_indices is not None:
        data_indices = np.array(
            [ind for ind in data_indices if ind not in set(test_indices)]
        )  # to keep initial order

    datasplitter = DataSplitter.factory(
        validation_method, data_indices, labels
    )  # DataSplitter object

    # Set aside a random partition of all data as a test set
    if test_indices is None:
        if test_set_ratio:  # only if test set not explicitly defined
            datasplitter.split_testset(
                test_ratio=test_set_ratio, random_state=random_seed
            )
            test_indices = datasplitter.test_indices
        else:
            test_indices = []
    # Split train / validation sets
    datasplitter.split_validation(n_splits, validation_ratio, random_state=random_seed)

    return datasplitter.train_indices, datasplitter.val_indices, test_indices


class DataSplitter(object):
    """Factory class, constructing subclasses based on feature type"""

    def __init__(self, data_indices, data_labels=None):
        """data_indices = train_val_indices | test_indices"""

        self.data_indices = data_indices  # global datasets indices
        self.data_labels = data_labels  # global raw datasets labels
        self.train_val_indices = np.copy(
            self.data_indices
        )  # global non-test indices (training and validation)
        self.test_indices = []  # global test indices

        if data_labels is not None:
            self.train_val_labels = np.copy(
                self.data_labels
            )  # global non-test labels (includes training and validation)
            self.test_labels = []  # global test labels # TODO: maybe not needed

    @staticmethod
    def factory(split_type, *args, **kwargs):
        if split_type == "StratifiedShuffleSplit":
            return StratifiedShuffleSplitter(*args, **kwargs)
        if split_type == "ShuffleSplit":
            return ShuffleSplitter(*args, **kwargs)
        else:
            raise ValueError("DataSplitter for '{}' does not exist".format(split_type))

    def split_testset(self, test_ratio, random_state=1337):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        raise NotImplementedError("Please override function in child class")

    def split_validation(self):
        """
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        raise NotImplementedError("Please override function in child class")


class StratifiedShuffleSplitter(DataSplitter):
    """
    Returns randomized shuffled folds, which preserve the class proportions of samples in each fold. Differs from k-fold
    in that not all samples are evaluated, and samples may be shared across validation sets,
    which becomes more probable proportionally to validation_ratio/n_splits.
    """

    def split_testset(self, test_ratio, random_state=1337):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        splitter = model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        # get local indices, i.e. indices in [0, len(data_labels))
        train_val_indices, test_indices = next(
            splitter.split(X=np.zeros(len(self.data_indices)), y=self.data_labels)
        )
        # return global datasets indices and labels
        self.train_val_indices, self.train_val_labels = (
            self.data_indices[train_val_indices],
            self.data_labels[train_val_indices],
        )
        self.test_indices, self.test_labels = (
            self.data_indices[test_indices],
            self.data_labels[test_indices],
        )

        return

    def split_validation(self, n_splits, validation_ratio, random_state=1337):
        """
        Input:
            n_splits: number of different, randomized and independent from one-another folds
            validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        splitter = model_selection.StratifiedShuffleSplit(
            n_splits=n_splits, test_size=validation_ratio, random_state=random_state
        )
        # get local indices, i.e. indices in [0, len(train_val_labels)), per fold
        train_indices, val_indices = zip(
            *splitter.split(
                X=np.zeros(len(self.train_val_labels)), y=self.train_val_labels
            )
        )
        # return global datasets indices per fold
        self.train_indices = [
            self.train_val_indices[fold_indices] for fold_indices in train_indices
        ]
        self.val_indices = [
            self.train_val_indices[fold_indices] for fold_indices in val_indices
        ]

        return


class ShuffleSplitter(DataSplitter):
    """
    Returns randomized shuffled folds without requiring or taking into account the sample labels. Differs from k-fold
    in that not all samples are evaluated, and samples may be shared across validation sets,
    which becomes more probable proportionally to validation_ratio/n_splits.
    """

    def split_testset(self, test_ratio, random_state=1337):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        splitter = model_selection.ShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        # get local indices, i.e. indices in [0, len(data_indices))
        train_val_indices, test_indices = next(
            splitter.split(X=np.zeros(len(self.data_indices)))
        )
        # return global datasets indices and labels
        self.train_val_indices = self.data_indices[train_val_indices]
        self.test_indices = self.data_indices[test_indices]
        if self.data_labels is not None:
            self.train_val_labels = self.data_labels[train_val_indices]
            self.test_labels = self.data_labels[test_indices]

        return

    def split_validation(self, n_splits, validation_ratio, random_state=1337):
        """
        Input:
            n_splits: number of different, randomized and independent from one-another folds
            validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        splitter = model_selection.ShuffleSplit(
            n_splits=n_splits, test_size=validation_ratio, random_state=random_state
        )
        # get local indices, i.e. indices in [0, len(train_val_labels)), per fold
        train_indices, val_indices = zip(
            *splitter.split(X=np.zeros(len(self.train_val_indices)))
        )
        # return global datasets indices per fold
        self.train_indices = [
            self.train_val_indices[fold_indices] for fold_indices in train_indices
        ]
        self.val_indices = [
            self.train_val_indices[fold_indices] for fold_indices in val_indices
        ]

        return


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class TSRegressionArchive:
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(
        self,
        root_dir,
        file_list=None,
        pattern=None,
        n_proc=1,
        limit_size=None,
        config=None,
    ):

        # self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(
            root_dir, file_list=file_list, pattern=pattern
        )
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, "*"))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".ts")
        ]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0]
        )  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config["task"] == "regression":
            df, labels = load_from_tsfile_to_dataframe(
                filepath, return_separate_X_and_y=True, replace_missing_vals_with="NaN"
            )
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config["task"] == "classification":
            df, labels = load_data.load_from_tsfile_to_dataframe(
                filepath, return_separate_X_and_y=True, replace_missing_vals_with="NaN"
            )
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(
                labels.cat.codes, dtype=np.int8
            )  # int8-32 gives an error when using nn.CrossEntropyLoss
        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(
                    filepath,
                    return_separate_X_and_y=True,
                    replace_missing_vals_with="NaN",
                )
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = load_from_tsfile_to_dataframe(
                    filepath,
                    return_separate_X_and_y=True,
                    replace_missing_vals_with="NaN",
                )
            labels_df = None

        lengths = df.applymap(
            lambda x: len(x)
        ).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if (
            np.sum(horiz_diffs) > 0
        ):  # if any row (sample) has varying length across dimensions
            logger.warning(
                "Not all time series dimensions have same length - will attempt to fix by subsampling first dimension..."
            )
            df = df.applymap(
                subsample
            )  # TODO: this addresses a very specific case (PPGDalia)

        if self.config["subsample_factor"]:
            df = df.applymap(
                lambda x: subsample(x, limit=0, factor=self.config["subsample_factor"])
            )

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if (
            np.sum(vert_diffs) > 0
        ):  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning(
                "Not all samples have same length: maximum length set to {}".format(
                    self.max_seq_len
                )
            )
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(lengths[row, 0] * [row]))
                for row in range(df.shape[0])
            ),
            axis=0,
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class UEADataset(Dataset):
    def __init__(self, data, indices):
        super(UEADataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = (
            indices  # list of data IDs, but also mapping between integer index and ID
        )
        self.feature_df = self.data.feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        return X, y.squeeze()

    def __len__(self):
        return len(self.IDs)


def load_uea_dataset(config, debug):
    logger.info("Loading and preprocessing data ...")

    my_data = TSRegressionArchive(
        config["data_dir"],
        pattern=config["pattern"],
        limit_size=config["limit_size"],
        config=config,
    )
    if config["task"] == "classification":
        validation_method = "StratifiedShuffleSplit"
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = "ShuffleSplit"
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config[
        "test_pattern"
    ]:  # used if test data come from different files / file patterns
        test_data = TSRegressionArchive(
            config["data_dir"], pattern=config["test_pattern"], n_proc=-1, config=config
        )
        test_indices = test_data.all_IDs
    if config[
        "test_from"
    ]:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(
            set([line.rstrip() for line in open(config["test_from"]).readlines()])
        )
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info(
            "Loaded {} test IDs from file: '{}'".format(
                len(test_indices), config["test_from"]
            )
        )
    if config[
        "val_pattern"
    ]:  # used if val data come from different files / file patterns
        val_data = TSRegressionArchive(
            config["data_dir"], pattern=config["val_pattern"], n_proc=-1, config=config
        )
        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    if config["val_ratio"] > 0:
        train_indices, val_indices, test_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_method=validation_method,
            n_splits=1,
            validation_ratio=config["val_ratio"],
            test_set_ratio=config[
                "test_ratio"
            ],  # used only if test_indices not explicitly specified
            test_indices=test_indices,
            random_seed=1337,
            labels=labels,
        )
        train_indices = train_indices[
            0
        ]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[
            0
        ]  # `split_dataset` returns a list of indices *per fold/split*
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config["output_dir"], "data_indices.json"), "w") as f:
        try:
            json.dump(
                {
                    "train_indices": list(map(int, train_indices)),
                    "val_indices": list(map(int, val_indices)),
                    "test_indices": list(map(int, test_indices)),
                },
                f,
                indent=4,
            )
        except ValueError:  # in case indices are non-integers
            json.dump(
                {
                    "train_indices": list(train_indices),
                    "val_indices": list(val_indices),
                    "test_indices": list(test_indices),
                },
                f,
                indent=4,
            )

    # Pre-process features
    normalizer = None
    if config["norm_from"]:
        with open(config["norm_from"], "rb") as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config["normalization"] is not None:
        normalizer = Normalizer(config["normalization"])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(
            my_data.feature_df.loc[train_indices]
        )
        if not config["normalization"].startswith("per_sample"):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(
                os.path.join(config["output_dir"], "normalization.pickle"), "wb"
            ) as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(
                val_data.feature_df.loc[val_indices]
            )
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(
                test_data.feature_df.loc[test_indices]
            )

    if debug:
        my_data = my_data
        val_data = my_data
        test_data = my_data
        ind = np.random.choice(len(train_indices), size=100, replace=False)
        train_indices = train_indices[ind]
        val_indices = train_indices
        test_indices = train_indices

    train_dataset = UEADataset(my_data, train_indices)
    val_dataset = UEADataset(val_data, val_indices)
    test_dataset = UEADataset(test_data, test_indices)

    config.feat_dim = my_data.feature_df.shape[1]
    config.max_seq_len = my_data.max_seq_len
    config.num_classes = (
        len(my_data.class_names)
        if config.task == "classification"
        else my_data.labels_df.shape[1]
    )

    return train_dataset, val_dataset, test_dataset, config


if __name__ == "__main__":
    args = Options().parse()
    config = setup(args)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))
    load_uea_dataset(config.data)
