import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as pyplot


def readInWL(filenameIn):
    """
    Function which reads in tidal station data, parses it,
    and returns a Pandas DataFrame

    Args:
        filenameIn - the filename (with full path) of the file to be read in
    Returns:
        dataIn - a pandas dataframe indexed by time, with coerced errors, with a
            subselection of tidal station data columns
    """

    dataIn = pd.read_csv(filenameIn, index_col=0, parse_dates=False,
                         usecols=['DATE_TIME', 'A1_WL_VALUE_MSL', 'A1_WL_SIGMA',
                                  'B1_WL_VALUE_MSL', 'VER_WL_VALUE_MSL', 'PRED_WL_VALUE_MSL']
                         )
    dataIn.columns = ['primary', 'sigma', 'backup', 'verified', 'prediction']
    dataIn.index.name = 'time'
    dataIn = dataIn.apply(pd.to_numeric, errors='coerce')
    dataIn.head()

    return dataIn


def initial_data_prep(dataIn, timeStart='2007-01-01 00:00:00', timeEnd='2017-12-31 23:54:00'):
    """
    Function performs initial data preparation
    such as replacing -999999 with Nans
    and subsetting the data to a specific date range.

    Args:
        dataIn - a pandas dataframe of the tidal station
        timeStart - the start time to slice the pandas dataframe
        timeEnd - the ending time to slice the pandas dataframe
    Raises:
        TypeError
        ValueError
    Returns:
        data_indexed - a pandas dataframe that has NaNs removed
            and subsampled by time b
    """

    # Check that the data frame in is of the appropriate type
    if type(dataIn) is not pandas.core.frame.DataFrame:
        raise TypeError(f'The input file is not a DataFrame.')

    if pd.to_datetime(timeStart) > pd.to_datetime(timeEnd):
        raise ValueError(f'The start time {timeStart} is after the ending time {timeEnd}')

    # Replace -99999 with NaNs
    dataIn.mask((dataIn <= -1000), inplace=True)

    # drop any lines with NaNs
    dataUnNaN = dataIn.dropna()

    # decide what time window we want to extract and clean
    # only data after 2007 are any good
    dataSub = dataUnNaN[timeStart:timeEnd]

    # Change the index
    data_indexed = dataSub.reset_index()

    return data_indexed


def remove_extra_datapoints(data_indexed, suppress_fig=False):
    """
    Function which removes extra datapoints
    which occur between the expected datapoints.
    The timedelta between real datapoints should
    be 6 minutes.

    Args:
        data_indexed - a dataframe of tidal station data
            which has been subsampled and NaNs removed
    Returns:
        cleaned - a dataframe of tidal station data where
            extra datapoints have been removed
        fig - a 2 x 1 subplot of the cleaned dataframe
    """

    # Pull out the time column
    time_asdatetime = data_indexed.time
    # convert to datetime
    time_asdt = pd.to_datetime(time_asdatetime)
    # pull out the minutes from the datetime
    minutes = time_asdt.dt.minute

    # all minutes must be 0, 6, 12, ... 54
    goodminutes = range(0, 60, 6)

    # find and remove the bad minutes
    goodmask = np.isin(minutes, goodminutes)
    cleaned = data_indexed.loc[goodmask]

    # lets make sure we removed all the bad records
    # by plotting the time differences of the cleaned data
    cleantime = cleaned.time
    cleandt = pd.to_datetime(cleantime)
    cleandiff = np.diff(cleandt)

    # Function to print out the figure
    def print_cleaned_dataframe(difftime, cleandiff):
        fig = plt.figure()
        # Plot the first subplot
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("Original Time Differences")
        ax1.plot(difftime)

        # Plot the second subplot
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("Cleaned Time Differences")
        ax2.plot(cleandiff)

    if not suppress_fig:
        print_cleaned_dataframe(difftime, cleandiff)

    return cleaned


def findTarget(df):
    """
    Outputs a pandas dataframe with single column of boolean variables
    indicating if data is flagged or not. Flagged data indicates data passed
    NOAA tide QC. Uses 'verified' and 'primary' columns

    Args:
        df - a pandas dataframe of the cleaned tidal data
    Returns:
        target - a mask of the tidal data dataframe which marks the NOAA
            QARTOD QC results

    """
    goodPts = np.array(1*(df.loc[:, 'verified'] == df.loc[:, 'primary']))
    target = pd.DataFrame(goodPts)
    target.columns = ['goodPts']
    target.head()

    return target
