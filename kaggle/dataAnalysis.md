# Notes and data analysis
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data explanation
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# /kaggle/input/eeg-muse2-motor-imagery-brain-electrical-activity/museMonitor_2024-06-05--17-23-37_9208698270315378717.csv
# /kaggle/input/eeg-muse2-motor-imagery-brain-electrical-activity/museMonitor_2024-06-22--18-41-35_2861114036213037750.csv
# ...
# /kaggle/input/eeg-muse2-motor-imagery-brain-electrical-activity/museMonitor_2024-06-21--00-02-00_3021179978499686494.csv

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# import warnings
# warnings.filterwarnings('ignore')

# Define the directory path
directory_path = '/kaggle/input/eeg-muse2-motor-imagery-brain-electrical-activity/'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Read and concatenate all CSV files
combined_df = pd.concat([
    pd.read_csv(os.path.join(directory_path, file), dtype={"Elements": str})
    for file in csv_files
], ignore_index=True)
print("Read data success")
## Add dtype={"Elements": str} to handle mixed datatypes: str and NaN.
## This keeps all non-missing values as strings and leaves missing ones as NaN
## No need to ignore warnings


# Understanding the dataset and "Elements" column statistics

# Overview of the dataset
# combined_df.head()
#                  TimeStamp  Delta_TP9  Delta_AF7  Delta_AF8  Delta_TP10  \
# 0  2024-06-05 17:23:37.961   0.329342  -0.259216  -0.090011    0.551984   
# 1  2024-06-05 17:23:37.967   0.329342  -0.259216  -0.090011    0.551984   
# 2  2024-06-05 17:23:37.968   0.329342  -0.259216  -0.090011    0.551984   
# 3  2024-06-05 17:23:37.969   0.329342  -0.259216  -0.090011    0.551984   
# 4  2024-06-05 17:23:37.969   0.329342  -0.259216  -0.090011    0.551984   

#    Theta_TP9  Theta_AF7  Theta_AF8  Theta_TP10  Alpha_TP9  ...    Gyro_X  \
# 0   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 1   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 2   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 3   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 4   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   

#     Gyro_Y    Gyro_Z  HeadBandOn  HSI_TP9  HSI_AF7  HSI_AF8  HSI_TP10  \
# 0  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 1  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 2  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 3  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 4  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   

#    Battery  Elements  
# 0     80.0       NaN  
# 1     80.0       NaN  
# 2     80.0       NaN  
# 3     80.0       NaN  
# 4     80.0       NaN 

# Columns overview
# print(combined_df.columns)
# Index(['TimeStamp', 'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
#        'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9',
#        'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 'Beta_TP9', 'Beta_AF7',
#        'Beta_AF8', 'Beta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8',
#        'Gamma_TP10', 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT',
#        'Mellow', 'Concentration', 'Accelerometer_X', 'Accelerometer_Y',
#        'Accelerometer_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'HeadBandOn',
#        'HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10', 'Battery', 'Elements'],
#       dtype='object')


# Elements column statistics
# print(combined_df[combined_df["Elements"].notna()]["Elements"].value_counts())
# Elements
# /muse/elements/blink         3513
# /Marker/3                     112
# /Marker/2                      70
# /Marker/1                      67
# /muse/elements/jaw_clench      34
# /Marker/4                       1
# Name: count, dtype: int64

# Missing values summary:
# - TimeStamp is always non-NaN
# e.g. 
# missing_percentage = combined_df.isnull().sum() / len(combined_df) * 100
# print(missing_percentage)
# TimeStamp           0.000000
# Delta_TP9           0.054650
# Delta_AF7           0.054650
# Delta_AF8           0.054650
# Delta_TP10          0.054650
# Theta_TP9           0.054650
# Theta_AF7           0.054650
# Theta_AF8           0.054650
# Theta_TP10          0.054650
# Alpha_TP9           0.054650
# Alpha_AF7           0.054650
# Alpha_AF8           0.054650
# Alpha_TP10          0.054650
# Beta_TP9            0.054650
# Beta_AF7            0.054650
# Beta_AF8            0.054650
# Beta_TP10           0.054650
# Gamma_TP9           0.054650
# Gamma_AF7           0.054650
# Gamma_AF8           0.054650
# Gamma_TP10          0.054650
# RAW_TP9             6.468099
# RAW_AF7             8.872474
# RAW_AF8             3.711213
# RAW_TP10            6.853946
# AUX_RIGHT          10.544188
# Mellow              0.054650
# Concentration       0.054650
# Accelerometer_X     0.054650
# Accelerometer_Y     0.054650
# Accelerometer_Z     0.054650
# Gyro_X              0.054650
# Gyro_Y              0.054650
# Gyro_Z              0.054650
# HeadBandOn          0.054650
# HSI_TP9             0.054650
# HSI_AF7             0.054650
# HSI_AF8             0.054650
# HSI_TP10            0.054650
# Battery             0.054650
# Elements           99.945350
# dtype: float64



# Missing value caused by adding a marker
# - Some rows have all NaN values except in the "TimeStamp" and "Elements" column (missing market data).
# e.g.
# print(combined_df[combined_df["Elements"].notna()].head(1))
#                  TimeStamp  Delta_TP9  Delta_AF7  Delta_AF8  Delta_TP10  \
# 21768  2024-06-05 17:25:02        NaN        NaN        NaN         NaN   

#        Theta_TP9  Theta_AF7  Theta_AF8  Theta_TP10  Alpha_TP9  ...  Gyro_X  \
# 21768        NaN        NaN        NaN         NaN        NaN  ...     NaN   

#        Gyro_Y  Gyro_Z  HeadBandOn  HSI_TP9  HSI_AF7  HSI_AF8  HSI_TP10  \
# 21768     NaN     NaN         NaN      NaN      NaN      NaN       NaN   

#        Battery   Elements  
# 21768      NaN  /Marker/1 

# All EEG power values in the delta, theta, lpha, beta, and gamma bands are present (not NaN)
# No need for further preprossing for current stage


#QUESTIONS:
# I don't understand why there are duplicated TimeStamps
#                  TimeStamp  Delta_TP9  Delta_AF7  Delta_AF8  Delta_TP10  \
# 0  2024-06-05 17:23:37.961   0.329342  -0.259216  -0.090011    0.551984   
# 1  2024-06-05 17:23:37.967   0.329342  -0.259216  -0.090011    0.551984   
# 2  2024-06-05 17:23:37.968   0.329342  -0.259216  -0.090011    0.551984   
# 3  2024-06-05 17:23:37.969   0.329342  -0.259216  -0.090011    0.551984   
# 4  2024-06-05 17:23:37.969   0.329342  -0.259216  -0.090011    0.551984   

#    Theta_TP9  Theta_AF7  Theta_AF8  Theta_TP10  Alpha_TP9  ...    Gyro_X  \
# 0   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 1   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 2   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 3   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   
# 4   0.427443  -0.264384  -0.045775    0.591313   1.004643  ...  0.276642   

#     Gyro_Y    Gyro_Z  HeadBandOn  HSI_TP9  HSI_AF7  HSI_AF8  HSI_TP10  \
# 0  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 1  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 2  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 3  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   
# 4  0.18692 -0.986938         1.0      1.0      1.0      1.0       1.0   

#    Battery  Elements  
# 0     80.0       NaN  
# 1     80.0       NaN  
# 2     80.0       NaN  
# 3     80.0       NaN  
# 4     80.0       NaN  

# Ensure no NaNs or Infs are present
# combined_df = combined_df.replace([float('inf'), float('-inf')], pd.NA)
# combined_df = combined_df.dropna()

5/23/24: Features:
# 1. Extract pipeline from current data_segementation.py
# 2. Add CLI to eliminate call from main
# 3. Add tox testing 