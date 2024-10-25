import sys
sys.path.append('../../src/features')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/01_data_processed.pkl')
df= df[df['label']!='rest']

acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2

df['acc_r']=np.sqrt(acc_r)
df['gyr_r']=np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
df['label'].unique()

bench_df= df[df['label']=='bench']
ohp_df= df[df['label']=='ohp']
squat_df= df[df['label']=='squat']
dead_df= df[df['label']=='dead']
row_df= df[df['label']=='row']

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = bench_df
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_x'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_y'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_z'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_r'].plot()

plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_x'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_y'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_z'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_r'].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000/200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df['set']==bench_df['set'].unique()[0]]
squat_set = squat_df[squat_df['set']==squat_df['set'].unique()[2]]
row_set = row_df[row_df['set']==row_df['set'].unique()[0]]
ohp_set = ohp_df[ohp_df['set']==ohp_df['set'].unique()[0]]
dead_set = dead_df[dead_df['set']==dead_df['set'].unique()[10]]
dead_set['category'].unique()
bench_set['acc_r'].plot()

column = 'acc_y'
LowPass.low_pass_filter(bench_set,col =column,sampling_frequency=fs,cutoff_frequency=0.45, order=10)[column+"_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
column = 'acc_y'
def rep_counter(data_set,column='acc_r',cutoff=0.4,fs=fs,order=10):
    data_array = LowPass.low_pass_filter(data_set,col =column,sampling_frequency=fs,cutoff_frequency=cutoff, order=order)
    indexes = argrelextrema(data_array[column+'_lowpass'].values,np.greater)
    peaks = data_array.iloc[indexes]
    
    #fig,ax = plt.subplots()
    #plt.plot(data_set[f'{column}_lowpass'])
    #plt.plot(peaks[f'{column}_lowpass'])
    #ax.set_ylabel(f'{column}_lowpass')
    exercise = data_set['label'].iloc[0].title()
    category = data_set['category'].iloc[0].title()
    #plt.title(f'{category} {exercise} : {len(peaks)} Reps')
    #plt.show()
    return len(peaks)

rep_counter(squat_set,cutoff=0.45)
rep_counter(ohp_set,cutoff=0.45)
rep_counter(dead_set,cutoff=0.4)
rep_counter(row_set,cutoff=0.69,column='acc_r')
rep_counter(bench_set,cutoff=0.45)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df['reps'] = df['category'].apply(lambda x:5 if x=='heavy' else 10)
rep_df = df.groupby(['label','category','set'])['reps'].max().reset_index()
rep_df['reps_pred']=0

for s in df['set'].unique():
    subset = df[df['set']==s]
    column = 'acc_r'
    cutoff=0.4
    
    if subset['label'].iloc[0]=='squat':
        cutoff = 0.35
    if subset['label'].iloc[0]=='bench':
        cutoff = 0.45
    if subset['label'].iloc[0]=='row':
        cutoff = 0.69
    if subset['label'].iloc[0]=='dead':
        cutoff = 0.4
    if subset['label'].iloc[0]=='ohp':
        cutoff = 0.35
    
    reps = rep_counter(subset,cutoff=cutoff,column=column)
    rep_df.loc[rep_df['set']==s,'reps_pred']=reps

rep_df
        

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df['reps'], rep_df['reps_pred']).round(2)
error
rep_df.groupby(['label','category'])['reps','reps_pred'].mean().plot.bar()
