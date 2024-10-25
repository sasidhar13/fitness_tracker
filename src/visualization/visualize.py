import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"]==1]

plt.plot(set_df['acc_y'])

plt.plot(set_df['acc_y'].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
all_labels = df['label'].unique()

for label in all_labels:
    subset = df[df['label']==label]
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()

for label in all_labels:
    subset = df[df['label']==label]
    fig, ax = plt.subplots()
    plt.plot(subset[:200]['acc_y'].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
    
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"]=[20,9]
mpl.rcParams["figure.dpi"]=100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
squat_a_df = df.query("label=='squat'").query("participant=='A'").reset_index()
fig,ax = plt.subplots()
squat_a_df.groupby(['category'])['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('accelration on y axis')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
squat_df = df.query("label=='dead'").sort_values("participant").reset_index()
fig,ax = plt.subplots()
squat_df.groupby(['participant'])['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('accelration on y axis')
plt.legend()
plt.show()
# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label_test = 'dead'
participant_test = 'E'
all_axis_df = df.query(f"label=='{label_test}'").query(f"participant=='{participant_test}'").reset_index()
fig,ax = plt.subplots()
all_axis_df[['acc_x','acc_y','acc_z']].plot()
plt.xlabel('samples')
plt.ylabel('accelration on y axis')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
label_all = df['label'].unique()
participant_all = df['participant'].unique()

for label_test in label_all:
    for participant_test in participant_all:
        all_axis_df = df.query(f"label=='{label_test}'").query(f"participant=='{participant_test}'").reset_index()
        
        
        #condition to filter out participants that didnt attempt the exercise/label
        if len(all_axis_df)>0:
            fig,ax = plt.subplots(nrows=2,sharex=True)
            all_axis_df[['acc_x','acc_y','acc_z']].plot(ax=ax[0])
            all_axis_df[['gyr_x','gyr_y','gyr_z']].plot(ax=ax[1])
            #ax[0].set_xlabel(f'{label_test} samples')
            ax[1].set_xlabel(f'{label_test} samples')
            ax[0].set_ylabel(' accelration all axis')
            ax[1].set_ylabel(' gyroscope all axis')

            ax[0].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            plt.savefig(f"../../reports/figures/{label_test}_{participant_test}")
            plt.show()
# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------