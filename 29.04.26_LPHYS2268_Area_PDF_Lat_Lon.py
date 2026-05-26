import numpy as np
import matplotlib.pyplot as plt
import random
import xarray as xr
import imageio
import os

import seaborn as sns
from scipy.stats import kde
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, label

#----------------Variiables------------------

Year_range = [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
#Year_range = [2014,2015,2016,2017,2018]
#Year_range = [2014]
start_dates = [122,153,183,213]
#start_dates = [153]

number_of_beads = 100
sea_ice_min = 0.15

#Relic and could be done without but left in to reduce workload of ammending code
number_of_files = 7
file_range = range(number_of_files)

#Arctic circle coordinate
x0 = 175
y0 = 155

#Data paths
DATA_PATH = r"/home/elic/ctaylor/LPHYS2268/Data_files"
FIGURE_PATH = r"/home/elic/ctaylor/LPHYS2268/Figures/Area_PDFs"

#Plotting variabbles-------------------------------------
xlim_bottom = 90
xlim_top = 270
ylim_bottom = 90
ylim_top = 280

#Different colours to sort through for plotting
peak_colours = ['red', 'lime', 'cyan', 'magenta', 'yellow', 'orange', 'white']


#----------Build custom colourmap-----------------------------
cmap = plt.cm.Blues_r
#land colours defined as nan
cmap.set_bad(color='darkgreen')   
#Set colour for when threshold is broken (open ocean/no sea ice)
cmap.set_under(color='midnightblue')

#Lists to collect all dropout positions
dropout_x_all = []
dropout_y_all = []

#load coordinate file and extract lat and lon arrays for later use in plotting and interpolation
ds_coords = xr.open_dataset(f"{DATA_PATH}/EASEGRID_x_y_lat_lon.nc")
LAT = ds_coords['LAT'].values
LON = ds_coords['LON'].values

#Lucs IDW fuction
def idw_interpolation(var, time_index, x_pos, y_pos):

    x0 = int(np.floor(x_pos))
    x1 = x0 + 1
    y0 = int(np.floor(y_pos))
    y1 = y0 + 1

    points = [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]

    weighted_sum = 0
    weight_sum = 0

    for px,py in points:

        value = var.isel(time_counter=time_index,y=py,x=px).values

        dist = np.sqrt((x_pos-px)**2 + (y_pos-py)**2)

        if dist == 0:
            return value

        weight = 1/(dist**2)

        weighted_sum += weight*value
        weight_sum += weight

    return weighted_sum/weight_sum

#Loops starting with start dates

for start_date in start_dates:
    #Strings for start dates
    if start_date == 153:
        start_date_string = "01-11"
    if start_date == 122:
        start_date_string = "01-10"
    if start_date == 183:
        start_date_string = "01-12"
    if start_date == 213:
        start_date_string = "31-12"
    print(f"Processing start date: {start_date_string}")

    for start_yr in Year_range:
        print(f"  Processing year: {start_yr}")
        start_ensemble_number = 2024-start_yr
        steps = 365 - start_date
        total_steps = number_of_files * (365 - start_date)
        #Required for tracking paths after each loop
        long_running_step = 0
        #Create bead formation------------------------
        #Set side length
        n = int(np.sqrt(number_of_beads))
        #Create 10 evenly spaced data points that are centred on 0 and offset from the edges by 0.5
        spacing = np.linspace(-0.5, 0.5, n, endpoint=False) + 0.5/n
            
        #Create a 10x10 square grid shape
        gx, gy = np.meshgrid(spacing, spacing)  
            
        #shape the grid about the coordinate points
        bead_x = x0 + gx.ravel()  
        bead_y = y0 + gy.ravel() 
        
        #initialise arrays to store movement rows are time steps and xolumns are beads
        movement_x = np.zeros((total_steps + 1, number_of_beads))
        movement_y = np.zeros((total_steps + 1, number_of_beads))
        
        #initialise arrays to store ice concentration rows are time steps and xolumns are ice con
        iceconc_array = np.zeros((total_steps + 1, number_of_beads))
        
        #Store the initial position
        movement_x[0] = bead_x
        movement_y[0] = bead_y
        
        #Array to capture those that drop out and when
        drop_outs_array = np.zeros((number_of_beads,2))
        
        #Array to track the active beads so we can skip the non-active ones in the movement
        active_beads = np.zeros(number_of_beads)

        #File loop-------------------------------------------------------------------
        for year in file_range:
            #Break the loops if all beads have melted out or hit the land
            if np.all(active_beads == 1):
                break
            #Start from November 1st in first file
            if year == 0:
                file_start_day = start_date 
            #start from beginning of file in any other file
            else:
                file_start_day = 0  
            
            steps = 365 - file_start_day

            #-------Loading in NetCDF files and plotting for background--------------------------------------
            start_year = year + start_yr
            finish_year = start_year + 1

            #pull the files, the :02d turns the value into two figures so we get 08, 09 and 10
            #have to check the file because in the 2023 file it tries to run into 2024 which we dontt have data for and it breaks the code, this way it will just stop the loop when it gets to that point
            file_i = f"{DATA_PATH}/siconc_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"
            file_u = f"{DATA_PATH}/sivelu_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"
            file_v = f"{DATA_PATH}/sivelv_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"
            #This does the actual chck and then breaks the loop if needed (stops the script from failing)
            if not os.path.exists(file_i) or not os.path.exists(file_u) or not os.path.exists(file_v):
                print(f"File not found for year {start_year}-{finish_year}, stopping loop.")
                break
            
            ds_i = xr.open_dataset(f"{DATA_PATH}/siconc_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc")
            ds_u = xr.open_dataset(f"{DATA_PATH}/sivelu_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc")
            ds_v = xr.open_dataset(f"{DATA_PATH}/sivelv_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc")
            siconc = ds_i['siconc']
            
            #ensemble number decreases from 2014 to 2024
            start_ensemble_number = start_ensemble_number-1
            
            #Check the initial ice concentration at the bead positions and store in array
            for i in range(number_of_beads):
                if active_beads[i] == 1:
                    continue
                
                grid_x = int(bead_x[i])
                grid_y = int(bead_y[i])
                iceconc_array[0, i] = siconc.isel(time_counter=file_start_day, y=grid_y, x=grid_x).values
            
            for j in range(steps):
                #Break the loops if all beads have melted out or hit the land
                if np.all(active_beads == 1):
                    #print("All beads dropped out at step", long_running_step + j)
                    break
                #Check if bead has melted out or hit land and then skip if it has
                for i in range(number_of_beads):
                    if active_beads[i] == 1:
                        continue
                    
                    init_x = bead_x[i]
                    init_y = bead_y[i]
            
                    u_velocity = idw_interpolation(ds_u["sivelu"], file_start_day+j, bead_x[i], bead_y[i])
                    v_velocity = idw_interpolation(ds_v["sivelv"], file_start_day+j, bead_x[i], bead_y[i])
                    ice_conc = idw_interpolation(siconc, file_start_day+j, bead_x[i], bead_y[i])
                    
                    iceconc_array[long_running_step+j+1, i] = ice_conc

                    #check if bead is either below ice conc value or hitting land, add to list and then skip movement if true
                    if ice_conc < sea_ice_min or np.isnan(u_velocity) or np.isnan(v_velocity):
                        active_beads[i] = 1
                        if drop_outs_array[i, 0] == 0:
                            drop_outs_array[i, 0] = i+1
                            drop_outs_array[i, 1] = long_running_step + j
                            #print("Bead", i+1, "dropped out at", long_running_step + j + 1)
                        continue
                        
                    init_x = bead_x[i]
                    init_y = bead_y[i]
                    #Move along (86,400 seconds in a day, 25km grid size)
                    bead_x[i] = bead_x[i] + (u_velocity * 86400/25000)
                    bead_y[i] = bead_y[i] + (v_velocity * 86400/25000)
                
                #Save the new position to the trajectory arrays. +1 because it is the next step in the array and to preserve x0 and y0
                movement_x[j+1+long_running_step] = bead_x
                movement_y[j+1+long_running_step] = bead_y

                if not np.all(active_beads == 1):
                    last_active_step = long_running_step + j + 1
                      
         #Move long_running step along
        long_running_step = long_running_step + steps

        #Collect the dropout positions for this year and add to the overall list
        for i in range(number_of_beads):
            dropout_step = int(drop_outs_array[i, 1])

            # Only include beads that actually dropped out (step > 0)
            if dropout_step > 0:

                #Potential safety net when proofing with AI
                #idx = min(dropout_step, movement_x.shape[0] - 1)
                #Take x position and y position on the day the bead dropped out and add to list
                dropout_x_all.append(movement_x[dropout_step, i])
                dropout_y_all.append(movement_y[dropout_step, i])

#Convert the lists to numpy arrays so the KDE function can use them
dropout_x_all = np.array(dropout_x_all)
dropout_y_all = np.array(dropout_y_all)

print(f"\nTotal dropout locations collected: {len(dropout_x_all)}/4000")

#Process to plot 3D heatmap came from Claude AI but concept and requirements were derived by us. We checked that the dropouts and plotting behaved as expected.
#Build the KDE data/surface------------------------------------------------------------------------------

#Stack them into the (2, N) shape expected by gaussian_kde
xy_stack = np.vstack([dropout_x_all, dropout_y_all])

#Fit the KDE to the data
kde_2d = gaussian_kde(xy_stack)

#Build a grid to evauate KDE on, use map area
grid_res = 200
xi = np.linspace(xlim_bottom, xlim_top, grid_res)
yi = np.linspace(ylim_bottom, ylim_top, grid_res)
Xi, Yi = np.meshgrid(xi, yi)

#Stack the grid coordinates into (2, M) shape for evaluation
grid_coords = np.vstack([Xi.ravel(), Yi.ravel()])

#Evaluate the KDE on the grid
KDE = kde_2d(grid_coords).reshape(Xi.shape)

#Normalise the KDE from 0 to 1 for percentage contour plotting
KDE_normalised = KDE / np.max(KDE)

#Find peaks of KDE surface---------------------------------------------------------------------------------
neighbours = 10
#Creaet a boolean mask where True indicates a local maximum
local_max_mask = (KDE_normalised == maximum_filter(KDE_normalised, size=neighbours))

#Keep only peaks above a threshold to get rig of noise
threshold = 0.3
# Filter out low density noise
local_max_mask = local_max_mask & (KDE_normalised > threshold)

#Get coordinates and densities of the peaks
peak_rows, peak_cols = np.where(local_max_mask)

#Store the peaks
peaks = []

for row, col in zip(peak_rows, peak_cols):
    x_peak = Xi[row, col]
    y_peak = Yi[row, col]
    density_peak = KDE_normalised[row, col]
    peaks.append((x_peak, y_peak, density_peak))

#sort to strongest peak first

peaks.sort(key=lambda x: x[2], reverse=True)

print(f"Number of peaks found: {len(peaks)}")
#print the value of the peaks
for i, (x_peak, y_peak, density_peak) in enumerate(peaks):
    print(f"Peak {i+1}: Location=({x_peak:.1f}, {y_peak:.1f}), Density={density_peak:.3f}")

#peaks in lat and lon for later use
print("\nPeaks in lat and lon:")
for i, (x_peak, y_peak, density_peak) in enumerate(peaks):
    #round to nearest integer to use to mach to grid index
    col = int(round(x_peak))
    row = int(round(y_peak))
    lat_peak = LAT[row, col]
    lon_peak = LON[row, col]
    print(f"Peak {i+1}: Location=({lat_peak:.2f}, {lon_peak:.2f}), Density={density_peak:.3f})")

#Plotting the KDE and peaks---------------------------------------------------------------------------------
fig, ax = plt.subplots()
#Load in background map
#load in siconc file (doesn't matter what year because we are going to remove ice values and land shouldn't move)
siconc_background = xr.open_dataset(f"{DATA_PATH}/siconc_sipn_easegrid_2024-06-01_01_1d_20230601_20240531_icemod.nc")['siconc'].isel(time_counter=0)

#set all ice values to 0
masked = siconc_background * 0
masked.plot(ax=ax, cmap=cmap, vmin=sea_ice_min, vmax=1, add_colorbar=False)

#KDE contour plot
contour_filled = ax.contourf(Xi, Yi, KDE_normalised, levels=np.linspace(0.1,1.0,8), cmap='YlOrRd', alpha=0.5, zorder=2)

cbar = fig.colorbar(contour_filled, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Normalised dropout probability")

contour_lines = ax.contour(Xi, Yi, KDE_normalised, levels=np.linspace(0.1,1.0,8),colors='white', linewidths=0.6, alpha=0.7, zorder=3)
ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.1f', colors='white')

#Scatter of dropout locations
ax.scatter(dropout_x_all, dropout_y_all, color='k', s=6, alpha=0.3, zorder=4)

#plot the lat lon lines
lat_lines = [60,70,80,90]
#lon_lines = [0,30,60,90,120,150,180,210,240,270,300,330]
lon_lines = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]

# Create a full grid of x and y indices matching the LAT/LON arrays
full_x = np.arange(LAT.shape[1])  # 0 to 360
full_y = np.arange(LAT.shape[0])  # 0 to 360

#for lat in lat_lines:
#    lat_mask = np.abs(LAT - lat) < 1
#    ax.contour(full_x, full_y, lat_mask.astype(float), levels=[0.5],
#               colors='gray', linewidths=0.5, alpha=0.5, zorder=1)

#for lon in lon_lines:
#    lon_mask = np.abs(LON - lon) < 1
#    ax.contour(full_x, full_y, lon_mask.astype(float), levels=[0.5],
#               colors='gray', linewidths=0.5, alpha=0.5, zorder=1)


# Dotted lat/lon grid
#for lat in lat_lines:
#    mask = np.abs(LAT - lat) < 0.2  # tight tolerance = one thin ring of dots
#    rows, cols = np.where(mask)
#    ax.scatter(cols, rows, color='gray', s=0.3, alpha=0.4, zorder=1)

#for lon in lon_lines:
#    mask = np.abs(LON - lon) < 0.2
#    rows, cols = np.where(mask)
#    ax.scatter(cols, rows, color='gray', s=0.3, alpha=0.4, zorder=1)

#---------------------------------------------------------------------------------------------------------
#ChatGPT suggested solution, not perfect but better than my previous attempts
x = np.arange(LAT.shape[1])
y = np.arange(LAT.shape[0])
X, Y = np.meshgrid(x, y)

# Latitude lines
for lat_val in [60, 70, 80]:
    ax.contour(X, Y, LAT, levels=[lat_val],
               colors='grey', linewidths=0.7, alpha=0.7, zorder=1)

# Longitude lines
for lon_val in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]:
    ax.contour(X, Y, LON, levels=[lon_val],
               colors='grey', linewidths=0.7, alpha=0.7, zorder=1)


#---------------------------------------------------------------------------------------------------------


#Mark the peaks
for i, (x_peak, y_peak, density_peak) in enumerate(peaks):
    col = int(round(x_peak))
    row = int(round(y_peak))
    lat_peak = LAT[row, col]
    lon_peak = LON[row, col]
    ax.scatter(x_peak, y_peak, color=peak_colours[i % len(peak_colours)], s=35, marker ='D', zorder=5, label=f'Peak {i+1}: ({lat_peak:.1f}, {lon_peak:.1f})')

#Put start point star on map
ax.scatter(x0, y0, marker='*', color='k', edgecolors='white', s=150, zorder=5)

#Setup plot limits and labels
ax.set_xlim(xlim_bottom, xlim_top)
ax.set_ylim(ylim_bottom, ylim_top)
ax.invert_yaxis()
ax.legend(title="Dropout peaks", loc='upper right', fontsize=7, framealpha=0.6)
ax.set_title(f"Spatial PDF of bead dropout locations\n"
             f"{Year_range[0]}-{Year_range[-1]}, all start dates, "
             f"start point ({x0}, {y0})")
fig.tight_layout()
fig.savefig(f"{FIGURE_PATH}/Spatial_PDF_dropout_locations_{Year_range[0]}_to_{Year_range[-1]}_start_dates_{start_dates[0]}_{start_dates[-1]}.jpg",
            dpi=300, bbox_inches='tight')