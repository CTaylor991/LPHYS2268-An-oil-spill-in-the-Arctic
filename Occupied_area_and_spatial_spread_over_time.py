# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:34:03 2026

@author: anegu
"""

"""
Analyse de la surface occupée et de la dispersion des beads.

On calcule trois métriques :
1. Surface occupée par toutes les beads avec Convex Hull
2. Spread spatial : dispersion des beads autour de leur centre moyen
3. Distance moyenne depuis le point de départ : transport global du paquet

Modification :
- on teste 4 dates de départ :
  01-10, 01-11, 01-12, 31-12
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from scipy.spatial import ConvexHull


# ---------------------------------------------------------
# Paramètres généraux
# ---------------------------------------------------------

start_dates = {
    122: "01-10",
    153: "01-11",
    183: "01-12",
    213: "31-12"
}

Year_range = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

DATA_PATH = r"D:/UCLOUVAIN/MOODLE/M1/Q2/LPHYS2268/Projet/Data"

number_of_beads = 100
sea_ice_min = 0.15

x0 = 175
y0 = 155

number_of_files = 4


# ---------------------------------------------------------
# Dictionnaires globaux par date de départ
# ---------------------------------------------------------

mean_occupied_areas_by_date = {}
mean_spreads_by_date = {}
mean_distances_by_date = {}


# ---------------------------------------------------------
# Interpolation IDW
# ---------------------------------------------------------

def idw_interpolation(var, time_index, x_pos, y_pos):
    """
    Interpolation inverse distance weighting sur les 4 points voisins.
    """

    x0_floor = int(np.floor(x_pos))
    x1 = x0_floor + 1
    y0_floor = int(np.floor(y_pos))
    y1 = y0_floor + 1

    points = [(x0_floor, y0_floor), (x1, y0_floor),
              (x0_floor, y1), (x1, y1)]

    weighted_sum = 0
    weight_sum = 0

    for px, py in points:

        value = var.isel(time_counter=time_index, y=py, x=px).values

        dist = np.sqrt((x_pos - px)**2 + (y_pos - py)**2)

        if dist == 0:
            return value

        weight = 1 / dist**2
        weighted_sum += weight * value
        weight_sum += weight

    return weighted_sum / weight_sum


# ---------------------------------------------------------
# Surface par Convex Hull
# ---------------------------------------------------------

def convex_hull_area_km2(x_positions, y_positions, cell_size_km=25):
    """
    Surface occupée par toutes les beads.
    """

    points = np.column_stack((x_positions, y_positions))

    if len(points) < 3:
        return cell_size_km**2

    try:
        hull = ConvexHull(points)
        area_grid_units = hull.volume
        area_km2 = area_grid_units * cell_size_km**2

        return max(area_km2, cell_size_km**2)

    except:
        return cell_size_km**2


# ---------------------------------------------------------
# Spread spatial
# ---------------------------------------------------------

def spatial_spread_km(x_positions, y_positions, cell_size_km=25):
    """
    Mesure la dispersion des beads autour de leur centre moyen.
    """

    spread_grid = np.sqrt(np.var(x_positions) + np.var(y_positions))
    return spread_grid * cell_size_km


# ---------------------------------------------------------
# Distance moyenne depuis le point de départ
# ---------------------------------------------------------

def mean_distance_from_start_km(x_positions, y_positions, x0, y0, cell_size_km=25):
    """
    Mesure la distance moyenne parcourue depuis le point initial.
    """

    distances_grid = np.sqrt((x_positions - x0)**2 + (y_positions - y0)**2)
    return np.mean(distances_grid) * cell_size_km


# ---------------------------------------------------------
# Fonction pour calculer une moyenne entre ensembles
# ---------------------------------------------------------

def ensemble_mean(dictionary, Year_range):
    """
    Calcule la moyenne entre ensembles même si les séries n'ont pas toutes
    exactement la même longueur.
    """

    valid_values = [dictionary[yr] for yr in Year_range if yr in dictionary]

    max_length = max(len(a) for a in valid_values)

    padded = np.full((len(valid_values), max_length), np.nan)

    for i, a in enumerate(valid_values):
        padded[i, :len(a)] = a

    return np.nanmean(padded, axis=0)


# ---------------------------------------------------------
# Boucle principale sur les dates de départ
# ---------------------------------------------------------

for start_date, start_date_string in start_dates.items():

    print("\n====================================")
    print(f"Date de départ : {start_date_string}")
    print(f"Index start_date : {start_date}")
    print("====================================")

    drop_outs = {}
    occupied_areas = {}
    spreads = {}
    distances_from_start = {}

    # -----------------------------------------------------
    # Boucle sur les ensembles
    # -----------------------------------------------------

    for start_yr in Year_range:

        constant_start_yr = start_yr
        start_ensemble_number = 2024 - start_yr

        print(f"Traitement du membre d'ensemble : {constant_start_yr}")

        file_range = range(number_of_files)
        long_running_step = 0

        area_this_year = []
        spread_this_year = []
        distance_this_year = []

        # -------------------------------------------------
        # Création initiale des beads
        # -------------------------------------------------

        n = int(np.sqrt(number_of_beads))

        spacing = np.linspace(-0.5, 0.5, n, endpoint=False) + 0.5 / n
        gx, gy = np.meshgrid(spacing, spacing)

        bead_x = x0 + gx.ravel()
        bead_y = y0 + gy.ravel()

        drop_outs_array = np.zeros((number_of_beads, 2))

        # 0 = active, 1 = inactive
        active_beads = np.zeros(number_of_beads)

        # Métriques initiales
        area_this_year.append(convex_hull_area_km2(bead_x, bead_y))
        spread_this_year.append(spatial_spread_km(bead_x, bead_y))
        distance_this_year.append(mean_distance_from_start_km(bead_x, bead_y, x0, y0))

        # -------------------------------------------------
        # Boucle sur les fichiers
        # -------------------------------------------------

        for year in file_range:

            if year == 0:
                file_start_day = start_date
            else:
                file_start_day = 0

            steps = 365 - file_start_day

            start_year = year + start_yr
            finish_year = start_year + 1

            file_i = f"{DATA_PATH}/siconc_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"
            file_u = f"{DATA_PATH}/sivelu_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"
            file_v = f"{DATA_PATH}/sivelv_sipn_easegrid_2024-06-01_{start_ensemble_number:02d}_1d_{start_year}0601_{finish_year}0531_icemod.nc"

            if not os.path.exists(file_i) or not os.path.exists(file_u) or not os.path.exists(file_v):
                print(f"Fichier manquant pour {start_year}-{finish_year}, arrêt de ce membre.")
                break

            ds_i = xr.open_dataset(file_i)
            ds_u = xr.open_dataset(file_u)
            ds_v = xr.open_dataset(file_v)

            siconc = ds_i["siconc"]

            print("Taille du fichier siconc :", siconc.shape)

            start_ensemble_number -= 1

            # ---------------------------------------------
            # Boucle journalière
            # ---------------------------------------------

            for j in range(steps):

                # On déplace uniquement les beads encore actives
                if not np.all(active_beads == 1):

                    for i in range(number_of_beads):

                        if active_beads[i] == 1:
                            continue

                        u_velocity = idw_interpolation(
                            ds_u["sivelu"],
                            file_start_day + j,
                            bead_x[i],
                            bead_y[i]
                        )

                        v_velocity = idw_interpolation(
                            ds_v["sivelv"],
                            file_start_day + j,
                            bead_x[i],
                            bead_y[i]
                        )

                        ice_conc = idw_interpolation(
                            siconc,
                            file_start_day + j,
                            bead_x[i],
                            bead_y[i]
                        )

                        # La bead devient inactive si elle quitte la glace
                        # ou si elle arrive sur la terre
                        if ice_conc < sea_ice_min or np.isnan(u_velocity) or np.isnan(v_velocity):

                            active_beads[i] = 1

                            if drop_outs_array[i, 0] == 0:
                                drop_outs_array[i, 0] = i + 1
                                drop_outs_array[i, 1] = long_running_step + j

                            continue

                        # Déplacement : m/s vers cellule/jour
                        bead_x[i] += u_velocity * 86400 / 25000
                        bead_y[i] += v_velocity * 86400 / 25000

                # ---------------------------------------------
                # Calcul des métriques avec TOUTES les beads
                # ---------------------------------------------

                area_today = convex_hull_area_km2(bead_x, bead_y)
                spread_today = spatial_spread_km(bead_x, bead_y)
                distance_today = mean_distance_from_start_km(bead_x, bead_y, x0, y0)

                area_this_year.append(area_today)
                spread_this_year.append(spread_today)
                distance_this_year.append(distance_today)

            long_running_step += steps

            ds_i.close()
            ds_u.close()
            ds_v.close()

        drop_outs[constant_start_yr] = drop_outs_array.copy()
        occupied_areas[constant_start_yr] = area_this_year.copy()
        spreads[constant_start_yr] = spread_this_year.copy()
        distances_from_start[constant_start_yr] = distance_this_year.copy()

        print("Liste des dropouts :\nBead, jour\n", drop_outs_array)

    # Moyennes pour cette date de départ
    mean_occupied_areas_by_date[start_date_string] = ensemble_mean(occupied_areas, Year_range)
    mean_spreads_by_date[start_date_string] = ensemble_mean(spreads, Year_range)
    mean_distances_by_date[start_date_string] = ensemble_mean(distances_from_start, Year_range)


# ---------------------------------------------------------
# Figure 1 : Surface occupée moyenne pour les 4 dates
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for date_label, mean_area in mean_occupied_areas_by_date.items():
    plt.plot(mean_area, linewidth=2, label=date_label)

plt.xlabel("Day since release")
plt.ylabel("Occupied area using all beads (km²)")
plt.title("Occupied area over time for 4 start dates, start point (175,155)")
plt.legend(title="Start date")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# Figure 2 : Spread spatial moyen pour les 4 dates
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for date_label, mean_spread in mean_spreads_by_date.items():
    plt.plot(mean_spread, linewidth=2, label=date_label)

plt.xlabel("Day since release")
plt.ylabel("Spatial spread (km)")
plt.title("Spatial spread over time for 4 start dates, start point (175,155)")
plt.legend(title="Start date")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# Figure 3 : Distance moyenne depuis le point de départ
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for date_label, mean_distance in mean_distances_by_date.items():
    plt.plot(mean_distance, linewidth=2, label=date_label)

plt.xlabel("Day since release")
plt.ylabel("Mean distance from start (km)")
plt.title("Mean distance travelled for 4 start dates, start point (175,155)")
plt.legend(title="Start date")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# Diagnostic : vitesse maximale d'expansion pour chaque date
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for date_label, mean_spread in mean_spreads_by_date.items():

    d_spread = np.diff(mean_spread)

    valid = ~np.isnan(d_spread)

    if np.any(valid):

        max_spread_growth_day = np.where(valid)[0][np.argmax(d_spread[valid])]
        max_spread_growth = d_spread[max_spread_growth_day]

        print(
            f"Date {date_label} : "
            f"jour de croissance maximale du spread = {max_spread_growth_day}, "
            f"croissance = {max_spread_growth:.2f} km/jour"
        )

        plt.plot(
            d_spread,
            linewidth=2,
            label=f"{date_label} | max {max_spread_growth:.2f} km/day"
        )

plt.xlabel("Day since release")
plt.ylabel("Expansion rate based on spread (km/day)")
plt.title("Expansion rate for 4 start dates, start point (175,155)")
plt.legend(title="Start date", fontsize=8)
plt.grid(True)
plt.show()