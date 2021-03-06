# Synthetic Terrain Generation and line-of-sight Calculations

> This program generates synthetic 2-dimensional terrain and calculates line-of-sight along that terrain. The program can generate any number of synthetic terrain sets with the associated line-of-sight vectors that indicate is line-of-sight exists between the observer and all the points along the terrain.

![Sample Output](/Images/Synthetic_Terrain.png)

## Setup and Run

The simulation is setup, run, and controlled from the [SyntheticLinearTerrainandLOS.py](SyntheticLinearTerrainandLOS.py) file. The simulation parameters are set in lines 13-35. The settings for the terrain are set in lines 14-19. The terrain is generated by adding circular (would be spherical in 2d) objects of varying sizes to the flat surface of the terrain. Then, a 'flattening' factor is applied by taking the terrain values to the 'flattening' power - this makes the peaks steeper and the bottoms flatter similar to real terrain. Increasing the "hillmin", "hillmax", or "numhills" will also make the terrain more rugged. The terrainLength is the length of the 2D vector. This number sets the units for the simulation. If "terrainLength" is in meters then all other settings and results will be in meters. Likewise, if it is in feet, then all other settings and results will be in feet. The simulation generates a terrain height at every integer distance between 0 (the location of the observer) and the "terrainLength."

![Terrain_Settings](/Images/Terrain_Settings.PNG)

The observer and the target heights can also be set in lines 22 and 23. These should be expressed in the same units as the "terrainLength" variable set in line 18. The "observer" is placed at point 0 with a height of "observerHeight" above the terrain. line-of-sight is calculated between the observer (at its observer height) and each terrain point assuming that the target at that terrain point is at an elevation of "targetHeight" above the ground.

![Height_Settings](/Images/Height_Settings.PNG)

The simulation will generate and show an image of a single example terrain profile based on the settings in lines 26-28. "show_plot" and "save_plot" accept values of True or False. If "show_plot" is True, the program will render a plot similar to the example above. If "save_plot" is True, it will save that plot to the [Images](./Images) folder with the name set by "plot_file_name."

![Plot_Settings](/Images/Plot_Settings.PNG)

Beyond the single terrain set that is generated for the image as explained above, the program is capable of generating multiple sets of terrain with the associated line-of-sight vectors. This is controlled by lines 31-35. If "want_terrain_set" is set to True, then the program will generate the number of random sample terrain vectors and associated line-of-sight vectors specified by "numSamples." The terrain vectors are 1 x terrainLength vector of terrain heights at each integer distance between 0 and terrainLength. The line-of-sight vectors is a binary 1 x terrainLength vector of "0" or "1." A "0" indicates that the observer (at location 0 and height of observerHeight) cannot see the target with a height of targetHeight above the ground at the given point.  A "1" indicates that the observer can see the target. The terrain and visibility vectors are saves in the [Results](./Results) folder as csv files. Each row of the terrain file represents a separate 2D synthetic terrain and corresponds to the same row in the visibility file.

![Terrain_Generation_Settings](/Images/Terrain_Generation_Settings.PNG)

## Credits

This simulation was built with the help of the following people:

* Jim Jablonski
* John Grant

## Python Libraries

This simulation was built using the following python libraries and versions:

* Python v3.7.5
* numpy v1.16.5
* matplotlib v3.1.1
* joblib 0.14.0 (if doing multiprocessing)
* tqdm 4.38.0 (if doing multiprocessing)
