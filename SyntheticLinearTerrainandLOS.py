## Generating some synthetic terrain and corresponding visibility masks with viewpoint at far left.
#Terrain gen works by adding circular (would be spherical in 2d) objects to the surface of the terrain.
    # Adding many of these of varying sizes results in fairly realistic looking terrain. The 'flattening'
    # is simply taking the terrain values to a power - this makes the peaks steeper and the bottoms 
    # flatter similar to real terrain. 

# Needed libraries (for multiprocessing see additional libraries below)
import os
import numpy as np
import matplotlib.pyplot as plt

## User settings ##
# Terrain Generation Options
hillmin = 10  #min radius of hill
hillmax = 150  #max radius of hill
numhills = 500  #number of hills to create
flattening = 2  #flattening factor - larger values make the terrain more steep and rugged.
terrainLength = 200 # Total length of the synthetic terrain in meters
random_seed = 100 # seed for random number generator (used for reproducibility)

# Heights of the observer and target
observerHeight = 1 # height above ground for observer in meters
targetHeight = 0 # height above ground of distant target in meters

# Plotting Options
show_plot = True #Want to have an example output graphic print to screen
save_plot = True #save example output graphic to disk
plot_file_name = 'Synthetic_Terrain.png'

#Settings to save terrain data to file
want_terrain_set= True
multiprocessing = True
numSamples=100
terrain_file_name='newTerrains.csv'
visibility_file_name='newVisibility.csv'


#############################################################
## Begin Program ##
np.random.seed(random_seed) # fix random seed for reproducibility

if multiprocessing:
    # Need additional libraries for multiprocessing options
    import multiprocessing as mp
    from joblib import Parallel, delayed
    from tqdm import tqdm


def genLinearTerrain(terrainLength, hillmin, hillmax, numhills, flattening, num):
    '''Generats synthetic terrain by adding circular objects varying sizes at random points and applying a flattening
    factor (larger values of flattening make the terrain more rugged (peaks steeper and bottoms flatter)'''
    terrain=np.zeros(terrainLength)

    for i in range(0,numhills):
        # add hill
        radius=np.random.uniform(hillmin, hillmax, 1)

        centerpoint=(np.random.uniform(0-hillmax,terrainLength+hillmax))

        #based on centerpoint find which points along the line are effected by adding a circle. 
        minPointAffected=np.round(centerpoint-radius).astype(int)[0]
        maxPointAffected=np.round(centerpoint+radius).astype(int)[0]

        if (minPointAffected < 0):
            minPointAffected=0

        if (maxPointAffected > terrainLength):
            maxPointAffected = terrainLength

        for j in range(minPointAffected, maxPointAffected):
            radicand = (radius**2-(centerpoint-j)**2)
            if radicand >= 0:
                val = np.sqrt(radicand)
            else:
                val = 0
                
            terrain[j] = terrain[j] + val

    terrain=terrain**flattening

    terrain = (terrain - np.mean(terrain)) / np.std(terrain)    
    return(terrain)



def losCal(elevationsVector, observer_height, target_height):
    ''' Calculates line of sight (LOS) between two points of a given height. The elevations vector is a set of elevations
    between equally spaced points. The observer_height and the target_height should be expressed in the same units as the
    elevation vector'''
    
    lengthOfVector=len(elevationsVector)
    distances=np.array(list(range(0,lengthOfVector)))
    distances[0]=.01  # just to avoid a divide by 0 error - this pixel will be visible anyway. 
    visibility = np.zeros(lengthOfVector)
    observerGradients = np.zeros(lengthOfVector)
    
    visibility[0]=1
    visibility[1]=1
    for i in range(1,lengthOfVector):
        observerGradients[i] = ((elevationsVector[i] + target_height) - (elevationsVector[0] + observer_height)) / distances[i]
        
        maxheightLine = observerGradients[i] * distances[0:i] + (elevationsVector[0] + observer_height)
        
        canSee=sum((elevationsVector[0:i] > maxheightLine) *1)
        
        if canSee==0:
            visibility[i]=1
                
    return (visibility)

# Generate a number of 2D terrain and line of sight calculation sets
if want_terrain_set:
    # Multiprocessing will speed up calculations for a large number of terrains sets
    if multiprocessing:
        Terrain_fun_args = (terrainLength, hillmin, hillmax, numhills, flattening)
        
        def Terrain_helper_fun(args):
            return genLinearTerrain(*(*(Terrain_fun_args),args))
            
        num_cores = mp.cpu_count()-1
        terrains = Parallel(n_jobs=num_cores)(delayed(Terrain_helper_fun)(i) for i in tqdm(range(0, numSamples)))  

    # Single process 2D terrain generation and line of sight calculations.
    else:
        terrains=np.zeros((numSamples, terrainLength))
    
        for i in range(numSamples):
            terrains[i] = genLinearTerrain(terrainLength, hillmin, hillmax, numhills, flattening, numSamples)
                
            if (i % 100) == 0 or i == (numSamples - 1):
                print(f"starting terrain iteration: {i+1} of {numSamples}")
    

    visibility = np.apply_along_axis(losCal, 1, terrains, observer_height = observerHeight, target_height = targetHeight )

    np.savetxt(os.path.join('./Results',terrain_file_name), terrains, delimiter = ',')
    np.savetxt(os.path.join('./Results',visibility_file_name), visibility, delimiter = ',', fmt='%i')


#This section demonstrates the functionality of both functions with a crude display. 
if show_plot or save_plot:
    terrain = genLinearTerrain(terrainLength, hillmin, hillmax, numhills, flattening, 1)
    result = losCal(terrain, observerHeight, targetHeight)
    
    xpts = np.array(list(range(0,len(terrain))))
    
    canSeeTerrain = np.ma.masked_where(result==0, terrain)
    noSeeTerrain = np.ma.masked_where(result==1, terrain)
    observer_plt = terrain[0] + observerHeight
    
    fig, ax = plt.subplots()
    ax.plot(xpts,terrain)
    ax.plot(xpts, canSeeTerrain, color='blue', label='Can See')
    ax.plot(xpts, noSeeTerrain, color='red', label='Cannot See')
    
    ax.plot(0, observer_plt, 'bo', label='Observer') 
    plt.title('Synthetic Terrain')
    ax.legend()
    
    if save_plot:
        plt.savefig(os.path.join('./Images',plot_file_name))

    if show_plot:
        plt.show()







