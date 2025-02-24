- create a terrain object
    - terrain must be flat and green
    - terrain must have a given extent
    - [optional] terrain can be infinite
    - entities must be able to stand on the terrain

- create camera object
    - camera must be able to move in one direction at a given speed
    - camera must be able to move in any direction at a given speed
    - camera must be able to rotate at a given speed
    - camera must be able to ascend and descend at a given speed
    - camera must be able to move, rotate and change elevation at the same time
    - camera must be able to perform movement/rotation/elevation given probabilities for each
    - camera must stay within a certain distance (min/max) above the terrain
    - camera must look at the terrain, straight down (90°)
    - the game view corresponds to the portion of the terrain that the camera sees
    - the aspect ratio of the camera is provided 
    - camera must be blocked from moving to a position where the terrain no longer exist
    
- create animal object
    - animals must have an ID
    - animals should have a rectangular shape
    - animals should be given a color
    - animal should be able to stand on the terrain
    - animals should be able to move on the terrain in one direction given a speed
    - animals should be able to move in any direction at a given speed
    - animals should not be able to pass through each other (bounding box)
    
- ID and position of each animal in the camera frame should be recorded (in [0,1] for both frame height and width)