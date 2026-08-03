# 2D - 3D Pose estimation project
**Ron Shavit & Roy Levy**  
**2026**

## Setup
### Installing the project
Clone repository from https://github.com/RonShavit/PoseEstimationProject.git
### Setting up the virtual enviroment
1. Run:  
    for Windows:  
    `python -m venv venv`  
    `venv\Scripts\activate`  
    for Linux/MacOS:  
    `python3 -m venv venv`  
    `source venv/bin/activate`

2. (Installing requiremnts) Run:  
    `pip install -r "requirements.txt"`

## Running the Project
Run: `python world_split_v5.22.py`  
You will be then prompted to select a map. by defualt:
>Select a map:
>
>1\. map_1   
>2\. map_2  
>3\. map_3  
>4\. map_4  
>5\. map_5  
>
>Enter selection: 

Enter a number corresponding to a given option.  
  
You will then be asked if to simulate different lighting than used in the database (y/N). This will only affect feature runs (see below).
After choosing, the world will load in recording mode.
### Modes and controls
**1. Recording Mode**  
In recording mode you may fly around freely and capture your current position for use in Viewing/Picking mode.

***Controls:***
- **w**: Move forward
- **s**: Move backwards
- **a**: Move left
- **d**: Move Right
- **↑**: Turn camera upwards
- **↓**: Turn camera downwards
- **←**: Turn camera left
- **→**: Turn camera right
- **b**: Record current position
- **r**: Enter Viewing mode
- **p**: Enter Picking mode
- **t**: enter Trackers mode
- **f**: enter Features run mode

***Views:***
- Left view: Controlable camera. Moves when pressing WASD/arrow keys
- Right view: Static in this mode. Recorded position will appear as blue pyramids
  
***Notes:***
- The recorded position will appear as a square pyramid, with the base of the pyramid representing the front of the camera, and the tip of the pyramid (colored black) representing the back of it
- Sometimes, the position of the camera may be outside of the left view camera's field of view, and as such the pyramid representing it will not appear on the right view
  
**2. Viewing mode**  
In Viewing mode you may look through positions previously saved in Recording mode.
***Controls:***
- **←**: Move to previous recorded position
- **→**: Move to next recorded position
- **r**: Enter Recording mode
- **p**: Enter Picking mode
- **t**: enter Trackers mode
- **f**: enter Features run mode

***Views:***
- Left view: Shows previously recorded position, starting with the most recently recorded one.
- Right view: Static in this mode. Recorded position will appear as blue pyramids, with the position currently viewed appearing as red instead

***Notes:***
- Initially, the starting position will be automaticly recorded
- The recorded position will appear as a square pyramid, with the base of the pyramid representing the front of the camera, and the tip of the pyramid (colored black) representing the back of it
- Sometimes, the position of the camera may be outside of the left view camera's field of view, and as such the pyramid representing it will not appear on the right view

  

**3. Picking mode**  
In picking mode you may choose pairs of 2D-3D points and use them to estimate the postition of the camera.  
***Controls:***
- **Left click**: On left view, pick a point in 3D space. On right view, pick a corresponding point in 2D space 
- **c**: Estimate current position of the camera (on the right view) and display estimated position/rotation error
- **r**: Enter Recording mode
- **t**: enter Trackers mode
- **f**: enter Features run mode

***Views:***
- Left view: Shows last position the camera was at for 3D point choice
- Right view: Shows most recently recorded position for 2D point choise

***Notes:***
- Initially, the starting position will be automaticly recorded
- As the Position estimation algorithm requires at least 4 points, pressing **c** with less than 4 pairs choosen will not return an estimate
- A point in 3D space (left view) must be selected before a corresponding point in 2D space is chosen (right view)
- After pressing **c** to estimate the position, if an estimate was returned, an overlay of the view from the estimated position will be displayed over the actual view on the right


**4. Trackers mode**  
In trackers mode colorful trackers will appear in 3D space. You may move the camera and estimate its position using those trackers.  

***Controls:***
- **w**: Move forward
- **s**: Move backwards
- **a**: Move left
- **d**: Move Right
- **↑**: Turn camera upwards
- **↓**: Turn camera downwards
- **←**: Turn camera left
- **→**: Turn camera right
- **b**: Estimate current position. Record both estimated and actual camera position.
- **n**: view previous recorded position/estimation pair
- **m**: view next recorded position/estimation pair
- **r**: Enter Recording mode
- **p**: Enter Picking mode
- **f**: enter Features run mode


***Views:***
- Left view: Controlable camera. Moves when pressing WASD/arrow keys. Trackers will be visable on this view
- Right view: Static global camera. Pairs of estimated position/actual position recorded by pressing **b** will appear as pairs of green/blue square pyramids respectivly

***Notes:***
- As the Position estimation algorithm requires at least 4 points, pressing **b** with less than 4 trackers visable will not return an estimate, and will not record the results
- The recorded estimation/position will appear as a square pyramid, with the base of the pyramid representing the front of the camera, and the tip of the pyramid (colored black) representing the back of it
- Sometimes, the position/estimation of the camera may be outside of the left view camera's field of view, and as such the pyramid representing it will not appear on the right view
- Trackers too far away from the camera might not be detected ny the algorithm
- After pressing **b** to estimate position or **n**/**m** to view recorded position/estimation pairs, an overlay of the estimated position's view will appear over the actual view
- By defualt, 8 trackers are included. See `Editing trackers` to learn how to change the trackers


**5. Features run mode**
In Features run mode, you may control the camera and use previously gathered data to estimate the current camera position.

***Controls:***
- **w**: Move forward
- **s**: Move backwards
- **a**: Move left
- **d**: Move Right
- **↑**: Turn camera upwards
- **↓**: Turn camera downwards
- **←**: Turn camera left
- **→**: Turn camera right
- **b**: Estimate current position. Record both estimated and actual camera position.
- **n**: view previous recorded position/estimation pair
- **m**: view next recorded position/estimation pair
- **r**: Enter Recording mode
- **p**: Enter Picking mode
- **t**: enter Trackers mode

***Views:***
- Left view: Controlable camera. Moves when pressing WASD/arrow keys. Trackers will be visable on this view
- Right view: Static global camera. Pairs of estimated position/actual position recorded by pressing **b** will appear as pairs of green/blue square pyramids respectivly

***Notes:***
- The recorded estimation/position will appear as a square pyramid, with the base of the pyramid representing the front of the camera, and the tip of the pyramid (colored black) representing the back of it
- Sometimes, the position/estimation of the camera may be outside of the left view camera's field of view, and as such the pyramid representing it will not appear on the right view
- When viewing a point with little covrage in the database, the estimation may use less strict solving methods and will return a less accurate estimation. Sometimes, when there are not enough data points, no estimatio will be returned, but the real position will be recorded for reviewing purposes (see `Gathering data` to see how to gather editional data).  

## Gathering data
Run: `python world_split_v5.22.py --pre`  
You will be then prompted to select a map. by defualt:
>Select a map:
>
>1\. map_1   
>2\. map_2  
>3\. map_3  
>4\. map_4  
>5\. map_5  
>
>Enter selection: 

Enter a number corresponding to a given option.   
After selected a map you will be asked to choose a data gathering mode:
>1\. Continue / edit existing active database  
>2\. Create a new empty database  
>3\. Demo mode - temporary, read-only, never saves  
>4\. Cancel  

1. Data gathered will be added to already existing database
2. Data gathered will be added to a new database. If a database already exists it will be deleted
3. Data gathered will not be saved. If a database already exists it will be unmodified
4. Exits immediatly   

After choosing an option from 1-3, the world will load in data gathering mode

### Data gathering mode
***Controls:***
- **w**: Move forward
- **s**: Move backwards
- **a**: Move left
- **d**: Move Right
- **↑**: Turn camera upwards
- **↓**: Turn camera downwards
- **←**: Turn camera left
- **→**: Turn camera right
- **v**: Choose a viewpoint to select features from
- **Right click**: Select a feature from the ones available on the right view
- **Left click**: Select a corresponding 3D point on the left view
- **CTRL**+**S**: Save changes to database
- **U**: Undo last selection
- **[**: Go to previous saved viewpoint
- **]**: Go to next saved viewpoint

***Views:***
- Left view: Controlable camera, selected 3D points for the current viewpoint will appear as red spheres in 3D space
- Right view: Shows to current viewpoint, features will appear as green "+" markers, selected features will appear as green squares

***Notes:***  
- Using **[** / **]** allows you to switch between saved viewpoints and add features to them. If edit database is innitially selected, you may edit viewpoints from the database
- You are required to choose a feature on the right view before selecting a corresponding 3D point on the left view
- Only one database is nativly saved per map. You may backup a database by finding it in feature_dbs\"map name"\active_sift.npz

## Changing environment parameters
### Changing avilable maps

Each map is comprised of 2 images: A **height map** and a **color map**, and 3 additional parameters: **margin**, **map_scale** and **blur_sigma**.  
- **height map**: A grayscale map were black (#000000) is the lowest point and white (#ffffff) is the highest point
- **color map**: A colored image which will be applied on top of the terrein created from the height map
- **margin**: Distance between pixels sampled to create the terrein. A margin of 1 means every pixel is used
- **map_scale**: Height in units of the highest points. the rest of the points are scaled accordingly
- **blur_sigma**: Strength of gaussian blur applied to height map

#### Adding a new map
At the top of world_split_v5.22.py there is a dictinary called `MAP_PROFILES`. Each entry in it is of form:
>key: map name - string  
>value: map profile  

where a map profile is a dictionary with the following:  
| key             | value              | type   |
| --------------- | ------------------ | -------|
| `"height_path"` | path to height map | string |
| `"color_path"`  | path to color map  | string |
| `"margin"`      | margin             | int    |
| `"map_scale"`   | map_scale          | float  |
| `"blur_sigma"`  | blur_sigma         | float  |  

To add a new map, add a new entry to `MAP_PROFILES`. It will be added to all menus automaticly

***NOTES:***
- If color map is of a different size than the height map, it will be automaticly resized
- Changing any of the parameters will likley make its respective map's database to not be relevent for it


### Editing trackers
Trackers are defines using 6 parameters. 3 for position (*xyz*) and 3 for color (0≤*rgb*≤255). 
A trackers file contains line where each line defines one tracker and is of format `x, y, z, r, g, b` (lines starting with "#" are ignored). 
To use a new trackers file, open the `CONFIG` file and change the `trackers_path` line to your file's path.

***Notes:***
- Using trackers with similar colors to each other or the map might result in poor estimations
- Using less than 4 trackers will not allow any estimation to be returned


### Changing the starting position
The camera will start at the facing the middle of the map (on the x axis) and 100 units back from the origin (on th z axis) with the hrizontal starting angle being 0. Two parameters are changable:
- Starting height (on the y axis)
- Vertical starting angle (up/down)  
To change these parameters, open the `CONFIG` file and edit the lines `start_h` (for starting height) or `start_a` (for starting vertical angle).

***Notes:***
- The y axis for starting height is inverted (positive is down), meaning for starting 100 units up `start_h` should be -100
- For starting vertical angle, rotation is downward and is given in degrees (i. e. `start_a:90` will start the camera looking straight down)

