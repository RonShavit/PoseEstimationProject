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

### Running the Project
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
#### Modes and controls
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
- By defualt, 8 trackers are included. See ***TODO*** to learn how to change the trackers