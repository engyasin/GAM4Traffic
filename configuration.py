
# Configuration parameters of sdd training

class config:

    #repeated in highway env
    grid_size = [[-6,6],[-4,20]] 

    #action space
    action_size = [[-18,102],[-60,60]] #x_range,y_range

    # input , output steps
    sample_in_out = (8,1)

    # number of features [presense, x,y,vx,vy,heading]
    n_features = 6

    # vehicle count
    vehicle_count = 5

    # vel range
    speed_range = [-120,120]

    #scale
    scale = 10

    grad_clip = 0.5


    # distance factor of the reward
    distance_factor = 1/500