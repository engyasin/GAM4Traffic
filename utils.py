
from configuration import config
import torch
import numpy as np

device = ['cpu','cuda'][torch.cuda.is_available()]

x_ranges = torch.Tensor([range(config.action_size[0][0],config.action_size[0][1])]*
                    (config.action_size[1][1]-config.action_size[1][0])).to(device).unsqueeze(-1)
y_ranges = torch.Tensor([range(config.action_size[1][0],config.action_size[1][1])]*
                    (config.action_size[0][1]-config.action_size[0][0])).T.to(device).unsqueeze(-1)

def make_label(acts,stdxy):

    # vxx in range(1,13)
    # std
    batch = acts.shape[0]
    xs_ = x_ranges.repeat(1,1,batch)
    ys_ = y_ranges.repeat(1,1,batch)

    vxx = stdxy[:,0]/5
    vyy = stdxy[:,1]/5


    mu = acts.int().clone()#torch.clip(acts,min=0,max=120).int()
    #mu += torch.Tensor([config.action_size[0][0],config.action_size[1][0]]).int().to(device)


    gauss_x = torch.exp(-(xs_-mu[:,0])**2/(2*vxx**2))/(vxx * np.sqrt(2 * np.pi))
    gauss_y = torch.exp(-(ys_-mu[:,1])**2/(2*vyy**2))/(vyy * np.sqrt(2 * np.pi))

    label = (gauss_x*gauss_y).permute(2,1,0)# batch,120,120

    # add some small random noise
    label += (torch.rand_like(label)/(1e6)).to(device)
    #label += (np.random.rand(*label.shape)/100)
    label /= label.amax(dim=(1,2)).unsqueeze(1).unsqueeze(1)
    #breakpoint()
    return label


def FDE():
    pass

def AFDE():
    pass


def get_agent_num(all_agents):

    ped_num = np.random.randint(5,all_agents)
    cyc_num = int(np.ceil((all_agents - ped_num)*2/3.0))
    car_num = all_agents-(ped_num+cyc_num)
    #agent_numbers = [2*factor,4*factor,10*factor]
    agent_numbers = [car_num,cyc_num,ped_num]

    return agent_numbers


def read_sdd_annotation(filename='',first_frame=0):
    """
    Read text files of the  Stanford drone dataset format
    Input: file name

    Return:
    Dict for the data
    Frames index of first apperance of each object
    """
    tracks = {}
    initial_frames = []
    #segemnted_ids = []
    vid_length = 0
    with open(filename,mode='r') as f:
        line = f.readline()
        while line:
            data = line.split(' ')
            track_id = int(data[0])
            if int(data[5])<first_frame:
                continue
            #while track_id in segemnted_ids:
            #    track_id += 1e3
            sample = {'bbox':[int(data[1]),int(data[3]),int(data[2]),int(data[4])], # xmin,xmax, ymin, ymax
                      'frame':int(data[5]),
                      'lost':int(data[6]),
                      'occluded':int(data[7]),
                      'generated':int(data[8]),
                      'label':(data[9][:-1]).strip('"'),
                      'heading':None,
                      'center':[(int(data[1])+int(data[3]))/2,(int(data[2])+int(data[4]))/2],
                      'goal_classes':[int(data[4])-int(data[2]),int(data[3])-int(data[1])],}#w,h
            
            line = f.readline()

            #if sample['generated']:# or sample['lost'] or sample['occluded']:
                #bad sample
            #    print(sample['generated'])
            #    segemnted_ids.append(track_id)
            #    continue

            if track_id in tracks.keys():
                # no missing frames
                missing_ = sample['frame']-(tracks[track_id][-1]['frame'])-1
                tracks[track_id].extend([{} for _ in range(missing_)]+[sample])
            else:
                tracks[track_id] = [sample]
                initial_frames.append(sample['frame'])


    # direction every 1 second for ped. 0.5 for others
    for t_id,track in tracks.items():
        shift = [15,30]['Pedestrian' in track[0]['label']]
        vid_length = max(track[-1]['frame'],vid_length)
        for i,sample in enumerate(track):
            if len(sample):
                local_dest = min(i+shift,len(track)-1)
                dest = min(i+144+6,len(track)-1) # 144f=4.8 seconds in the future
                if len(track[local_dest]):
                    c2 = [track[local_dest]['bbox'][1]+track[local_dest]['bbox'][0],
                          track[local_dest]['bbox'][3]+track[local_dest]['bbox'][2]]
                    c1 = [sample['bbox'][1]+sample['bbox'][0],
                          sample['bbox'][3]+sample['bbox'][2]]
                    goal = [(track[dest]['bbox'][1]+track[dest]['bbox'][0])/2,
                          (track[dest]['bbox'][3]+track[dest]['bbox'][2])/2]
                    class_ = []# w,h
                    heading = np.arctan2(c2[1]-c1[1],c2[0]-c1[0])
                    tracks[t_id][i].update({'heading':(2*np.pi+heading)%(2*np.pi)})
                    tracks[t_id][i]['goal_classes'].extend(goal)

    return tracks,initial_frames,vid_length

def bbox_to_cnt_wh(bbox,scale=1,origin=(0,0)):
    """Convert [xmin,xmax,ymin,ymax] to center and w,h

    """
    #sdd has bigger objects ==> minimize
    w = 0.6*((bbox[1]-bbox[0])/scale) #xs
    h = 0.6*((bbox[3]-bbox[2])/scale) #ys
    cnt = [((bbox[1]+bbox[0])/2)-origin[0],
           ((bbox[2]+bbox[3])/2)-origin[1]]
    cnt = [cnt[0]/scale,cnt[1]/scale]

    return cnt,w,h


#  train_sample['label'].max(dim=2)[0].max(dim=2)[1] # batch,1 # y=raws
#  train_sample['label'].max(dim=3)[0].max(dim=2)[1]

def from_2d_2_xy(label):

    ys = label.max(dim=2)[0].max(dim=2)[1] # batch,1 # y=raws
    xs = label.max(dim=3)[0].max(dim=2)[1]

    return torch.hstack((xs,ys))

def read_sdd(filename=''):
    """
    Read text files of the  Stanford drone dataset format
    Input: file name

    Return:
    Dict for the data
    Frames index of first apperance of each object
    """
    tracks = {}
    with open(filename,mode='r') as f:
        line = f.readline()
        while line:
            data = line.split(' ')
            track_id = int(data[0])
            sample = {'bbox':[int(data[1]),int(data[3]),int(data[2]),int(data[4])], # xmin,xmax, ymin, ymax
                      'frame':int(data[5]),
                      'lost':int(data[6]),
                      'occluded':int(data[7]),
                      'generated':int(data[8]),
                      'label':(data[9][:-1]).strip('"'),
                      'center':[(int(data[1])+int(data[3]))/2,(int(data[2])+int(data[4]))/2],
                      'goal_classes':[int(data[4])-int(data[2]),int(data[3])-int(data[1])],}#w,h
            
            line = f.readline()

            if track_id in tracks.keys():
                # no missing frames
                missing_ = sample['frame']-(tracks[track_id][-1]['frame'])-1
                tracks[track_id].extend([{} for _ in range(missing_)]+[sample])
            else:
                tracks[track_id] = [sample]

    return tracks

def tracks_to_array(tracks):
    # direction every 1 second for ped. 0.5 for others
    step = int(30*0.4)
    Xs,ys= [],[]
    type_code = {'Pedestrian':0, 'Car':5,'Bus':4, 'Biker':2, 'Cart':3, 'Skater':1}
    for _,track in tracks.items():
        # Pedestrian, Car, Bike, Cart, Skater
        if (len(track)<240): continue
        n_samples = (len(track)-240+1)//step
        for i in range(n_samples):
            # subsample frames
            losts, occluded, generated = 0,0,0
            all_steps_8 = []
            all_widths_8 = []
            for x in range(8):
                all_steps_8.append(track[(i*step)+(step*x)]['center'])
                all_widths_8.append(track[(i*step)+(step*x)]['goal_classes'])
                losts += track[(i*step)+(step*x)]['lost']
                occluded += track[(i*step)+(step*x)]['occluded']
                generated += track[(i*step)+(step*x)]['generated']
                
            all_steps_in = np.array(all_steps_8)
            all_widths_in = np.array(all_widths_8)
            step20 = np.array(track[(i*step)+(step*19)]['center'])#,2
            c1,c2 = all_steps_in[6,:],all_steps_in[7,:].copy()
            heading = np.arctan2(c2[1]-c1[1],c2[0]-c1[0])
            c, s = np.cos(heading), np.sin(heading)
            R_mat = np.array([[c, s], [-s, c]])
            #tranform
            all_steps_in -= c2
            step20 -= c2
            input_vec = (R_mat@ all_steps_in.T).T.flatten()[:14] #7,2==>14

            step20 = R_mat@ step20
            input_others = np.array([losts,occluded,generated]+[type_code[track[0]['label']]])
            # (track[(i*step)+(step*7)]['goal_classes'])+
            Xs.append(np.hstack((input_vec,all_widths_in.flatten(),input_others)))
            ys.append(step20)

    return np.array(Xs),np.array(ys)
 


def annotation2array(filename='',first_frame=0):
    """
    Read text files of the  Stanford drone dataset format
    Input: file name

    Return:
    Dict for the data
    Frames index of first apperance of each object
    """
    tracks = {}
    initial_frames = []
    #segemnted_ids = []
    vid_length = 0
    with open(filename,mode='r') as f:
        line = f.readline()
        while line:
            data = line.split(' ')
            track_id = int(data[0])
            if int(data[5])<first_frame:
                continue
            #while track_id in segemnted_ids:
            #    track_id += 1e3
            sample = {'bbox':[int(data[1]),int(data[3]),int(data[2]),int(data[4])], # xmin,xmax, ymin, ymax
                      'frame':int(data[5]),
                      'lost':int(data[6]),
                      'occluded':int(data[7]),
                      'generated':int(data[8]),
                      'label':(data[9][:-1]).strip('"'),
                      'heading':None,
                      'center':[(int(data[1])+int(data[3]))/2,(int(data[2])+int(data[4]))/2],
                      'goal_classes':[int(data[4])-int(data[2]),int(data[3])-int(data[1])],}#w,h
            
            line = f.readline()

            #if sample['generated']:# or sample['lost'] or sample['occluded']:
                #bad sample
            #    print(sample['generated'])
            #    segemnted_ids.append(track_id)
            #    continue

            if track_id in tracks.keys():
                # no missing frames
                missing_ = sample['frame']-(tracks[track_id][-1]['frame'])-1
                tracks[track_id].extend([{} for _ in range(missing_)]+[sample])
            else:
                tracks[track_id] = [sample]
                initial_frames.append(sample['frame'])


    # direction every 1 second for ped. 0.5 for others
    step = int(30*0.4)
    Xs,ys,Nx = [],[],[]
    # Nx has: [height of traj(x), width, mean angels, std angels]
    for t_id,track in tracks.items():
        if ('Pedestrian' not in track[0]['label']) or (len(track)<240): #or 240-11#Pedestrian, Bike
            # less than evaluation creteria of 8 s
            continue
        n_samples = (len(track)-240+1)//step
        # row_x = [x1,y1, .. x7,y7, lost,occ, gen, width, height]
        # row_y = [x20,y20] (cluster later: manually)
        for i in range(n_samples):
            # subsample frames
            losts, occluded, generated = 0,0,0
            all_steps_8 = []
            for x in range(8):
                all_steps_8.append(track[(i*step)+(step*x)]['center'])
                losts += track[(i*step)+(step*x)]['lost']
                occluded += track[(i*step)+(step*x)]['occluded']
                generated += track[(i*step)+(step*x)]['generated']
            meta_f = [losts,occluded,generated]
            all_steps_in = np.array(all_steps_8)#[track[(i*step)+(step*x)]['center'] for x in range(8)])
            step20 = np.array(track[(i*step)+(step*19)]['center'])#,2
            #full_ys = [track[(i*step)+(step*x)]['center'] for x in range(8,20)])
            step8 = track[(i*step)+(step*7)]#,2
            c1,c2 = all_steps_in[6,:],all_steps_in[7,:].copy()
            heading = np.arctan2(c2[1]-c1[1],c2[0]-c1[0])
            c, s = np.cos(heading), np.sin(heading)
            R_mat = np.array([[c, s], [-s, c]])
            #tranform
            all_steps_in -= c2#all_steps_in[7,:] # 8,2
            step20 -= c2#all_steps_in[7,:] # ,2
            input_vec = (R_mat@ all_steps_in.T).T #7,2==>14
            #directions = np.diff(input_vec,axis=0)
            #angles = abs(np.arctan2(directions[:,0],directions[:,1]))
            #meta_f = [step8['lost'],step8['occluded'],step8['generated']]
            #Nx.append((abs(input_vec[0,:])).tolist()+[angles.mean(),angles.std()]+meta_f)
            step20 = R_mat@ step20
            input_others = np.array(meta_f+step8['goal_classes'])
            input_vec_ = input_vec[:7,:].flatten()
            Xs.append(np.append(input_vec_,input_others))
            ys.append(step20)
            if input_vec_.any():
                breakpoint()

    return np.array(Xs),np.array(ys)#,np.array(Nx)
    #return tracks,initial_frames,vid_length