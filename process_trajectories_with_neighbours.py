#!/usr/bin/env python
import numpy as np
import scipy.io as sio
from prepare_data import create_dataset_with_all_neighbours_and_t
from move_pedestrian_tracks_to_centre import My_GridWorld
from sklearn.preprocessing import scale
from rotate_trajectory import Rotate2D
import scipy
#d=sio.loadmat('../filtered_spline_data.mat') #---USE THIS FOR EIF DATA----#
#d=sio.loadmat('../filtered_GC_data.mat')      #---USE THIS FOR GC DATA----#
# dataset_name='deathCircle'
# path='../stanford_Drone_Dataset/'
# file_name=path+'%s_processed_tracks-(ROTATED).mat'%dataset_name
# save_file_name=path+'%s_processed_tracks-with_neighbours'%dataset_name
# d=sio.loadmat(file_name)     #---USE THIS FOR stanford DATA----#

dataset_name='InD'
path='./InD_data/'
#file_name=path+'filtered_%s_data.mat'%dataset_name
save_file_name=path+'%s_raw_tracks-with_neighbours'%dataset_name
#d=sio.loadmat(file_name)     #---USE THIS FOR stanford DATA----#
data = open("inD_test.pkl","rb")
clustering_x=data[:,:,0]
clustering_y=data[:,:,1]
clustering_t=data[:,:,2]
 
print(clustering_t.shape)
print(clustering_x.shape)
print(clustering_y.shape)
 
clustering_x=np.expand_dims(clustering_x, axis=-1)
clustering_y=np.expand_dims(clustering_y, axis=-1)
clustering_t=np.expand_dims(clustering_t, axis=-1)
# 
main_mode_data=np.concatenate((clustering_x,clustering_y,clustering_t),axis=-1)
print(main_mode_data.shape)

secondary_mode=None
neighbours=10
total_track_len=clustering_x.shape[1]
secondary_neighbours=0
hidden_dim=400

print('finding the neighbours and concatenating them to numpy matrix')
(selected_x_all,selected_y_all,front_x_all,front_y_all,left_x_all,left_y_all,right_x_all,right_y_all,weights_front_all,weights_left_all,weights_right_all,time_all) = create_dataset_with_all_neighbours_and_t(main_mode_data, secondary_mode, neighbours, secondary_neighbours, secondary_neighbours)
# 
for i in range(len(selected_x_all)):
    track_x=selected_x_all[i]
    track_y=selected_y_all[i]
    front_x=front_x_all[i]
    front_y=front_y_all[i]
    left_x=left_x_all[i]
    left_y=left_y_all[i]
    right_x=right_x_all[i]
    right_y=right_y_all[i]
    front_w=weights_front_all[i]
    left_w=weights_left_all[i]
    right_w=weights_right_all[i]
     
    track_x=np.expand_dims(track_x, axis=0)
    track_y=np.expand_dims(track_y, axis=0)
    
    print('--------- track shape --------')
    print(track_x.shape)
#     
    if (track_x.shape[-1] != total_track_len):
        track_x=np.reshape(track_x, (1, total_track_len))
        track_y=np.reshape(track_y, (1, total_track_len))
        
    
    for j in range(len(front_x)):
        f_x=front_x[j]
        f_y=front_y[j]
        l_x=left_x[j]
        l_y=left_y[j]
        r_x=right_x[j]
        r_y=right_y[j]
        f_w= front_w[j]
        l_w= left_w[j]
        r_w= right_w[j]
        
        f_x=np.asarray(f_x)
        f_x=np.expand_dims(f_x, axis=-1)
        f_y=np.asarray(f_y)
        f_y=np.expand_dims(f_y, axis=-1)
        l_x=np.asarray(l_x)
        l_x=np.expand_dims(l_x, axis=-1)
        l_y=np.asarray(l_y)
        l_y=np.expand_dims(l_y, axis=-1)
        r_x=np.asarray(r_x)
        r_x=np.expand_dims(r_x, axis=-1)
        r_y=np.asarray(r_y)
        r_y=np.expand_dims(r_y, axis=-1)
        
        f_w=np.asarray(f_w)
        f_w=np.expand_dims(f_w, axis=-1)
        l_w=np.asarray(l_w)
        l_w=np.expand_dims(l_w, axis=-1)
        r_w=np.asarray(r_w)
        r_w=np.expand_dims(r_w, axis=-1)
        
        
        if j==0:
            f_x_all=f_x
            f_y_all=f_y
            l_x_all=l_x
            l_y_all=l_y
            r_x_all=r_x
            r_y_all=r_y
            f_w_all=f_w
            l_w_all=l_w
            r_w_all=r_w
        else:
            f_x_all=np.concatenate((f_x_all,f_x),axis=-1)
            f_y_all=np.concatenate((f_y_all,f_y),axis=-1)
            l_x_all=np.concatenate((l_x_all,l_x),axis=-1)
            l_y_all=np.concatenate((l_y_all,l_y),axis=-1)
            r_x_all=np.concatenate((r_x_all,r_x),axis=-1)
            r_y_all=np.concatenate((r_y_all,r_y),axis=-1)
            
            f_w_all=np.concatenate((f_w_all,f_w),axis=-1)
            l_w_all=np.concatenate((l_w_all,l_w),axis=-1)
            r_w_all=np.concatenate((r_w_all,r_w),axis=-1)
            
    f_x_all=np.expand_dims(f_x_all, axis=0)
    f_y_all=np.expand_dims(f_y_all, axis=0)
    l_x_all=np.expand_dims(l_x_all, axis=0)
    l_y_all=np.expand_dims(l_y_all, axis=0)
    r_x_all=np.expand_dims(r_x_all, axis=0)
    r_y_all=np.expand_dims(r_y_all, axis=0)
    
    f_w_all=np.expand_dims(f_w_all, axis=0)
    l_w_all=np.expand_dims(l_w_all, axis=0)
    r_w_all=np.expand_dims(r_w_all, axis=0)
    
    f_w_all = np.repeat(np.expand_dims(f_w_all, axis=3), hidden_dim, axis=3)
    l_w_all = np.repeat(np.expand_dims(l_w_all, axis=3), hidden_dim, axis=3)
    r_w_all = np.repeat(np.expand_dims(r_w_all, axis=3), hidden_dim, axis=3)
    
    
    #---------- concat data to create a large matrix for saving --------#
    if i==0:
        all_track_x=track_x
        all_track_y=track_y
    else:
        
        all_track_x=np.concatenate((all_track_x,track_x),axis=0)
        all_track_y=np.concatenate((all_track_y,track_y),axis=0)
    
    if i==0:
        all_f_x= f_x_all
        all_f_y= f_y_all
        all_l_x= l_x_all
        all_l_y= l_y_all
        all_r_x= r_x_all
        all_r_y= r_y_all
        
        all_f_w= f_w_all
        all_l_w= l_w_all
        all_r_w= r_w_all
        
    else:
        all_f_x= np.concatenate((all_f_x,f_x_all), axis=0)
        all_f_y= np.concatenate((all_f_y,f_y_all), axis=0)
        all_l_x= np.concatenate((all_l_x,l_x_all), axis=0)
        all_l_y= np.concatenate((all_l_y,l_y_all), axis=0)
        all_r_x= np.concatenate((all_r_x,r_x_all), axis=0)
        all_r_y= np.concatenate((all_r_y,r_y_all), axis=0)
        
        all_f_w= np.concatenate((all_f_w,f_w_all), axis=0)
        all_l_w= np.concatenate((all_l_w,l_w_all), axis=0)
        all_r_w= np.concatenate((all_r_w,r_w_all), axis=0)
     
     print('--------- all shapes --------')
     print(all_track_x.shape)
     print(all_track_y.shape)
     print(all_l_x.shape)
    print(all_l_y.shape)
     print(all_r_x.shape)
     print(all_f_y.shape)
     
     print(all_f_w.shape)
     print(all_l_w.shape)
     print(all_r_w.shape)
     
 sio.savemat('temp_xx.mat',
             {'selected_x_all':all_track_x,'selected_y_all':all_track_y,'front_x_all':all_f_x,
              'front_y_all':all_f_y,'left_x_all':all_l_x,'left_y_all':all_l_y,'right_x_all':all_r_x,
              'right_y_all':all_f_y,'all_f_w':all_f_w,'all_l_w':all_l_w,
              'all_r_w':all_r_w})

d=sio.loadmat('temp_xx.mat')
all_track_x=d['selected_x_all']
total_track_len=all_track_x.shape[1]
all_track_y=d['selected_y_all']
all_f_x=d['front_x_all']
all_f_y=d['front_y_all']
all_l_x=d['left_x_all']
all_l_y=d['left_y_all']
all_r_x=d['right_x_all']
all_r_y=d['right_y_all']
all_f_w=d['all_f_w']
all_l_w=d['all_l_w']
all_r_w=d['all_r_w']
       
grid_map = np.zeros([80, 80])
gw=My_GridWorld(grid_map,{})

print('shifting the trajectories to the grid centre.......')

for i in range(all_track_x.shape[0]):
    traj=np.concatenate((np.expand_dims(all_track_x[i,:],axis=-1),np.expand_dims(all_track_y[i,:],axis=-1)),axis=-1)
    front_neighbours=np.concatenate((np.expand_dims(all_f_x[0,:,:],axis=-1),np.expand_dims(all_f_y[0,:,:],axis=-1)),axis=-1)
    left_neighbours=np.concatenate((np.expand_dims(all_l_x[0,:,:],axis=-1),np.expand_dims(all_l_y[0,:,:],axis=-1)),axis=-1)
    right_neighbours=np.concatenate((np.expand_dims(all_r_x[0,:,:],axis=-1),np.expand_dims(all_r_y[0,:,:],axis=-1)),axis=-1)
    
    print(front_neighbours.shape)
    
    traj_new,idx,f_new=gw.move_future_start_point_to_grid_middle(traj,front_neighbours,considering_len=int(total_track_len/2))
    _,_,l_new=gw.move_future_start_point_to_grid_middle(traj,left_neighbours,considering_len=int(total_track_len/2))
    _,_,r_new=gw.move_future_start_point_to_grid_middle(traj,right_neighbours,considering_len=int(total_track_len/2))
    
    # #--- Rotate the trajectories for data augmentation---
    # no_neighbours=front_neighbours.shape[1] #neighbours --> ( time_steps, no_neighbours)
    # rotation_ang=2*scipy.pi/np.random.randint(1, high=10)
    # rotated_traj_new = Rotate2D(np.asarray(traj_new),ang=rotation_ang)
    # rotated_front_neighbours=np.zeros(front_neighbours.shape)
    # rotated_left_neighbours=np.zeros(left_neighbours.shape)
    # rotated_right_neighbours=np.zeros(right_neighbours.shape)
    # for k in range(no_neighbours):
    #     rotated_front_neighbours[k,:,:] = Rotate2D(front_neighbours[k,:,:],ang=rotation_ang)
    #     rotated_left_neighbours[k,:,:] = Rotate2D(left_neighbours[k,:,:],ang=rotation_ang)
    #     rotated_right_neighbours[k,:,:] = Rotate2D(right_neighbours[k,:,:],ang=rotation_ang)
     
    traj_new=np.asarray(traj_new)
    traj_new=np.expand_dims(traj_new,axis=0)
    f_new=np.expand_dims(f_new,axis=0)
    l_new=np.expand_dims(l_new,axis=0)
    r_new=np.expand_dims(r_new,axis=0)
    
    # rotated_traj_new=np.expand_dims(rotated_traj_new,axis=0)
    # rotate_f_new=np.expand_dims(rotated_front_neighbours,axis=0)
    # rotate_l_new=np.expand_dims(rotated_left_neighbours,axis=0)
    # rotate_r_new=np.expand_dims(rotated_right_neighbours,axis=0)
    # 
    # #print('rotate_r_new'+ str(rotate_r_new.shape))
    # #print('r_new'+ str(r_new.shape))
      
    
    if i==0:
        selected_x_all=traj_new[:,:,0]
        selected_y_all=traj_new[:,:,1]
        front_x_all=f_new[:,:,:,0]
        front_y_all=f_new[:,:,:,1]
        left_x_all=l_new[:,:,:,0]
        left_y_all=l_new[:,:,:,1]
        right_x_all=r_new[:,:,:,0]
        right_y_all=r_new[:,:,:,1]
        # we need to concatenate this again becuase we need to get the weights for the rotate data as well
        front_w_all=np.expand_dims(all_f_w[i,:,:,:],axis=0)
        left_w_all=np.expand_dims(all_l_w[i,:,:,:],axis=0)
        right_w_all=np.expand_dims(all_r_w[i,:,:,:],axis=0)
        
    else:
        selected_x_all=np.concatenate((selected_x_all,traj_new[:,:,0]),axis=0)
        #selected_x_all=np.concatenate((selected_x_all,rotated_traj_new[:,:,0]),axis=0) # add the rotated_stuff
        selected_y_all=np.concatenate((selected_y_all,traj_new[:,:,1]),axis=0)
        #selected_y_all=np.concatenate((selected_y_all,rotated_traj_new[:,:,1]),axis=0) # add the rotated_stuff
        front_x_all=np.concatenate((front_x_all,f_new[:,:,:,0]),axis=0)
        #front_x_all=np.concatenate((front_x_all,rotate_f_new[:,:,:,0]),axis=0) # add the rotated_stuff
        front_y_all=np.concatenate((front_y_all,f_new[:,:,:,1]),axis=0)
        #front_y_all=np.concatenate((front_y_all,rotate_f_new[:,:,:,1]),axis=0) # add the rotated_stuff
        left_x_all=np.concatenate((left_x_all,l_new[:,:,:,0]),axis=0)
        #left_x_all=np.concatenate((left_x_all,rotate_l_new[:,:,:,0]),axis=0) # add the rotated_stuff
        left_y_all=np.concatenate((left_y_all,l_new[:,:,:,1]),axis=0)
        #left_y_all=np.concatenate((left_y_all,rotate_l_new[:,:,:,1]),axis=0) # add the rotated_stuff
        right_x_all=np.concatenate((right_x_all,r_new[:,:,:,0]),axis=0)
        #right_x_all=np.concatenate((right_x_all,rotate_r_new[:,:,:,0]),axis=0) # add the rotated_stuff
        right_y_all=np.concatenate((right_y_all,r_new[:,:,:,1]),axis=0)
        #right_y_all=np.concatenate((right_y_all,rotate_r_new[:,:,:,1]),axis=0) # add the rotated_stuff
        
        #---- we concatenate them 2 times, one for the original one for the rotated, beacuase the distances are the same#
        front_w_all=np.concatenate((front_w_all,np.expand_dims(all_f_w[i,:,:,:],axis=0)),axis=0)
        #front_w_all=np.concatenate((front_w_all,np.expand_dims(all_f_w[i,:,:,:],axis=0)),axis=0)  # for the rotated_stuff
        left_w_all=np.concatenate((left_w_all,np.expand_dims(all_l_w[i,:,:,:],axis=0)),axis=0)
        #left_w_all=np.concatenate((left_w_all,np.expand_dims(all_l_w[i,:,:,:],axis=0)),axis=0) # for the rotated_stuff
        right_w_all=np.concatenate((right_w_all,np.expand_dims(all_r_w[i,:,:,:],axis=0)),axis=0)
        #right_w_all=np.concatenate((right_w_all,np.expand_dims(all_r_w[i,:,:,:],axis=0)),axis=0) # for the rotated_stuff
        
    print('--------- all shapes --------')
    print(selected_x_all.shape)
    print(selected_y_all.shape)
    print(front_x_all.shape)
    print(front_y_all.shape)
    print(left_x_all.shape)
    print(right_x_all.shape)
    print(front_w_all.shape)
    print(left_w_all.shape)
    print(right_w_all.shape)
           
sio.savemat(save_file_name,
            {'selected_x_all':selected_x_all,'selected_y_all':selected_y_all,'front_x_all':front_x_all,
             'front_y_all':front_y_all,'left_x_all':left_x_all,'left_y_all':left_y_all,'right_x_all':right_x_all,
             'right_y_all':right_y_all,'all_f_w':front_w_all,'all_l_w':left_w_all,
             'all_r_w':right_w_all})