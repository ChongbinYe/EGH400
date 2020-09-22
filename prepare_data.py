#
#	Data tools for multi-modal trajectory prediction. This module contains functions to:
#		- load data
#		- prepare data for processing
#		- visualise data
#
#	Change Log:
#		27/01/2018:			Initial Version (SD), based on code from Tharindu
#		28/01/2018 (SD):	Works for a single mode now
#		30/01/2018 (SD):	Add command line options, add hooks for multi-modal
#							version
#		03/02/2018 (SD):	Mulit-model extraction added. Removed "threshold"
#							time-shift like part for now
#
#

import loader
import numpy as np
import scipy.io as sio
import math
import argparse
import pickle
#from plot_trajectories import plot_trajectory_with_neighbours

#
# determine if traj_1 is in front of traj_2, if so, return True, else return False
#
def in_front_of(traj_1_x, traj_1_y, traj_2_x, traj_2_y):
	# if traj_2 is in front, then AB < AM, and BM < AM
	#	A is the first point of traj_1
	#	B is the last point of traj_1
	#	M is the last point of traj_2
	# will be positive for points on one side and negative for points on the other
	
	# pull out the points we need, do this to make it clearer
	Ax = traj_1_x[0]
	Ay = traj_1_y[0]
	Bx = traj_1_x[-1]
	By = traj_1_y[-1]
	X = traj_2_x[-1]
	Y = traj_2_y[-1]

	AB = pow(Ax - Bx, 2.0) + pow(Ay - By, 2.0)
	AM = pow(Ax - X, 2.0) + pow(Ay - Y, 2.0)
	BM = pow(Bx - X, 2.0) + pow(By - Y, 2.0)

	if ((AB < AM) & (BM < AM)):
		return True
	else:
		return False
#
# determine if one trajectory is to the left of another. When determining this we
#	- consider only the main direction of motion of traj_1, i.e. just the first and last point
#	- consider only the last position of traj_2, i.e. is it's last point to the left
#
def to_left_of(traj_1_x, traj_1_y, traj_2_x, traj_2_y):
	# sign of the determinant of the vectors AB and AM.
	#	A is the first point of traj_1
	#	B is the last point of traj_1
	#	M is the last point of traj_2
	# will be positive for points on one side and negative for points on the other
	
	# pull out the points we need, do this to make it clearer
	Ax = traj_1_x[0]
	Ay = traj_1_y[0]
	Bx = traj_1_x[-1]
	By = traj_1_y[-1]
	X = traj_2_x[-1]
	Y = traj_2_y[-1]

	position = np.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
	if (position > 1):
		return True;

#
# determine if a trajectory is to the right of another. 
# Function just calls to_left_of and inverts the result. NOTE: This means that trajectories
# that lie exactly on the path of traj_1 will be classed as being to the right of. We're going
# to assume that cases of this happening will be very rare at most.
#
def to_right_of(traj_1_x, traj_1_y, traj_2_x, traj_2_y):
	if (is_left_of(traj_1_x, traj_1_y, traj_2_x, traj_2_y) == True):
		return False;
	else:
		return True;
	
#
# split up a set of neighbouring trajectories according to whether they are to the left, right, or in front of a taget
#
def split_neighbours(traj_of_interest_x, traj_of_interest_y, neighbours_x, neighbours_y):
	front_x = np.zeros(neighbours_x.shape)
	front_y = np.zeros(neighbours_x.shape)
	left_x = np.zeros(neighbours_x.shape)
	left_y = np.zeros(neighbours_x.shape)
	right_x = np.zeros(neighbours_x.shape)
	right_y = np.zeros(neighbours_x.shape)
	front_idx = 0
	left_idx = 0
	right_idx = 0
	# iterate through neighbours
	for i in range(neighbours_x.shape[0]):
		# check front. Need to check front first as all traj will be either left
		# or right
		if in_front_of(traj_of_interest_x, traj_of_interest_y, neighbours_x[i, :], neighbours_y[i, :]):
			front_x[front_idx,:] = neighbours_x[i, :]
			front_y[front_idx,:] = neighbours_y[i, :]
			front_idx += 1
		# check left
		elif to_left_of(traj_of_interest_x, traj_of_interest_y, neighbours_x[i, :], neighbours_y[i, :]):
			left_x[left_idx,:] = neighbours_x[i, :]
			left_y[left_idx,:] = neighbours_y[i, :]
			left_idx += 1
		# if not front and left, must be right
		else:
			right_x[right_idx,:] = neighbours_x[i, :]
			right_y[right_idx,:] = neighbours_y[i, :]
			right_idx += 1
	
	return front_x, front_y, front_idx, left_x, left_y, left_idx, right_x, right_y, right_idx

#
# load data
# This loads the file using the c++/python loader, and then creates trajectories of the target length from the data
# Will extract sequences of a target length, and down-sample by a given factor as well. The downsample is used to 
# allow the network (defined elsewhere) to predict/model longer trajectories without needing to increase the network
# size
#
# Limitions:
#	At the moment this does not consider a sliding window when breaking up trajectories, this could be used to get more data
#	
def load_data(file_path, seq_length=50, downsmaple_factor=5, source = 0, offset = 0):
	# call the c++/python loader to load the file, this loads the data outputted
	# by c++ and puts into python structures. 
	# Note for that different source data, this would need to change.
    if (source == 0):
        traj_list = loader.load_cplusplus_trajectories(file_path)    
    else:
        traj_list = loader.load_python_trajectories(file_path)

    data_all=[]
    first_done=False

	# downsample trajectories
    traj_list_new=[]
    for i in range(len(traj_list)):
            traj=traj_list[i]
            traj_new=[]
            for j in range(len(traj)):
            	if j % downsmaple_factor == 0:
            		traj_new.append(traj[j])
            traj_list_new.append(traj_new)                
    
    traj_list=traj_list_new
    
	# loop through the downsampled trajectory list
	# put things into a giant numpy array, and break trajectories down into segments of 
	# length seq_length (default 50)
    print(traj_list[1])
    print(len(traj_list[1]))
    for i in range(len(traj_list)):
        traj=traj_list[i]
        traj_x=[]
        traj_y=[]
        traj_t=[]
        track_length=len(traj)
        #no_sub_seq=int(math.floor(track_length/seq_length))
        for j in range(len(traj)):
           
            obs=traj[j]
            
            time=obs[0]
            x=obs[1]
            y=obs[2]
            traj_x.append(x)
            traj_y.append(y)
            traj_t.append(time)
            
        
        traj_x=np.asarray(traj_x)
        traj_y=np.asarray(traj_y)
        traj_t=np.asarray(traj_t)
        
        start_idx=offset
        #print((start_idx + seq_length),'...',len(traj),'...',traj)
        while ((start_idx + seq_length) < len(traj)):
#        for j in range(no_sub_seq):
            end_idx = start_idx + seq_length 
            
            sub_track_x=traj_x[start_idx:end_idx]
            sub_track_y=traj_y[start_idx:end_idx]
            sub_track_t=traj_t[start_idx:end_idx]
            
            
            data_out=np.stack((sub_track_x, sub_track_y, sub_track_t),axis=1)
            data_out=np.expand_dims(data_out, axis=0)           

            if first_done== False:
                first_done=True
                data_all=data_out
            else:
                #print('data_all:'+str(data_all.shape))
                data_all=np.concatenate((data_all, data_out),axis=0)
            start_idx = start_idx + seq_length
    
    return data_all


#
# deal with extra neighbours, or pad out the neigbour arrays if their missing a few values
# Two modes are currently defined for this:
#	extra_neighbours == 0:	average the extra ones
#	extra_neighbours == 1:	take the closest of the rest, and ignore the others
#	
def merge_extra_neighbours(neighbour_x, neighbour_y, neighbour_w, num_neighbours = 10, extra_neighbours = 0):
	updated_x = np.zeros([num_neighbours, neighbour_x.shape[0]])
	updated_y = np.zeros([num_neighbours, neighbour_x.shape[0]])
	updated_w = np.full([num_neighbours, neighbour_x.shape[0]], 0.00000000000000000000000000000000000000000000001)
	for i in range(min(neighbour_x.shape[1], num_neighbours - 1)):
		updated_x[i, :] = neighbour_x[:, i]
		updated_y[i, :] = neighbour_y[:, i]
		updated_w[i, :] = neighbour_w[:, i]
	
	# handle extra neighbours
	# option 0: average all the remaining neighbours
	if (neighbour_x.shape[1] >= num_neighbours):
		if (extra_neighbours == 0):
			count = 0
			for i in range(num_neighbours - 1, neighbour_x.shape[1]):
				updated_x[num_neighbours - 1, :] += neighbour_x[:, i]
				updated_y[num_neighbours - 1, :] += neighbour_y[:, i]
				updated_w[num_neighbours - 1, :] += neighbour_w[:, i]
				count += 1
			updated_x[num_neighbours - 1, :] /= count
			updated_y[num_neighbours - 1, :] /= count
			updated_w[num_neighbours - 1, :] /= count
		# option 1 (or at the moment not 0): just take the 10th and ignore the rest
		else:
			updated_x[num_neighbours - 1, :] = neighbour_x[:, num_neighbours - 1]
			updated_y[num_neighbours - 1, :] = neighbour_y[:, num_neighbours - 1]
			updated_w[num_neighbours - 1, :] = neighbour_w[:, num_neighbours - 1]
	
	return updated_x, updated_y, updated_w

#
# calculates distance between the main and all neighbour trajectories
#
def calculate_distance_to_adjecent_trajectories(selected_x, selected_y, adjecent_x, adjecent_y, dummy_value=-50):
    dist=np.zeros(adjecent_x.shape)
   
    #for each trajectory
    for i in range(adjecent_x.shape[1]):
        # for lenght of trajectory
        for j in range(adjecent_x.shape[0]):
            dist[j,i]=np.sqrt((adjecent_x[j,i]-selected_x[j])**2 + (adjecent_y[j,i]-selected_y[j])**2)
                
    dist=np.divide(1.0, dist, out=np.zeros_like(dist), where=dist!= 0)#1/dist
    
    rows,cols=np.where(adjecent_x == dummy_value)
    dist[rows,cols]=0.00000000000000000000000000000000000000000000001
    
    #print('dist:' + str(dist.shape))
    return dist
    
#
# find all trajectories that are temporally adjacent to a trajecroty of interest
# retuns the list of adjacent trajectories as arrays of x and y points
#        
def find_all_adjecent_trajectories(x, y, time, time_selected, selected_idx, dummy_value=-50):
    
    # Create a matrix of size (x,y) and fill it with dummy point(-50,-50) values
    adjecent_x=np.full((time.shape[0],time_selected.shape[0]),dummy_value)
    adjecent_y=np.full((time.shape[0],time_selected.shape[0]),dummy_value)
    
    for i in range(time_selected.shape[0]):
        
        # Find row and column idxs where time is equal to time of the selected trajectory
        rows, cols = np.where(time == time_selected[i])
        #print('rows: '+str(rows.shape))
        
        # Replace the dummy points with the values of those rows and cols
        adjecent_x[rows,cols]=x[rows,cols];
        adjecent_y[rows,cols]=y[rows,cols];
        #print('x: '+str(adjecent_x[rows,cols]))
    
    # The above process also accounts for the selected trajectory
    # Replace the Row of the selected trajectory again with dummy values
    adjecent_x[selected_idx,:]=dummy_value
    adjecent_y[selected_idx,:]=dummy_value
    
    # Find unique rows that have values other than dummy points
    rows,cols=np.where(adjecent_x > dummy_value)
    temp=np.unique(rows)
#    print('No of rows with data: '+str(temp.shape))
#    print(str(temp.shape[0]))
    d=np.zeros(adjecent_x.shape[0])
    
    # Find the rows that have most of the values (i.e max col size) other than dummy points
    for i in range(temp.shape[0]):
        idx=temp[i]
        a=np.where(rows == idx)
        c=cols[a]

        d[idx]=c.shape[0]
    
    ids= np.argsort(d)
    #print(d[ids[(ids.shape[0]-10):]])
    
    if (temp.shape[0] > 0):
    	adjecent_x=adjecent_x[ids[(ids.shape[0]-temp.shape[0]):],:]
    	adjecent_y=adjecent_y[ids[(ids.shape[0]-temp.shape[0]):],:]
    
		#convert shape (10,#time-steps) to (#time-steps,10)
    	adjecent_x=np.transpose( adjecent_x, (1, 0) )
    	adjecent_y=np.transpose( adjecent_y, (1, 0) )
    
    	return adjecent_x,adjecent_y
    else:
    	return None, None	
	
#	
# Create the dataset. This will:
#	- loop through all trajectories. For each trajectory:
#		- find all neighbours
#		- setup neighbour weights
#		- split into left, right, front
#		- ensure that we ahve the correct number of neighbours in each direction
#		- store the results as a dictionary in a list
#
def create_dataset_with_all_neighbours_and_t(main_mode, num_neighbours = 10, extra_neighbours = 0):
	data=[];
	
	x_all = main_mode[:,:,0]
	y_all = main_mode[:,:,1]
	t_all = main_mode[:,:,2]
				
	for i in range(x_all.shape[0]):
		selected_x = x_all[i,:]
		selected_y = y_all[i,:]
		selected_t = t_all[i,:]

		# get adjacent trajectories for the main main
		[adjecent_x, adjecent_y] = find_all_adjecent_trajectories(x_all, y_all, t_all, selected_t, i)
		
		# did we find any? If so, process them
		if (adjecent_x is not None):
						
			# need to split adjacent trajectories into front, left and right
			front_x, front_y, n_f, left_x, left_y, n_l, right_x, right_y, n_r = split_neighbours(selected_x, selected_y, adjecent_x, adjecent_y)
			
			# get distances to trajectories in each direction
			weights_front = calculate_distance_to_adjecent_trajectories(selected_x, selected_y, front_x, front_y)
			weights_left = calculate_distance_to_adjecent_trajectories(selected_x, selected_y, left_x, left_y)
			weights_right = calculate_distance_to_adjecent_trajectories(selected_x, selected_y, right_x, right_y)
			#print('inter1')
			#print(np.shape(left_x))

			# if we have more than max_traj, deal with this. Can either:
			#	- merge/average remaining trajectories, taking average traj and average weights
			#	- take the 'best of the rest' and just discard others
			front_x, front_y, weights_front = merge_extra_neighbours(front_x, front_y, weights_front, num_neighbours, extra_neighbours)
			left_x, left_y, weights_left = merge_extra_neighbours(left_x, left_y, weights_left, num_neighbours, extra_neighbours)
			right_x, right_y, weights_right = merge_extra_neighbours(right_x, right_y, weights_right, num_neighbours, extra_neighbours)
			#print('inter2')
			#print(np.shape(left_x))
			
			# convert 1D to 2D
			selected_x=np.expand_dims(selected_x, axis=1)
			selected_y=np.expand_dims(selected_y, axis=1)
			selected_t=np.expand_dims(selected_t, axis=1)
		
		else:
		
			# no adjacent trajectories, need to create dummy variables and store them
#			print(selected_y.shape)
#			print(selected_y.shape[0])
			front_x = np.zeros([num_neighbours, selected_y.shape[0]])
			front_y = np.zeros([num_neighbours, selected_y.shape[0]])
			weights_front = np.full([num_neighbours, selected_y.shape[0]], 0.00000000000000000000000000000000000000000000001)
			left_x = np.zeros([num_neighbours, selected_y.shape[0]])
			left_y = np.zeros([num_neighbours, selected_y.shape[0]])
			weights_left = np.full([num_neighbours, selected_y.shape[0]], 0.00000000000000000000000000000000000000000000001)
			right_x = np.zeros([num_neighbours, selected_y.shape[0]])
			right_y = np.zeros([num_neighbours, selected_y.shape[0]])
			weights_right = np.full([num_neighbours, selected_y.shape[0]], 0.00000000000000000000000000000000000000000000001)			
			
		sample = {'selected_x' : selected_x, 'selected_y' : selected_y, \
				  'front_x' : front_x, 'front_y' : front_y, 'front_w' : weights_front, \
				  'left_x' : left_x, 'left_y' : left_y, 'left_w' : weights_left, \
				  'right_x' : right_x, 'right_y' : right_y, 'right_w' : weights_right, \
				  'time' : selected_t }

		data.append(sample)

	return data

#
# Main function, use to extract data for later processing by the network
#
def main():	

	# setup command line parser
	parser = argparse.ArgumentParser(description='Create datasets for trajectory prediction')

	#
	# command line parser takes:
	# 	mode: defines whether we are processing a single file (mode == 0) or a list (mode == 1)
	#	primary and secondary data: can be a file or a list
	#	output file: where to save the data that's extracted
	#	trajectory parameters: length, decimate rate, and the number of neighbours to pull out from each mode
	#
	parser.add_argument('--mode', type=int, dest='mode', default=0, help='operating mode, 0 for process a single file (or pair), 1 for a list')
	parser.add_argument('--primary_mode', action='store', dest='primary_mode', help='location of primary mode data. May be either a data file, or a text file with a list of datafiles in it (depending on mode argument)')
	parser.add_argument('--output', action='store', dest='output', help='Where to save stuff')
	parser.add_argument('--length', type=int, dest='traj_length', default=50, help='length of trajectories to extract')
	parser.add_argument('--decimate', type=int, dest='decimate', default=5, help='rate to decimate input data by')
	parser.add_argument('--neighbours', type=int, dest='neighbours', default=10, help='maximum number of neighbours to extract per direction (left, right, front)')
	parser.add_argument('--datasource', type=int, dest='data_source', default=0, help='source of the data, 0=c++, 1=python')
	parser.add_argument('--windowstep', type=int, dest='window_step', default=1, help='sliding window step to use to create more samples')
	parser.add_argument('--slidinglimit', type=int, dest='sliding_limit', default=1, help='where to stop the sliding window')

	results = parser.parse_args()

	# storage for data
	data = []

	# are we processing a list or a single file
	if (results.mode == 0):
		# if it's just a single file, put it in a list anyway, this means that the next
		# bit where we load all the files is the same for each mode
		primary_data = [results.primary_mode]
	else:
		with open(results.primary_mode) as f:
			primary_data = f.readlines()
		primary_data = [x.strip() for x in primary_data]
	# print(primary_data)
	# loop through all the files, load each, extract trajectories, and append to the
	# list of data that we are building
	for i in range(len(primary_data)):
		
		# load primary data
		for j in range(0, results.sliding_limit, results.window_step):
			p = load_data(primary_data[i], results.traj_length, results.decimate, results.data_source, j)
			# load secondary if we have it, otherwise just set it to None
			
			# get data
			d = create_dataset_with_all_neighbours_and_t(p, results.neighbours)
			data = data + d

	# save data
	print(np.shape(data))
	output_file = open(results.output, 'wb')
	for d in data:
		pickle.dump(d, output_file)
	output_file.close()
	
if __name__ == '__main__':
	main()	
    
    