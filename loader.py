#
#	Code to load the trajectories that come out of c++ into python. Given 
#	that this is loading binary data, it relies on things being precisely
#	where they're expected to be. If the corresponding c++ trajectories use
#	a different data type somewhere to what's in here, it will fail.
#
#	Change Log:
#		??/??/2017:			Initial Version (SD)
#		312/01/2018 (SD):	Added some functionality to pull out dataset
#							stats and do some mediocre visualisaions. At
#							moment I've gone with hardcoded paths in there,
#							which is obviously really bad
#

import struct
import numpy
import random
import pickle

def load_cplusplus_trajectories(filename):
	
	with open(filename, mode='rb') as file:
		fileContent = file.read()
	
	(numtraj,) = struct.unpack("i", fileContent[:4])
	print(numtraj)

	traj_list = []

	nextread = 4
	for t in range(numtraj):
		(traj_size,) = struct.unpack("i", fileContent[nextread:nextread+4])
		nextread = nextread + 4
		traj = []
		for tt in range(traj_size):
			(time,) = struct.unpack("d", fileContent[nextread:nextread+8])
			nextread = nextread + 8
			(x,) = struct.unpack("d", fileContent[nextread:nextread+8])
			nextread = nextread + 8
			(y,) = struct.unpack("d", fileContent[nextread:nextread+8])
			nextread = nextread + 8
			(w,) = struct.unpack("d", fileContent[nextread:nextread+8])
			nextread = nextread + 8
	
			traj.append([time, x, y, w])
		
		traj_list.append(traj)
					
	return traj_list

def load_python_trajectories(filename):
	f = open(filename, 'rb')
	traj_list = pickle.load(f)
	f.close()
	return traj_list
	
def get_stats(traj_list):
	total = len(traj_list);
	min_len = 999999;
	max_len = 0;
	ave_len = 0;
	
	min_x = 99999999;
	min_y = 99999999;
	max_x = 0;
	max_y = 0;
	
	for x in traj_list:
		min_len = min(min_len, len(x))
		max_len = max(max_len, len(x))
		ave_len = ave_len + len(x)
		
		for xx in x:
			min_x = min(min_x, xx[1])
			max_x = max(max_x, xx[1])
			min_y = min(min_y, xx[2])
			max_y = max(max_y, xx[2])

	ave_len = ave_len / total
	
	print(str(total) + ' & ' + str(min_len) + ' & ' + str(max_len) + ' & ' + str(ave_len))
	
	print(min_x)
	print(max_x)
	print(min_y)
	print(max_y)

def get_info():

	traj_list = load_cplusplus_trajectories('trackfile-2016-03-09-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('trackfile-2016-03-11-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('trackfile-2016-03-14-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('trackfile-2016-03-16-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('trackfile-2016-03-18-filtered.dat')
	get_stats(traj_list)
	
	traj_list = load_cplusplus_trajectories('2016-03-09-Mode2-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('2016-03-11-Mode2-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('2016-03-14-Mode2-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('2016-03-16-Mode2-filtered.dat')
	get_stats(traj_list)
	traj_list = load_cplusplus_trajectories('2016-03-18-Mode2-filtered.dat')
	get_stats(traj_list)
	
