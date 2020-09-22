import numpy
import scipy.io as sio
import math
import argparse
import pickle
from itertools import chain
from collections import defaultdict
import random

def file_to_list(filename):
	return [line.rstrip('\n').split(',') for line in open(filename)]

def batches_in_data(filename, batch_size):
	data = file_to_list(filename)
	batches = 0
	for d in data:
		batches = batches + math.floor(int(d[1])/batch_size)
	return batches

#
# load a data pickle and prep it for the network based the various inputs
# Function is a bit of monster (more than intedned) and does the following:
#	- loads all the data from the pickle
#	- does required data wrangling to arrange data. Some of this coul dbe moved into prepare_data. However
#	  doing this here maintain flexibility for different networks
#	- splits data into train and test
#	- creates and returns dictionaries that are the data inputs used by keras
#
def load_data(data_file, input_length, hidden_dim, train_amount):
	
	# allocate storage for all the inputs
	pedestrian_of_interest = numpy.array([])
	left_neighbours = numpy.array([])
	right_neighbours = numpy.array([])
	front_neighbours = numpy.array([])

	left_hw_weights = numpy.array([])
	right_hw_weights = numpy.array([])
	front_hw_weights = numpy.array([])
	
	other_mode_neighbours = numpy.array([])
	other_mode_weights = numpy.array([])
	
	output = numpy.array([])
	secondary_output = numpy.array([])
	
	f = open(data_file, 'rb')
	while True:
		try:		
			# get next entry
			d = pickle.load(f)
			
			# process it
			# start by stacking x and y coords
			selected = numpy.hstack([numpy.reshape(d['selected_x'], (input_length*2,1)), numpy.reshape(d['selected_y'], (input_length*2,1))])

			# coords for neighbours need to be interleaved
			# left neighbours
			left = numpy.zeros([d['left_x'].shape[0], d['left_x'].shape[1], 2], dtype=d['left_x'].dtype)
			for i in range(d['left_x'].shape[0]):
				left[i, :, 0] = d['left_x'][i, :]
				left[i, :, 1] = d['left_y'][i, :]
			# right neighbours
			right = numpy.zeros([d['right_x'].shape[0], d['right_x'].shape[1], 2], dtype=d['right_x'].dtype)
			for i in range(d['right_x'].shape[0]):
				right[i, :, 0] = d['right_x'][i, :]
				right[i, :, 1] = d['right_y'][i, :]
			# front neighbours
			front = numpy.zeros([d['front_x'].shape[0], d['front_x'].shape[1], 2], dtype=d['front_x'].dtype)
			for i in range(d['front_x'].shape[0]):
				front[i, :, 0] = d['front_x'][i, :]
				front[i, :, 1] = d['front_y'][i, :]
						
			# get weights
			left_w = numpy.repeat(numpy.expand_dims(d['left_w'], axis=2), hidden_dim, axis=2)
			right_w = numpy.repeat(numpy.expand_dims(d['right_w'], axis=2), hidden_dim, axis=2)
			front_w = numpy.repeat(numpy.expand_dims(d['front_w'], axis=2), hidden_dim, axis=2)
			
			# separate input and output parts
			# for the selected trajectory, keep both halves
			# note taht for all of the following we expand dims on axis 0, this adds another
			# dimension as the first dim, which we then append over to get all the samples in a
			# single array
			selected_in = numpy.expand_dims(selected[:input_length,:], axis = 0)
			selected_out = numpy.expand_dims(selected[input_length:,:], axis = 0)

			# secondary output is the difference between the final position, and the position if 
			# the subject just continued on their current heading
			# exploit the fact that predicted and known are the same length to do this, our predicted
			# location with constant velocity will be last knoww location plus the movemen across the whole
			# given sequence
			# subtract this from the final position to get the secondary output. Note that this may be negative
			sec_out = selected[-1, :] - (selected[input_length,:] + (selected[input_length - 1,:] - selected[0,:]))
			
			# for the rest which are only inputs, we can ditch the second half
			left = numpy.expand_dims(left[:, :input_length], axis = 0)
			right = numpy.expand_dims(right[:, :input_length], axis = 0)
			front = numpy.expand_dims(front[:, :input_length], axis = 0)
			left_w = numpy.expand_dims(left_w[:, :input_length, :], axis = 0)
			right_w = numpy.expand_dims(right_w[:, :input_length, :], axis = 0)
			front_w = numpy.expand_dims(front_w[:, :input_length, :], axis = 0)
			
			# stack everything as part of the overall arrays
			pedestrian_of_interest = numpy.vstack([pedestrian_of_interest, selected_in]) if pedestrian_of_interest.size else selected_in
			output = numpy.vstack([output, selected_out]) if output.size else selected_out
			secondary_output = numpy.vstack([secondary_output, sec_out]) if secondary_output.size else sec_out

			left_neighbours = numpy.vstack([left_neighbours, left]) if left_neighbours.size else left
			right_neighbours = numpy.vstack([right_neighbours, right]) if right_neighbours.size else right
			front_neighbours = numpy.vstack([front_neighbours, front]) if front_neighbours.size else front

			left_hw_weights = numpy.vstack([left_hw_weights, left_w]) if left_hw_weights.size else left_w
			right_hw_weights = numpy.vstack([right_hw_weights, right_w]) if right_hw_weights.size else right_w
			front_hw_weights = numpy.vstack([front_hw_weights, front_w]) if front_hw_weights.size else front_w	
									
		except EOFError:
			break

	# split into train and test
	# work out cut-off element
	training_cutoff = int(numpy.floor(numpy.shape(pedestrian_of_interest)[0] * train_amount))
	
	# all elements below the cut-off are training
	# stick training data straight into the dictionary to pass to keras
	training_input_dict = {}
	training_input_dict['pedestrian_of_interest'] = pedestrian_of_interest[:training_cutoff,:]
	training_input_dict['left_neighbours'] = left_neighbours[:training_cutoff,:]
	training_input_dict['right_neighbours'] = right_neighbours[:training_cutoff,:]
	training_input_dict['front_neighbours'] = front_neighbours[:training_cutoff,:]
	training_input_dict['left_hw_weights'] = left_hw_weights[:training_cutoff,:]
	training_input_dict['right_hw_weights'] = right_hw_weights[:training_cutoff,:]
	training_input_dict['front_hw_weights'] = front_hw_weights[:training_cutoff,:]

	training_output_dict = {}
	training_output_dict['output_trajectory'] = output[:training_cutoff,:]
	training_output_dict['secondary_output'] = secondary_output[:training_cutoff,:]

	# the rest are testing, again, whack them into dictionaries
	testing_input_dict = {}
	testing_input_dict['pedestrian_of_interest'] = pedestrian_of_interest[training_cutoff:,:]
	testing_input_dict['left_neighbours'] = left_neighbours[training_cutoff:,:]
	testing_input_dict['right_neighbours'] = right_neighbours[training_cutoff:,:]
	testing_input_dict['front_neighbours'] = front_neighbours[training_cutoff:,:]
	testing_input_dict['left_hw_weights'] = left_hw_weights[training_cutoff:,:]
	testing_input_dict['right_hw_weights'] = right_hw_weights[training_cutoff:,:]
	testing_input_dict['front_hw_weights'] = front_hw_weights[training_cutoff:,:]

	testing_output_dict = {}
	testing_output_dict['output_trajectory'] = output[training_cutoff:,:]
	testing_output_dict['secondary_output'] = secondary_output[training_cutoff:,:]
	
	return training_input_dict, training_output_dict, testing_input_dict, testing_output_dict
	
#
# Create a dataset that is multiple files with one held out as the test set. Calls load_data to do all
# the actual loading
# NOTE: I feel like this is a more python way to do the dictionary appending that goes on in here. But in my
# 		(admittedly limited) searching I haven't worked it out
#
# The intended use with this function is that it will be called en-masse (as patch of a batch) to generate 
# a number of folds, each of which is dumped to a pickle. This data is then loaded in a subsequent call
# for processing/learning, etc.
#	
def create_data_set(file_list, test_set_index, output_name, input_length, hidden_dim):
	
	training_input_dict = defaultdict(list)
	training_output_dict = defaultdict(list)

	testing_input_dict = defaultdict(list)
	testing_output_dict = defaultdict(list)
	
	with open(file_list) as f:
		data_list = f.readlines()
	data_list = [x.strip() for x in data_list]
	
	counter = 0
	training_files = 0
	for f in data_list:
		# this file is to be the test set
		if (counter == test_set_index):
			testing_input_dict, testing_output_dict, _, _ = load_data(f, input_length, hidden_dim, 1)
		else:
			input_temp, output_temp, _, _ = load_data(f, input_length, hidden_dim, 1)
			
			if (training_files == 0):
				training_input_dict['pedestrian_of_interest'] = input_temp['pedestrian_of_interest']
				training_input_dict['left_neighbours'] = input_temp['left_neighbours']
				training_input_dict['right_neighbours'] = input_temp['right_neighbours']
				training_input_dict['front_neighbours'] = input_temp['front_neighbours']
				training_input_dict['left_hw_weights'] = input_temp['left_hw_weights']
				training_input_dict['right_hw_weights'] = input_temp['right_hw_weights']
				training_input_dict['front_hw_weights'] = input_temp['front_hw_weights']

				training_output_dict['output_trajectory'] = output_temp['output_trajectory']
				
			else:
				training_input_dict['pedestrian_of_interest'] = numpy.vstack([training_input_dict['pedestrian_of_interest'], input_temp['pedestrian_of_interest']])
				training_input_dict['left_neighbours'] = numpy.vstack([training_input_dict['left_neighbours'], input_temp['left_neighbours']])
				training_input_dict['right_neighbours'] = numpy.vstack([training_input_dict['right_neighbours'], input_temp['right_neighbours']])
				training_input_dict['front_neighbours'] = numpy.vstack([training_input_dict['front_neighbours'], input_temp['front_neighbours']])
				training_input_dict['left_hw_weights'] = numpy.vstack([training_input_dict['left_hw_weights'], input_temp['left_hw_weights']])
				training_input_dict['right_hw_weights'] = numpy.vstack([training_input_dict['right_hw_weights'], input_temp['right_hw_weights']])
				training_input_dict['front_hw_weights'] = numpy.vstack([training_input_dict['front_hw_weights'], input_temp['front_hw_weights']])

				training_output_dict['output_trajectory'] = numpy.vstack([training_output_dict['output_trajectory'], output_temp['output_trajectory']])			
			
			training_files = training_files + 1

		counter = counter + 1

	# create file names for dumping data
	output_train_name = output_name + '-train.pkl'
	output_test_name = output_name + '-test.pkl'
	# training data
	f = open(output_train_name, 'wb')
	pickle.dump((training_input_dict, training_output_dict), f)
	f.close()

	# testing data
	# if this is set to -1, just ignore it
	if (test_set_index != -1):
		f = open(output_test_name, 'wb')
		pickle.dump((testing_input_dict, testing_output_dict), f)
		f.close()


#
# Create a dataset that is suitable for use with keras and inputting data with generators. For now, I'm going to 
# keep it simple and put each input file for the dataset into it's own file, but this could be changed in time
#
def create_data_set_generator(file_list, output_name, input_length, hidden_dim):

	with open(file_list) as f:
		data_list = f.readlines()
	data_list = [x.strip() for x in data_list]

	num_samples = []

	counter = 0
	training_files = 0
	for f in data_list:
		output_file = output_name + '-set-' + str(counter) + '.pkl'
		fold_file = open(output_file, 'w')
		input_data, output_data, _, _ = load_data(f, input_length, hidden_dim, 1)

		# iterate through each entry in the data, create a dictionary for it, and pickle it
		print(numpy.shape(input_data['pedestrian_of_interest'])[0])
		num_samples.append(numpy.shape(input_data['pedestrian_of_interest'])[0])
		for i in range(numpy.shape(input_data['pedestrian_of_interest'])[0]):

			in_dict = {}
			out_dict ={}

			in_dict['pedestrian_of_interest'] = input_data['pedestrian_of_interest'][i:i+1, :]
			in_dict['left_neighbours'] = input_data['left_neighbours'][i:i+1, :]
			in_dict['right_neighbours'] = input_data['right_neighbours'][i:i+1, :]
			in_dict['front_neighbours'] = input_data['front_neighbours'][i:i+1, :]
			in_dict['left_hw_weights'] = input_data['left_hw_weights'][i:i+1, :]
			in_dict['right_hw_weights'] = input_data['right_hw_weights'][i:i+1, :]
			in_dict['front_hw_weights'] = input_data['front_hw_weights'][i:i+1, :]

			out_dict['output_trajectory'] = output_data['output_trajectory'][i:i+1, :]

			pickle.dump((in_dict, out_dict), fold_file)

		fold_file.close()
		counter = counter + 1

	return num_samples


# generator to load data from n different data files. For each next(), a block of data batchsize big will be taken from a single file in order
# The file that is selected will be randomly chosen. When a file is finished, it is reloaded to go again from the start.
# Note that this means that we don't have truly random samples (we always go sequentially through the files, though in a random order), but it does
# mean that all samples that go into a batch are sequential, which might be useful (??).
def data_generator_n_files(files, batchsize, minibatches = 1):

    #loading data
	file_list = file_to_list(files)
	num_files = len(file_list)
	
	minibatchsize = batchsize / minibatches
	
	features = []
	currentIndex = []
	for f in file_list:
		# load each file, and store the current index we are up to
		features.append(open(f[0], "rb"))
		# current index is 0, we've just started
		currentIndex.append(0)
	
	while 1:
	
		ret_in = {}
		ret_out = {}

		# loop over i minibatches, pulling minibatchsize images from each pass
		# allows multiple cameras to be used in a single pass
		for i in range(minibatches):
			
			# pick a file to sample from
			idx = random.randint(0, num_files - 1)
			
			# check that we have minibatchsize samples left in that file
			if ((currentIndex[idx] + minibatchsize) >= int(file_list[idx][1])):
				# if we don't, close the file, and re-open it, and we'll go back to the start
				features[idx].close()
				features[idx] = open(file_list[idx][0], "rb")
				# reset the current index back to the start
				currentIndex[idx] = 0
							
			# get the next batch, pull minibatchsize samples from the selected file
			for i in range(minibatchsize):
				# load our next sample, and stick in the arrays we've created
				if (len(ret_in) == 0):
					(ret_in, ret_out) = pickle.load(features[idx])
				else:
					(a, b) = pickle.load(features[idx])
					for d in ret_in:
						ret_in[d] = numpy.vstack([ret_in[d], a[d]])
					for d in ret_out:
						ret_out[d] = numpy.vstack([ret_out[d], b[d]])
			
			# update the index
			currentIndex[idx] = currentIndex[idx] + minibatchsize

		yield ret_in, ret_out		
			

# a sequential verison of the above - i.e. this won't shuffle anything, intended for use with validation and
# test sets
# note, this will still do the truncation thing when dealing with batch sizes that don't evenly divide by the number of samples in the file,
# so for cases where we are using this for the test set, best to set batch size to 1.
#
def data_generator_n_files_sequential(files, batchsize):

    #loading data
	file_list = file_to_list(files)
	num_files = len(file_list)
	
	currentFile = 0
	currentIndex = 0
	features = []
	for f in file_list:	
		features.append(open(f[0], "rb"))
	
	while 1:
		# have we reached the end of the current file?
		if ((currentIndex + batchsize) >= int(file_list[currentFile][1])):
			# yep, close and reload the file
			features[currentFile].close()
			features[currentFile] = open(file_list[currentFile][0], "rb")

			# move on to the next file
			currentFile = currentFile + 1
			currentIndex = 0
			
			# have we gone through all files?
			if (currentFile >= num_files):
				# yep, go back to the first file
				currentFile = 0
			
		ret_in = {}
		ret_out = {}

		# loop over i minibatches, pulling minibatchsize images from each pass
		# allows multiple cameras to be used in a single pass
		for i in range(batchsize):
			
			if (len(ret_in) == 0):
				(ret_in, ret_out) = pickle.load(features[idx])
			else:
				(a, b) = pickle.load(features[idx])
				for d in ret_in:
					ret_in[d] = numpy.vstack([ret_in[d], a[d]])
				for d in ret_out:
					ret_out[d] = numpy.vstack([ret_out[d], b[d]])
			
		# update the index
		currentIndex = currentIndex + batchsize

		yield ret_in, ret_out

def main():

	# setup command line parser
	parser = argparse.ArgumentParser(description='Create a set of folds for a cross fold validation. Intended for use with the trajectory prediction framework, and designed to process data that has already been paresd by the prepare_data.py program.')

	#
	# command line parser takes:
	#	list of files to use
	#	number of folds
	#	output prefix
	#	length of input trajetories
	#	wheher we are using the aux data or not
	#
	parser.add_argument('--filelist', action='store', dest='file_list', help='list of file to use for fold creation. Shoudl be one file per line. Each file should have been processed with the same settings, and should represent a different piece of data (i.e. a separate capture session)')
	parser.add_argument('--numfolds', type=int, dest='num_folds', help='number of folds to create, this should be the same as the number of lines in the provided file list', default=1)	
	parser.add_argument('--output', action='store', dest='output', help='Output prefix to use for saving files')
	parser.add_argument('--input_length', type=int, dest='input_length', default=25, help='length of trajectories to extract, and to predict.')
	parser.add_argument('--hidden_dim', type=int, dest='hidden_dim', default=24, help='size of hidden dimensions in the model')
	parser.add_argument('--target_fold', type=int, dest='target_fold', help='Do a specific fold, if set to -1 then to all folds', default=-1)
	parser.add_argument('--generator', action='store_true', dest='generator', help='Are we making folds in generator format')
	parser.add_argument('--singleset', action='store_true', dest='single_set', help='Are we a single dataset?, i.e. no train, test and validation?')
	
	results = parser.parse_args()

	if (results.generator == True):
		ns = create_data_set_generator(results.file_list, results.output, results.input_length, results.hidden_dim)
		for i in range(results.num_folds):
			out_train = results.output + '-fold-' + str(i) + '-train.txt'	
			out_train_file = open(out_train, 'w')
			out_valid = results.output + '-fold-' + str(i) + '-validation.txt'
			out_valid_file = open(out_valid, 'w')
			out_test = results.output + '-fold-' + str(i) + '-test.txt'
			out_test_file = open(out_test, 'w')
			val1 = (i + (results.num_folds/3)) % results.num_folds
			val2 = (i + 2*(results.num_folds/3)) % results.num_folds
			for j in range(results.num_folds):
				output_file = results.output + '-set-' + str(j) + '.pkl'
				if (j == i):
					out_test_file.write(output_file + ', ' + str(ns[j]) + '\n')
				elif ((j == val1) | (j == val2)):
					out_valid_file.write(output_file + ', ' + str(ns[j]) + '\n')
				else:
					out_train_file.write(output_file + ', ' + str(ns[j]) + '\n')
			out_train_file.close()
			out_test_file.close()
	elif (results.single_set):
		out_prefix = results.output
		create_data_set(results.file_list, -1, out_prefix, results.input_length, results.hidden_dim)
	else:
		if (results.target_fold == -1):
			for i in range(results.num_folds):
				out_prefix = results.output + '-fold-' + str(i)
				create_data_set(results.file_list, i, out_prefix, results.input_length, results.hidden_dim)
		else:
			out_prefix = results.output + '-fold-' + str(results.target_fold)
			create_data_set(results.file_list, results.target_fold, out_prefix, results.input_length, results.hidden_dim)
	
if __name__ == '__main__':
	main()	
