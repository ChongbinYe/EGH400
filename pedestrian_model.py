#
#	Model definition for the neighbourhood pedestrian model
#
#	Change Log:
#		16/01/2018:			Initial Version (SD), based on Tharindu's code
#		30/01/2018 (SD):	Write main method, start to test this version 
#							of things
#		02/02/2018 (SD):	Revised network arch to make dimensions line up. 
#							Similar revisions to data wrangling such that it's
#							consistent
#		03/02/2018 (SD):	Fixes to merge and concats that broke the network,
#							finalise dual input operation
#		04/02/2018 (SD):	added error metrics
#		01/03/2019 (SD):	gutted bits of this to get it back to a more realistic implemenation of the original soft+hw paper
#							TODO: re-implement seq2seq or something like that
#		06/03/2019 (SD):	Removed AttentionSeq2Seq and replaced with a rough equivilent, likely to need further tweaks, but
#							provides a reasonable approximation for now
#

import numpy
import scipy.io as sio
import math
import argparse
import pickle
from keras.models import Model
from keras.layers import Input, merge,Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.merge import Concatenate, Average,Dot,Multiply
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Reshape
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.metrics import mean_squared_error
from plot_trajectories import plot_results_with_neighbours
import cv2

# removed due to lack of support for recent toolkits
#from seq2seq.models import AttentionSeq2Seq

#
# Build the subnetwork for a neighbourhood
# Basically, an LSTM that encodes input trajectories. Note that weights aren't applied here.
# 
def NeighbourNetwork(hidden_dim, lstm_dropout = 0.25):

	def f(input):
		neighbour_model = TimeDistributed(LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(input)
		return neighbour_model
       
	return f
		
#
# Build the pedestrian model
#
def pedestrian_model(input_length, input_dim, num_neighbours, output_length, output_dim, hidden_dim, \
					 lstm_dropout = 0.25):

	# lists of inputs and outputs, build these lists as we create inputs and outputs
	inputs = []
	outputs = []

	#
	# define inputs
	#
	# pedestrian of interest
	pi_input = Input(shape = (input_length, input_dim), name = 'pedestrian_of_interest')
	inputs = inputs + [pi_input]
	# neighbours, represent as a 3D tensor to allow for multiple neighbours per input
	left_input = Input(shape = (num_neighbours, input_length, input_dim), name = 'left_neighbours')
	right_input = Input(shape = (num_neighbours, input_length, input_dim), name = 'right_neighbours')
	front_input = Input(shape = (num_neighbours, input_length, input_dim), name = 'front_neighbours')
	inputs = inputs + [left_input]
	inputs = inputs + [right_input]
	inputs = inputs + [front_input]

	# define hard-wired weight inputs
	left_weights = Input(shape = (num_neighbours, input_length, hidden_dim), name = 'left_hw_weights')
	right_weights = Input(shape = (num_neighbours, input_length, hidden_dim), name = 'right_hw_weights')
	front_weights = Input(shape = (num_neighbours, input_length, hidden_dim), name = 'front_hw_weights')
	inputs = inputs + [left_weights]
	inputs = inputs + [right_weights]
	inputs = inputs + [front_weights]
    
	#
	# actual model
	#
	# historically we've used seq2seq and done this bit
    # pi_model=AttentionSeq2Seq(output_dim = hidden_dim, hidden_dim = hidden_dim, output_length = output_length, input_shape=(input_length, input_dim), depth = 2)(pi_input)
    # this is an encoder-decoder built with soft attention at the end of the decoder. The encoders are also bi-directional. Encoders and 
    # decoders are LSTM cells
    # As seq2seq is not compatible with current versions of keras/tensorflow/theano, we are going to swap that out for a rough approximate
    # of what it was doing
    #
    # basically, we've have stacked bi-directions LSTMs to encode, and then stacked single direction LSTMs to decode, with soft attention
    # at the end
    # overall, this seems to do a reasonable job and is very much similar in theme to what the old seq2seq module was doing
    #
    # Some Notes:
    #	- training seems to have a strong plateau at the moment, afterwhich it then improves again. My experience is that
    #	  after 3-5 epochs it looks like learning has largely stopped, until about 15-20 epochs at which point it starts to improve
    #	  again, and will ultimatley get a lot better.
    #

    # stacked encoders
    # both are returning sequences, are bi-directions and use a sum merge, this is taken from what seq2seq did
	#encoder_outputs, e1_h, e1_c = LSTM(hidden_dim, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
	#	return_sequences=True, return_state=True, unroll=False, name='pi_encoder1')(pi_input)
	encoder_outputs, e1_h, e1_c, _, _ = Bidirectional(LSTM(hidden_dim, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
		return_sequences=True, return_state=True, unroll=False, name='pi_encoder1'), merge_mode='sum')(pi_input)
	e1_states = [e1_h, e1_c]
	encoder_outputs, e2_h, e2_c, _, _ = Bidirectional(LSTM(hidden_dim, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
		return_sequences=True, return_state=True, unroll=False, name='pi_encoder2'), \
		merge_mode='sum')(encoder_outputs)
	e2_states = [e2_h, e2_c]

	# Set up the decoder
	# this is going to be similar to the encoder, two decoders, both returning sequnces, and stacked
	# NOTE: I'm trying making these bi-directional. This isn't what seq2seq does
	decoder_lstm, d1_h, d1_c = LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
		return_state=True, unroll=False, name='pi_decoder1')(encoder_outputs)
	d1_states = [d1_h, d1_c]
	decoder_lstm = LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
		return_state=False, unroll=False, name='pi_decoder2')(decoder_lstm)
#	decoder_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
#		return_state=False, unroll=False, name='pi_decoder1'), merge_mode='sum')(encoder_outputs)
#	decoder_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, \
#		return_state=False, unroll=False, name='pi_decoder2'), merge_mode='sum')(decoder_lstm)

	# finally we're going to apply some soft attetion to the output, MLP with softmax activation
	pi_model = Dense(hidden_dim, activation='softmax', name='pi_attention')(decoder_lstm)

	# neighbourhood encodings
	left_neigbour_model = 	NeighbourNetwork(hidden_dim, lstm_dropout)(left_input)
	right_neigbour_model = 	NeighbourNetwork(hidden_dim, lstm_dropout)(right_input)
	front_neigbour_model = 	NeighbourNetwork(hidden_dim, lstm_dropout)(front_input)
	# multiply neighbourhood encodings with hard-wired weights
	left_neigbour_model = 	Multiply(name='left_neighbour_mult')([left_neigbour_model, left_weights])
	right_neigbour_model = 	Multiply(name='right_neighbour_mult')([right_neigbour_model, right_weights])
	front_neigbour_model = 	Multiply(name='front_neighbour_mult')([front_neigbour_model, front_weights])

	neighbourhood_context_vec = Concatenate(axis=-1)([left_neigbour_model, right_neigbour_model, front_neigbour_model])
	neighbourhood_context_vec = Reshape((input_length, -1))(neighbourhood_context_vec)

    # pass neighbourhood context through dense layer to get to shape (input_length x hidden_dim)
	# this effectively extracts out the important information for each point from all the neighbours
	neighbourhood_context_vec = TimeDistributed(Dense(hidden_dim))(neighbourhood_context_vec)
    
	# merge pedestrian of interest with neighbourhood
	merge_context_vec = Concatenate(axis=-1)([pi_model, neighbourhood_context_vec])
 
	# output encoding, we use an LSTM to model the output sequence
	decoder = LSTM(hidden_dim, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_dropout)(merge_context_vec)

	# and then use a dense layer to go from hidden_dim -> 2 (i.e. x, y) values
	output = TimeDistributed(Dense(output_dim), name='output_trajectory')(decoder)

	# add the output to out list
	outputs = outputs + [output]
    
	return Model(input=inputs, output=outputs)

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
	
	f = open(data_file, 'rb')
	while True:
		try:
			# get next entry
			d = pickle.load(f)

			# process it
			# start by stacking x and y coords
			selected_in = d[0]['pedestrian_of_interest']
			selected_out = d[1]['output_trajectory']

			# coords for neighbours need to be interleaved
			# left neighbours
			left = d[0]['left_neighbours']
			right = d[0]['right_neighbours']
			front = d[0]['front_neighbours']
			
			# get weights
			left_w = d[0]['left_hw_weights']
			right_w = d[0]['right_hw_weights']
			front_w = d[0]['front_hw_weights']
			
			# stack everything as part of the overall arrays
			pedestrian_of_interest = numpy.vstack([pedestrian_of_interest, selected_in]) if pedestrian_of_interest.size else selected_in
			output = numpy.vstack([output, selected_out]) if output.size else selected_out

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
	
	return training_input_dict, training_output_dict, testing_input_dict, testing_output_dict

#
# error metrics, fde
#
def final_displacement_error(y_all_gt,y_all_pred):
	y_true=y_all_gt[:,-1]
	y_pred=y_all_pred[:,-1]
	loss= mean_squared_error(y_true, y_pred)
	return math.sqrt(loss)

#
# error metrics, ade
#
def average_displacement_error(y_all_gt,y_all_pred):
	y_true=y_all_gt
	y_pred=y_all_pred
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	loss= mean_squared_error(y_true, y_pred,multioutput='raw_values')
	return math.sqrt(numpy.mean(loss))

	
#
# main function, create model, load data, and run it
#	
def main():

	# setup command line parser
	parser = argparse.ArgumentParser(description='Trajectory prediction deep recurrent neural networks.')

	#
	# command line parser. We're getting:
	#	- input data
	#	- output file prefix and path. All output files will be the output path plus stuff tacked onto the end
	#	- flags to indicate if we are using the auxiliary mode, or secondary outupt
	#	- arguments for trajectory length and the number of neighbours, i.e. anything that impacts that data
	#	  shape of the network inputs, and thus the network itself
	#	- some network params such as batchsize and epochs, though these don't really need to be fiddled with
	#
	parser.add_argument('--data', action='store', dest='data', help='data to use for training')
	parser.add_argument('--output', action='store', dest='output', help='path and name to dump output to')
	parser.add_argument('--input_length', type=int, dest='input_length', default=25, help='lenth of trajectories to learn from')
	parser.add_argument('--hidden_dim', type=int, dest='hidden_dim', default=24, help='size of hidden dimensions in the model')
	parser.add_argument('--dropout_rate', type=float, dest='dropout_rate', default=0.25, help='drop-out rate for LSTM units')
	parser.add_argument('--output_length', type=int, dest='output_length', default=25, help='length of trajectories to predict')
	parser.add_argument('--neighbours', type=int, dest='neighbours', default=10, help='maximum number of neighbours to extract per direction (left, right, front)')
	# network training params
	parser.add_argument('--batchsize', type=int, dest='batch_size', default=32, help='batch size for model training')
	parser.add_argument('--epochs', type=int, dest='epochs', default=100, help='number of epochs')
	parser.add_argument('--valsplit', type=float, dest='val_split', default=0.1, help='validation data split')
	parser.add_argument('--loss', action='store', dest='loss', default='mean_squared_error', help='loss to use, suggest one of mean_absolute_error or mean_squared_error depending on how the data is scaled')

	results = parser.parse_args()

	# create the model
	model = pedestrian_model(results.input_length, 2, results.neighbours, results.output_length, 2, results.hidden_dim, \
							 results.dropout_rate)
							 
	model.summary()

	# get data
	# NOTE: at the moment we've hard coded 85% for training
	input_train, target_train, input_test, target_test = load_data(results.data, results.input_length, results.hidden_dim, 0.95)

	# setup model training
	# compile the model, same settings as Tharindu's version
	model.compile(loss=results.loss, optimizer='Adam', metrics=['mae', 'mse', 'acc'])

	# create checkpoint to save model
	checkpoint_filepath = results.output + ".hdf5"
	checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
	callbacks_list = [checkpoint]

	# train the model
	model.fit(input_train, target_train, batch_size=results.batch_size, epochs=results.epochs, callbacks=callbacks_list, validation_split=results.val_split, shuffle=True)
    
	# load best model here, and then predict
	model.load_weights(checkpoint_filepath)

	# predict
	predicted = model.predict(input_test, batch_size=32, verbose=0)

	print('input')
	print(input_test['pedestrian_of_interest'][0,:])
	print('predicted')
	print(predicted[0,:])
	print('expected')
	print(target_test['output_trajectory'][0,:])
	print('')

	print('input')
	print(input_test['pedestrian_of_interest'][1,:])
	print('predicted')
	print(predicted[1,:])
	print('expected')
	print(target_test['output_trajectory'][1,:])
	print('')

	print('input')
	print(input_test['pedestrian_of_interest'][2,:])
	print('predicted')
	print(predicted[2,:])
	print('expected')
	print(target_test['output_trajectory'][2,:])
	print('')

	print('input')
	print(input_test['pedestrian_of_interest'][3,:])
	print('predicted')
	print(predicted[3,:])
	print('expected')
	print(target_test['output_trajectory'][3,:])
	print('')

	print('input')
	print(input_test['pedestrian_of_interest'][4,:])
	print('predicted')
	print(predicted[4,:])
	print('expected')
	print(target_test['output_trajectory'][4,:])
	print('')

	# get error metrics and print to screen
	print('Average Displacement Error: ' + str(average_displacement_error(predicted, target_test['output_trajectory'])))
	print('Final Displacement Error:   ' + str(final_displacement_error(predicted, target_test['output_trajectory'])))

	print('IRL feature extraction....')
	save_path='./data'
	input_train, target_train, input_test, target_test = load_data(results.data, results.input_length, results.hidden_dim, 0.95)
	traj_obs=input_train['pedestrian_of_interest']
	
	traj_future=target_train['output_trajectory']
	
	
	left_neighbours=input_train['left_neighbours']
	right_neighbours=input_train['right_neighbours']
	front_neighbours=input_train['front_neighbours']
	
	
	#no_of_trajectories=intermediate_output.shape[0]
	
	no_of_trajectories=traj_obs.shape[0]
	
	#---- normalising the data-----
	# for i in range(intermediate_output.shape[1]):
	# 	intermediate_output[:,i,:]=normalize(intermediate_output[:,i,:], axis=1, norm='l2')
	# 
	
	for i in range(no_of_trajectories):
		print('===> saving  ' +str(i) +'/'+str(no_of_trajectories))
		traj_obs_new=traj_obs[i]
		traj_future_new=traj_future[i]
		print(traj_obs_new.shape)
		print(traj_future_new.shape)
		traj_new=numpy.concatenate((traj_obs_new,traj_future_new),axis=0)
		print(traj_new.shape)
		#feat=numpy.reshape(intermediate_output[i,:,:],(intermediate_output.shape[1], 20,20))
	       
		#----------- plot the traj of PI and neighbours for the baseline models input-----
		x_first_part=traj_obs[i,:,0]
		y_first_part=traj_obs[i,:,1]
		x_second_part=traj_future[i,:,0]
		y_second_part=traj_future[i,:,1]
		x_pred=x_second_part
		y_pred=y_second_part
	       
		l_x=numpy.expand_dims(left_neighbours[i,:,:,0],axis=0)
		l_y=numpy.expand_dims(left_neighbours[i,:,:,1],axis=0)
		r_x=numpy.expand_dims(right_neighbours[i,:,:,0],axis=0)
		r_y=numpy.expand_dims(right_neighbours[i,:,:,1],axis=0)
		f_x=numpy.expand_dims(front_neighbours[i,:,:,0],axis=0)
		f_y=numpy.expand_dims(front_neighbours[i,:,:,1],axis=0)
		neighbour_x=numpy.concatenate((l_x,r_x,f_x),axis=0)
		neighbour_y=numpy.concatenate((l_y,r_y,f_y),axis=0)
		
		print(neighbour_x.shape)
		
		file_name='NGSIM_plots_for_features/example_track_%d.png'%i
		plot_results_with_neighbours(x_first_part,y_first_part,x_second_part,y_second_part,x_pred,y_pred,neighbour_x,neighbour_y,[],[-10, 100],[-10 ,100],save_file_name=file_name)
		img = cv2.imread(file_name)
		img = cv2.bitwise_not(img)
		cv2.imwrite(file_name, img)
		resized_image = cv2.resize(img, (80, 80))
		resized_image=numpy.swapaxes(resized_image,0,2)
		
		#sio.savemat(save_path+'/example_track_'+str(i)+'.mat',{'traj':traj_new,'feat':feat, 'baseline_feat':resized_image})
		sio.savemat(save_path+'/example_track_'+str(i)+'.mat',{'traj':traj_new, 'baseline_feat':resized_image})
	
if __name__ == '__main__':
	main()