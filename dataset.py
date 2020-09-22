import csv
import cv2
from os import listdir
from os.path import isfile, join
import random
import pickle
import numpy

class inD_detections():
    
    def __init__(self,filename):
        self.detections = []

        with open(filename,'r') as csvfile:
            reader = list(csv.reader(csvfile))
            for row in reader[1:]:
                if (len(self.detections) == 0):
                    self.detections.append(self.new_frame(row))
                else:
                    if (self.detections[-1]['id'] == int(row[0])):
                        self.detections[-1] = self.add_to_frame(row, self.detections[-1])
                    else:
                        self.detections.append(self.new_frame(row))
    
    def new_frame(self,row_data):
        frame={}
        frame['id']=int(row_data[0])
        frame['centre']=[]
        frame['centre'].append([float(row_data[4]),float(row_data[5])])
        return frame

    def add_to_frame(self,row_data,frame):
        frame['centre'].append([float(row_data[4]),float(row_data[5])])
        return frame

    def detections_for_frame(self,frame):
        for i in range(len(self.detections)):
            if (self.detections[i]['id']==frame):
                return self.detections[i]


class inD_gt():

    def __init__(self,filename):
        self.gt =[]
        with open(filename,'r') as csvfile:
            reader = list(csv.reader(csvfile))
            for row in reader[1:]:
                if (len(self.gt) == 0):
                    self.gt.append(self.new_track(row))
                else:
                    idx, continuous = self.find_track(int(row[1]), int(row[2]) - 1)
                    if (idx != -1):
                        if (continuous is True):
                            self.gt[idx] = self.add_to_track(row, self.gt[idx])
                        else:
                            self.gt.append(self.new_track(row))
                    else:
                        self.gt.append(self.new_track(row))

        self.assign_colours()

    def find_track(self,track_id,frame):
        retval = (-1,False)

        for i in range(len(self.gt)):
            if (self.gt[i]['id'] == track_id):
                if (self.gt[i]['frames'][-1] == frame):
                    return i, True
                else:
                    retval = (i,False)
        
        return retval[0],retval[1]

    def new_track(self,row_data):
        track = {}
        track['id'] = int(row_data[1])
        track['frames'] = []
        track['frames'].append(int(row_data[2]))
        track['centre'] = []
        track['centre'].append([int(float(row_data[4])),int(float(row_data[5]))])
        
        track['x'] = []
        track['x'].append(float(row_data[4]))
        track['y'] = []
        track['y'].append(float(row_data[5]))
        return track

    def add_to_track(self,row_data,track):
        track['frames'].append(int(row_data[2]))
        track['centre'].append([int(float(row_data[4])),int(float(row_data[5]))])
        track['x'].append(float(row_data[4]))
        track['y'].append(float(row_data[5]))
        return track

    def assign_colours(self):
        for i in range(len(self.gt)):
            self.gt[i]['colour'] = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))

    def detections_for_frame(self, frame):
        detections = []
        for i in range(len(self.gt)):
            for j in range(len(self.gt[i]['frames'])):
                if (self.gt[i]['frames'][j] == frame):
                    detections.append({'centre' : self.gt[i]['centre'][j], 'id' : self.gt[i]['id'], 'colour' : self.gt[i]['colour']})

        return detections

class inD_dataset():
        
    def __init__(self, file_path, detections, gt=None, target_width=None):
        self.detections = inD_detections(detections)
        if (gt is not None):
            self.gt = inD_gt(gt)
        else:
            self.gt = None

        self.filelist = [join(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]
        self.filelist.sort()	

        tempfile = cv2.imread(self.filelist[0])
        self. width= numpy.shape(tempfile)[1]
        self.height = numpy.shape(tempfile)[0]

        self.target_width = target_width
        if (self.target_width is not None):
            self.scale = self.target_width/self.width

    def convert(self, output_file, normalise = True, scale = False):

        data = []

        for t in self.gt.gt:
            data_track = []
            for tt in range(len(t['frames'])):
                if (normalise):
                    data_track.append([t['frames'][tt], t['x'][tt]/self.width, t['y'][tt]/self.height, 1])
                elif (scale):
                    data_track.append([t['frames'][tt], t['x'][tt]*self.scale, t['y'][tt]*self.scale, 1])					
                else:
                    data_track.append([t['frames'][tt], t['x'][tt], t['y'][tt], 1])

            data.append(data_track)

        f = open(output_file, 'wb')
        pickle.dump(data, f)
        f.close()


def convert_mot_data():
    inD_test = inD_dataset('./image/','./data/00_tracks.csv','./data/00_tracks.csv', target_width = 640)
    inD_test.convert('inD_test.pkl', normalise = False, scale = False)
def main():
    convert_mot_data()

if __name__ == "__main__":
    main()
