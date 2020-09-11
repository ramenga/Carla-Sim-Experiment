import csv
import argparse
import os
import cv2

from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

lines=[]
images=[]
measurements=[]

def load_data(args):

	#load the csv file and put each line into a list named lines
	with open(args.data_dir+'/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	for line in lines:
		source_path = line[0] #first element of each line is the path to the file
		image = cv2.imread(source_path)
		images.append(image)
		cv2.imshow("",image)
		cv2.waitKey(1)
		steer_angle = float(line[3])
		measurements.append(steer_angle)






	
	cv2.waitKey(50)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
	args = parser.parse_args()

	print(args.data_dir)

	load_data(args)


