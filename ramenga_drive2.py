'''
Actors:vehicles,pedestrians,sensors, traffic lights and signs
Blueprint:attributes of the actors
World: Server running the simulation
'''

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model=None #The Keras model


try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt'else'linux-x86-64'))[0])
except IndexError:
	pass

import carla

actor_list=[] #create an empty list to store all actors

#define camera image constants
IM_WIDTH = 640
IM_HEIGHT = 480

def extract_white(image):
	ysize = image.shape[0]
	xsize = image.shape[1]
	# Note: always make a copy rather than simply using "="
	color_select = np.copy(image)
	red_threshold = 50
	green_threshold = 50
	blue_threshold = 50

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * 2.2
	hsv[:,:,2] =  hsv[:,:,2] * ratio
	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	color_select = np.copy(image)
	rgb_threshold = [red_threshold, green_threshold, blue_threshold]
	thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
	color_select[thresholds] = [0,0,0]

	#Select only triangular region
	region_select = color_select
	print("SHAPE of IMG",region_select.shape)
	left_bottom = [0,int(region_select.shape[0]*0.9)] #shape output is (y,x,) but this system accept (x,y,)
	right_bottom = [int(region_select.shape[1]),int(region_select.shape[0]*0.9)]
	apex = [int(region_select.shape[1]*0.4), int(region_select.shape[0]*0.5)]
	fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
	fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
	fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
	region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
	# Color pixels red which are inside the region of interest
	region_select[~region_thresholds] = [0, 0, 0]

	cv2.imshow("",region_select) 
	# Display the image                 
	#plt.imshow(color_select)
	#plt.show()

def edge_detection(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * 2.5
	hsv[:,:,2] =  hsv[:,:,2] * ratio
	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

	low_threshold = 1
	high_threshold = 220
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	#cv2.imshow("",edges) 

	rho = 1  #pixels, distance in Hough Space grid
	theta = (np.pi/180)*2  #Angular resolution of grid in Hough Space
	threshold = 5 	#intersections in a given grid cell to qualify as a line
	min_line_length = 30	#minimum length of line acceptable as output
	max_line_gap = 10		#max dist between segments  to allow to be connected into a single line
	line_image = np.copy(image)*0 #creating a blank to draw lines on

	# Run Hough on edge detected image
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on the blank
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

	#cv2.imshow("",line_image) 

	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
	cv2.imshow("",combo) 


def process_img(image):
	raw_img = np.array(image.raw_data) #get image raw data which is a single flat array
	img_shaped = raw_img.reshape((IM_HEIGHT,IM_WIDTH,4)) #Reshape the single flat data into RGBA array
	img_rgb = img_shaped[:,:,:3] #get rgb only from rgba
	
	#cv2.imshow("",img_rgb) #openCV display rgb image
	
	rgbImage = cv2.cvtColor(img_shaped, cv2.COLOR_RGBA2RGB)
	#extract_white(rgbImage)
	edge_detection(rgbImage)
	#cv2.imshow("",rgbImage) #openCV display rgb image
	#print("imagecaptured")
	cv2.waitKey(2) #waits for key event for a delay time
	return img_rgb #return data

def drive():
	null

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Autonomous Driving')
	parser.add_argument(
    	'model',
    	type=str,
    	help='Path to model h5 file. Model should be on the same path.'
	)
	args = parser.parse_args()

	#load model
	model = load_model(args.model) #load model from file 

try:
	#create a client object to connect to the sim(server)
	client = carla.Client('localhost',2000)
	client.set_timeout(10.0) #timeout for the network operation

	world = client.get_world() #after client object is used for connection, retrieve the world object from it

	# A blueprint contains information necessary to create a new actor (actor such as cars, sensor,etc)
	# Attributes of the actors can also be defined here (such as color of car, channel of sensor etc.)
	# The list of all available blueprints is kept in the blueprint library 
	blueprint_library = world.get_blueprint_library()		#get the blueprint library from world object

	bp = blueprint_library.filter("model3")[0]  #picking a car, Tesla Model 3
	print(bp)

	# Get spawn points from the current world map and choose a random one from it
	#spawn_point = random.choice(world.get_map().get_spawn_points()) #object containing a spawn point choose random from list
	spawn_point = world.get_map().get_spawn_points()[8] #use specific spawn point from the spawn point list

	# Spawn a vehicle at the spawn point
	vehicle = world.spawn_actor(bp,spawn_point)

	#vehicle.set_autopilot(True) #Rule based game engine autopilot (not real autopilot, just like a game npc)

	vehicle.apply_control(carla.VehicleControl(throttle=0.3,steer=0.0))
	actor_list.append(vehicle) #append actor to list

	#Set up the camera blueprint
	camera_bp = blueprint_library.find("sensor.camera.rgb")
	camera_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
	camera_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
	camera_bp.set_attribute("fov","130")

	#spawn point of camera
	spawn_point_cam = carla.Transform(carla.Location(x=2.5,z=0.7))

	#Set up the camera sensor (actor) at the spawn point
	sensor = world.spawn_actor(camera_bp,spawn_point_cam,attach_to=vehicle)
	actor_list.append(sensor)
	#print("HAY")
	#get information from camera, the lambda function also displays the data
	sensor.listen(lambda data : process_img(data)) #this is a concurrent process that continuously listens
	#print("DOH")

	time.sleep(20)
	vehicle.apply_control(carla.VehicleControl(throttle=0.0,steer=0.0))

finally:
	for actor in actor_list: #Cleanup job
		actor.destroy()

