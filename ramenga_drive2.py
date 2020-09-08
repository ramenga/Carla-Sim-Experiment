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

from tensorflow import keras

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

def process_img(image):
	raw_img = np.array(image.raw_data) #get image raw data which is a single flat array
	img_shaped = raw_img.reshape((IM_HEIGHT,IM_WIDTH,4)) #Reshape the single flat data into RGBA array
	img_rgb = img_shaped[:,:,:3] #get rgb only from rgba
	cv2.imshow("",img_rgb) #openCV display rgb image
	print("imagecaptured")
	cv2.waitKey(2) #waits for key event for a delay time
	return img_rgb/255.0 #return normalized data

try:
	#create a client object to connect to the sim(server)
	client = carla.Client('localhost',2000)
	client.set_timeout(10.0) #timeout for the network operation

	world = client.get_world() #after client object is used for connection, retrieve the world object from it

	# A blueprint contains information necessary to create a new actor (actor such as cars, sensor,etc)
	# Attributes of the actors can also be defined here (such as color of car, channel of sensor etc.)
	# The list of all available blueprints is kept in the blueprint library 
	blueprint_library = world.get_blueprint_library()		#get the blueprint library from world object

	bp = blueprint_library.filter("model3")[0]  #picking a car
	print(bp)

	# Get spawn points from the current world map and choose a random one from it
	#spawn_point = random.choice(world.get_map().get_spawn_points()) #object containing a spawn point choose random from list
	spawn_point = world.get_map().get_spawn_points()[8]
	#print(spawn_point)

	# Spawn a vehicle at the spawn point
	vehicle = world.spawn_actor(bp,spawn_point)

	#vehicle.set_autopilot(True) #Rule based game engine autopilot (not real autopilot, just like game npc)

	vehicle.apply_control(carla.VehicleControl(throttle=0.3,steer=0.0))
	actor_list.append(vehicle) #append actor to list

	#Set up the camera blueprint
	camera_bp = blueprint_library.find("sensor.camera.rgb")
	camera_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
	camera_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
	camera_bp.set_attribute("fov","120")

	#spawn point of camera
	spawn_point_cam = carla.Transform(carla.Location(x=2.5,z=0.7))

	#Set up the camera sensor (actor) at the spawn point
	sensor = world.spawn_actor(camera_bp,spawn_point_cam,attach_to=vehicle)
	actor_list.append(sensor)
	print("HAY")
	#get information from camera, the lambda function also displays the data
	sensor.listen(lambda data : process_img(data)) #this is a concurrent process that continuously listens
	print("DOH")

	time.sleep(20)
	vehicle.apply_control(carla.VehicleControl(throttle=0.0,steer=0.0))

finally:
	for actor in actor_list: #Cleanup job
		actor.destroy()
