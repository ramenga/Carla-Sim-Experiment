#Actors:vehicles,pedestrians,sensors, traffic lights and signs
#Blueprint:attributes of the actors
#World: Server running the simulation
import glob
import os
import sys
import random

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt'else'linux-x86-64'))[0])
except IndexError:
	pass

import carla

actor_list=[] 

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
	spawn_point = random.choice(world.get_map().get_spawn_points()) #object containing a spawn point

	# Spawn a vehicle at the spawn point
	vehicle = world.spawn_actor(bp,spawn_point)

	vehicle.set_autopilot(True) #Rule based game engine autopilot (not real autopilot, just like game npc)

finally:
	for actor in actor_list: #Cleanup job
		actor.destroy()
