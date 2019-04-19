
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary


import optparse
import subprocess
import random
import traci
import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


class SumoIntersection:
    def __init__(self):
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
            from sumolib import checkBinary  # noqa
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
        #random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        vehNr = 0
        # demand per second from different directions
        pH = 1. / 7
        pV = 1. / 11
        pAR = 1. / 15
        pAL = 1. / 15
        with open("input_routes.rou.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
    <route id="always_right" edges="1fi 1si 4o 4fi 4si 2o 2fi 2si 3o 3fi 3si 1o 1fi"/>
    <route id="always_left" edges="3fi 3si 2o 2fi 2si 4o 4fi 4si 1o 1fi 1si 3o 3fi"/>
    <route id="horizontal" edges="2fi 2si 1o 1fi 1si 2o 2fi"/>
    <route id="vertical" edges="3fi 3si 4o 4fi 4si 3o 3fi"/>

    ''', file=routes)
            lastVeh = 0
            
            for i in range(N):
                if random.uniform(0, 1) < pH:
                    print('    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="horizontal" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pV:
                    print('    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="vertical" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pAL:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_left" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pAR:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_right" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            print("</routes>", file=routes)
        print('no of vehicles = ',vehNr)
        return vehNr
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14
        max_arr=[0]
        junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1si')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2si')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3si')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4si')
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(34):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            max_arr.append(ind)
            if(ind <=33):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            max_arr.append(ind)
            if(ind <= 33):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[1]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            max_arr.append(ind)
            if(ind <= 33):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road4:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            max_arr.append(ind)
            if(ind <= 33):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = []
        if(traci.trafficlight.getPhase('0') == 4):
            light = [1, 0]
        else:
            light = [0, 1]

        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 34, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 34, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts],max_arr


if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    options = sumoInt.get_options()
    #print('options.nogui',options.nogui)
    if options.nogui:
    #if True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    #vehNr=sumoInt.generate_routefile()

    traci.start([sumoBinary, "-c", "cross3ltl.sumocfg",'--start'])

    state = sumoInt.getState()
    horizontal_sum = np.sum(state[0][0][0:6,])
    vertical_sum = np.sum(state[0][0][7:13,])

    waiting_time = 0
    max_ind=0
    trafic_light_change_no=0
    while traci.simulation.getMinExpectedNumber() > 0:
        state,max_arr = sumoInt.getState()

        if(max_ind<max(max_arr)):
            max_ind=max(max_arr)
        
        horizontal_sum = np.sum(state[0][0][0:6,])
        vertical_sum = np.sum(state[0][0][6:13,])  
        
        if(horizontal_sum>=vertical_sum):
            trafic_light_change_no+=1
            #print ('horizontal',horizontal_sum,vertical_sum)
            #print(state[0][0][0:6,])
            for i in range(10):
                
                traci.trafficlight.setPhase('0', 4)
                
                waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(\
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    #print(waiting_time)
                traci.simulationStep()
            for i in range(10):
                traci.trafficlight.setPhase('0', 5)
                waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(\
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                #print(waiting_time)
                traci.simulationStep()

        else:
            trafic_light_change_no+=1
            #print ('vertical',vertical_sum,horizontal_sum)
            #print(state[0][0][6:13,])
            
            for i in range(10):
                traci.trafficlight.setPhase('0', 0)
                
                waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(\
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    #print(waiting_time)
                traci.simulationStep()
            for i in range(10):
                traci.trafficlight.setPhase('0', 1)
                waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(\
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                #print(waiting_time)
                traci.simulationStep()

    #print(waiting_time)
    f= open("2.txt", "a") 
    f.write(str(waiting_time)+'\n')
    f.close()

