# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import abc
import logging
import math
import time
import json

from carla.client import VehicleControl
from carla.client import make_carla_client
from carla.driving_benchmark.metrics import Metrics
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
import numpy as np
from . import results_printer
from .recording import Recording

def cycle_signal(signal, value):
    signal.pop(0)
    signal.append(value)

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

class DrivingBenchmark(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.


    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            save_images=False,
            distance_for_success=2.0
    ):
        """
        Args
            city_name:
            name_to_save:
            continue_experiment:
            save_images:
            distance_for_success:
            collisions_as_failure: if this flag is set to true, episodes will terminate as failure, when the car collides.
        """

        self.__metaclass__ = abc.ABCMeta

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images
                                    )

        # We have a default planner instantiated that produces high level commands
        self._planner = Planner(city_name)

        # TO keep track of the previous collisions
        self._previous_pedestrian_collision = 0
        self._previous_vehicle_collision = 0
        self._previous_other_collision = 0



    def benchmark_agent(self, experiment_suite, agent, client):
        """
        Function to benchmark the agent.
        It first check the log file for this benchmark.
        if it exist it continues from the experiment where it stopped.


        Args:
            experiment_suite
            agent: an agent object with the run step class implemented.
            client:


        Return:
            A dictionary with all the metrics computed from the
            agent running the set of experiments.sldi
        """

        # Instantiate a metric object that will be used to compute the metrics for
        # the benchmark afterwards.
        metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

        # Function return the current pose and task for this benchmark.
        start_pose, start_experiment = self._recording.get_pose_and_experiment(
            experiment_suite.get_number_of_poses_task())

        logging.info('START')

        for experiment in experiment_suite.get_experiments()[int(start_experiment):]:

            positions = client.load_settings(
                experiment.conditions).player_start_spots
            t=len(positions)
            self._recording.log_start(experiment.task)

            for pose in experiment.poses[start_pose:]:
                for rep in range(experiment.repetitions):

                    start_index = pose[0]
                    end_index = pose[1]

                    client.start_episode(start_index)
                    # Print information on
                    logging.info('======== !!!! ==========')
                    logging.info(' Start Position %d End Position %d ',
                                 start_index, end_index)

                    self._recording.log_poses(start_index, end_index,
                                              experiment.Conditions.WeatherId)

                    # Calculate the initial distance for this episode
                    initial_distance = \
                        sldist(
                            [positions[start_index].location.x, positions[start_index].location.y],
                            [positions[end_index].location.x, positions[end_index].location.y])

                    time_out = experiment_suite.calculate_time_out(
                        self._get_shortest_path(positions[start_index], positions[end_index]))

                    # running the agent
                    (result, reward_vec, control_vec, final_time, remaining_distance, col_ped, col_veh, col_oth,metList) = \
                        self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index), experiment_suite.metrics_parameters,
                            experiment_suite.collision_as_failure)

                    # Write the general status of the just ran episode
                    self._recording.write_summary_results(
                        experiment, pose, rep, initial_distance,
                        remaining_distance, final_time, time_out, result, col_ped, col_veh, col_oth)

                    # Write the details of this episode.
                    self._recording.write_measurements_results(experiment, rep, pose, reward_vec,
                                                               control_vec, metList)
                    if result > 0:
                        logging.info('+++++ Target achieved in %f seconds! +++++',
                                     final_time)
                    else:
                        logging.info('----- Timeout! -----')

            start_pose = 0

        self._recording.log_end()

        return metrics_object.compute(self._recording.path)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _has_agent_collided(self, measurement, metrics_parameters):

        """
            This function must have a certain state and only look to one measurement.
        """
        collided_veh = 0
        collided_ped = 0
        collided_oth = 0

        if (measurement.collision_vehicles - self._previous_vehicle_collision) \
                > metrics_parameters['collision_vehicles']['threshold']/2.0:
            collided_veh = 1
        if (measurement.collision_pedestrians - self._previous_pedestrian_collision) \
                > metrics_parameters['collision_pedestrians']['threshold']/2.0:
            collided_ped = 1
        if (measurement.collision_other - self._previous_other_collision) \
                > metrics_parameters['collision_other']['threshold']/2.0:
            collided_oth = 1

        self._previous_pedestrian_collision = measurement.collision_pedestrians
        self._previous_vehicle_collision = measurement.collision_other

        return collided_ped, collided_veh, collided_oth
        
    def _is_agent_stuck(self, measurements, stuck_vec, old_coll):    
        # break the episode when the agent is stuck on a static object
        coll_other = measurements.collision_other 
        coll_other -= old_coll
        otherlane = measurements.intersection_otherlane > 0.4
        offroad = measurements.intersection_offroad > 0.3
        logging.info("offroad: {}, otherlane: {}, coll_other: {}, old_coll: {}".format(offroad,otherlane,coll_other,old_coll))
        
        # if still driving or got unstuck (v > 4km/h)
        if measurements.forward_speed*3.6 > 4:
            cycle_signal(stuck_vec, 0)
            if coll_other: 
                old_coll += coll_other
        elif offroad or otherlane or coll_other:
            cycle_signal(stuck_vec, 1)
        else:
            cycle_signal(stuck_vec, 0)

        return all(stuck_vec), stuck_vec, old_coll
                
    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name,
            metrics_parameters,
            collision_as_failure):
        """
         Run one episode of the benchmark (Pose) for a certain agent.


        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target to reach
            episode_name: The name for saving images of this episode
            metrics_object: The metrics object to check for collisions

        """

        # Send an initial command.
        time.sleep(2)
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        ### Reset CAL agent
        agent.reset_state()

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        measurement_vec = []
        # The vector containing all controls produced on this episode
        control_vec = []
        frame = 0
        distance = 10000
        col_ped, col_veh, col_oth = 0, 0, 0
        fail = False
        success = False
        
        ### own metrics
        stuck_vec = [0] * 60 # measure for 60 frames (6 seconds)
        center_distance_vec = []
        old_collision_value = 0
        direction_vec = []

        #edited ash
        metList = []

        while not fail and not success:

            # Read data from server with the client
            measurements, sensor_data = client.read_data()
            # The directions to reach the goal are calculated.
            directions = self._get_directions(measurements.player_measurements.transform, target)
            # Agent process the data.
            curr_time = time.time()
            control,prediction = agent.run_step(measurements, sensor_data, directions, target)
            control.steer = measurements.player_measurements.autopilot_control.steer
            curr_time = time.time() - curr_time
          
            # Send the control commands to the vehicle
            client.send_control(control)

            # save images if the flag is activated
            self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y

            logging.info("Controller is Inputting:")
            logging.info('Steer = %f Throttle = %f Brake = %f ',
                         control.steer, control.throttle, control.brake)

            current_timestamp = measurements.game_timestamp
            # Get the distance travelled until now

            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])
            # Write status of the run on verbose mode
            logging.info('Distance to target: %f', float(distance))
            
            # Check if reach the target
            col_ped, col_veh, col_oth = self._has_agent_collided(measurements.player_measurements, metrics_parameters)

            
            ### CHANGE TO ORIGINAL CODE #####################
            is_stuck, stuck_vec, old_collision_value = self._is_agent_stuck(measurements.player_measurements, 
                                                                            stuck_vec, old_collision_value)
            
            
            if distance < self._distance_for_success:
                success = True
            elif (current_timestamp - initial_timestamp) > (time_out * 1000):
                fail = True
            elif is_stuck:
                fail = True
            elif collision_as_failure and (col_ped or col_veh or col_oth):
                fail = True

            time_remain = (time_out * 1000 - (current_timestamp - initial_timestamp))/1000
            logging.info('Time remaining: %i m %i s', time_remain/60, time_remain%60)            
            logging.info('')
            
            # Increment the vectors and append the measurements and controls.
            frame += 1
            measurement_vec.append(measurements.player_measurements)
            control_vec.append(control)
            temp=agent.getMetData()
            temp['game_timestamp'] = measurements.game_timestamp
            metList.append(temp)
            print ("stp number: ", frame, " ", temp, " elapsed time: ", curr_time)

            #my code...
            player={}
            peds=[]
            px = measurements.player_measurements.transform.location.x
            py = measurements.player_measurements.transform.location.y

            player['pos_x'] = px
            player['pos_y'] = py

            yaw = measurements.player_measurements.transform.rotation.yaw
            yaw = math.radians(yaw)
            sfile=open('/home/self-driving/Desktop/CARLA_0.8.2/Indranil/CAL-master_copy_new/python_client/_benchmarks_results/sigData.csv','a+')
            pfile=open('/home/self-driving/Desktop/CARLA_0.8.2/Indranil/CAL-master_copy_new/python_client/_benchmarks_results/pData.csv','a+')

            hazard=False

            sig=False
            sigState=''

            speed_post=False
            speed_post_state=''

            veh_inside_box=False
            veh_dist=50.0

            '''
            prediction keys:
            center_distance
            hazard_stop
            red_light
            relative_angle
            speed_sign
            veh_distance 
            '''
            front_axel_x = 2.33

            for a in measurements.non_player_agents:

                if a.HasField('traffic_light'):

                    llx, lly = a.traffic_light.transform.location.x, a.traffic_light.transform.location.y

                    nx, ny = self.getNewCord((px, py), (llx, lly), yaw)

                    if self.inside_a2(nx, ny):
                        print("\n--traffic light found---!!")
                        print("State :", a.traffic_light.state)
                        sig=True
                        sigState=str(a.traffic_light.state)

                if a.HasField('speed_limit_sign'):
                    llx, lly = a.speed_limit_sign.transform.location.x, a.speed_limit_sign.transform.location.y
                    nx, ny = self.getNewCord((px, py), (llx, lly), yaw)
                    if self.inside_a2(nx, ny):
                        print("\n--Speed Sign found---!!")
                        print("Limit :", a.speed_limit_sign.speed_limit * 3.6)
                        speed_post=True
                        speed_post_state=str(
                            math.floor(a.speed_limit_sign.speed_limit * 3.6))


                if a.HasField('pedestrian'):
                    llx, lly = a.pedestrian.transform.location.x, a.pedestrian.transform.location.y
                    ped={}
                    ped['pos_x']=llx
                    ped['pos_y']=lly
                    peds.append(ped)
                    nx, ny = self.getNewCord((px, py), (llx, lly), yaw)
                    if self.inside_a1(nx, ny):
                        print("\n--Pedestrian Hazard---!!")
                        print("Pedestrian in front ")
                        hazard=True

                if a.HasField('vehicle'):
                    llx, lly = a.vehicle.transform.location.x, a.vehicle.transform.location.y
                    nx, ny = self.getNewCord((px, py), (llx, lly), yaw)
                    hlen = a.vehicle.bounding_box.extent.x
                    if self.inside_a3(nx, ny):
                        distan = self.getdis(0 + front_axel_x, 0, nx, ny - hlen)
                        veh_inside_box=True
                        veh_dist=min(distan,veh_dist)
                        print("Vehicle in front distance,",distan," Predicted ",prediction['veh_distance'])

            player['peds']=peds

            #write to file if hazard stop is true in current frame...
            if hazard is True:
                sfile.write(str('{:0>6d}'.format(frame))+",Hazard,"+str(hazard)+
                            ","+str(prediction['hazard_stop'][0])+","+str(prediction['hazard_stop'][1])+"\n")
            else:
                sfile.write(str('{:0>6d}'.format(frame)) + ",Hazard," + str(False) +
                            "," + str(prediction['hazard_stop'][0]) + "," + str(prediction['hazard_stop'][1]) + "\n")
            if sig is True:
                sfile.write(str('{:0>6d}'.format(frame)) + "," + "Traffic," +
                            sigState + "," + str(prediction['red_light'][0])
                            + "," + str(prediction['red_light'][1]) + "\n")
                print("Traffic light.. Actual:", a.traffic_light.state, " Predicted", prediction['red_light'][0])
            else:	
                sfile.write(str('{:0>6d}'.format(frame)) + "," + "Traffic," +
                            '0' + "," + str(prediction['red_light'][0])
                            + "," + str(prediction['red_light'][1]) + "\n")
            if speed_post is True:
                sfile.write(
                    str('{:0>6d}'.format(frame)) + ",SpeedSign," + speed_post_state +
                    "," + str(prediction['speed_sign'][0]) + "," + str(prediction['speed_sign'][1]) + "\n")
            else:
                sfile.write(
                    str('{:0>6d}'.format(frame)) + ",SpeedSign," + '-1' +
                    "," + str(prediction['speed_sign'][0]) + "," + str(prediction['speed_sign'][1]) + "\n")

            if veh_inside_box is True:
                sfile.write(str('{:0>6d}'.format(frame)) + ",Vehicle," + str(veh_dist) +
                            ","+str(prediction['veh_distance'])+"\n")
            else:
                sfile.write(str('{:0>6d}'.format(frame)) + ",Vehicle," + str(50.0) +		
                            "," + str(prediction['veh_distance']) + "\n")

            sfile.write(str('{:0>6d}'.format(frame)) + ",CenterDist," + str(temp['centerDist']) +
                        "," + str(prediction['center_distance']) + "\n")


            sfile.write(str('{:0>6d}'.format(frame)) + ",Angel," + str(yaw) +
                        "," + str(prediction['relative_angle']) + "\n")

            jout=json.dumps(player)
            pfile.write(jout+"\n")

        if success:
            return 1, measurement_vec, control_vec, float(
                current_timestamp - initial_timestamp) / 1000.0, distance,  col_ped, col_veh, col_oth, metList
        return 0, measurement_vec, control_vec, time_out, distance, col_ped, col_veh, col_oth, metList

    def getNewCord(self,origin, point, psi):
        a = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
        b = np.array([ (point[0]-origin[0]) , (point[1]-origin[1])  ])
        x = np.linalg.solve(a, b)
        return x

    def getdis(self,x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def inside_a2(self,x, y):
        front_axel_x = 0
        if (x >= (7.4 + front_axel_x) and x <= (14.0 + front_axel_x)) and (y >= 0.8 and y <= 5.8):
            return True
        return False

    def inside_a1(self,x, y):
        front_axel_x = 2.33
        if (x >= (0.0 + front_axel_x) and x <= (8.2 + front_axel_x)) and (y >= -2.0 and y <= 2.0):
            return True
        return False

    def inside_a3(self,x, y):
        front_axel_x = 2.33
        if (x >= (0 + front_axel_x) and x <= (50 + front_axel_x)) and (y >= -1.6 and y <= 1.6):
            return True
        return False

def run_driving_benchmark(agent,
                          experiment_suite,
                          city_name='Town01',
                          log_name='Test',
                          continue_experiment=False,
                          host='127.0.0.1',
                          port=2000
                          ):
    while True:
        try:

            with make_carla_client(host, port) as client:
                # Hack to fix for the issue 310, we force a reset, so it does not get
                #  the positions on first server reset.
                client.load_settings(CarlaSettings())
                client.start_episode(0)

                # We instantiate the driving benchmark, that is the engine used to
                # benchmark an agent. The instantiation starts the log process, sets

                benchmark = DrivingBenchmark(city_name=city_name,
                                             name_to_save=log_name + '_'
                                                          + type(experiment_suite).__name__
                                                          + '_' + city_name,
                                             continue_experiment=continue_experiment, save_images=True)
                # This function performs the benchmark. It returns a dictionary summarizing
                # the entire execution.

                benchmark_summary = benchmark.benchmark_agent(experiment_suite, agent, client)

                print("")
                print("")
                print("----- Printing results for training weathers (Seen in Training) -----")
                print("")
                print("")
                av = results_printer.print_summary(benchmark_summary, experiment_suite.train_weathers,
                                              benchmark.get_path())
                #open('/home/rsi/Desktop/CAL/PythonClient/_benchmarks_results/' + log_name + '_av_succ_'+ str(av) + '.txt', 'w+')
                print("")
                print("")
                print("----- Printing results for test weathers (Unseen in Training) -----")
                print("")
                print("")

                results_printer.print_summary(benchmark_summary, experiment_suite.test_weathers,
                                              benchmark.get_path())

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(2)
