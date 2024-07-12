from typing import Any, Callable
import mujoco
from mujoco import viewer
import time
import numpy as np
import itertools
import cv2

from .abstract.robot import RobotWrapperAbstract
from .abstract.controller import ControllerAbstract
from .abstract.data_recorder import DataRecorderAbstract

class Simulator(object):
    def __init__(self,
                 robot: RobotWrapperAbstract,
                 controller: ControllerAbstract,
                 data_recorder: DataRecorderAbstract = None,
                 ) -> None:
        
        self.robot = robot
        self.controller = controller
        self.data_recorder = (data_recorder
                              if data_recorder != None
                              else DataRecorderAbstract()
                              )
        self.sim_dt = self.robot.mj_model.opt.timestep
                        
        self.sim_step = 0
        self.simulation_it_time = []
        self.q, self.v = None, None
        self.visual_callback_fn = None
        self.verbose = False
        self.stop_sim = False
        
    def _reset(self) -> None:
        """
        Reset flags and timings.
        """
        self.sim_step = 0
        self.simulation_it_time = []
        self.verbose = False
        self.stop_sim = False
        self.robot.collided = False
        self.controller.diverged = False
        
    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        self.q, self.v = self.robot.get_pin_state()
        
        # Record data
        self.data_recorder.record(self.q,
                                  self.v,
                                  self.robot.mj_data)
        
        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(self.q,
                                              self.v,
                                              robot_data = self.robot.mj_data)
        # Apply torques
        self.robot.send_mj_joint_torques(torques)

        # MuJoCo sim step
        self.robot.step()
        self.sim_step += 1
        
        # TODO: Add external disturbances
        
    def _simulation_step_with_timings(self,
                                      real_time: bool,
                                      ) -> None:
        """
        Simulation step with time keeping and timings measurements.
        """
        
        step_start = time.time()
        self._simulation_step()
        step_duration = time.time() - step_start
        
        self.simulation_it_time.append(step_duration)
        
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.sim_dt - step_duration
        if real_time and time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
    def _stop_sim(self) -> bool:
        """
        True if the simulation has to be stopped.

        Returns:
            bool: stop simulation
        """        
        if self.stop_on_collision and (self.robot.collided or self.robot.is_collision()):
            if self.verbose: print("/!\ Robot collision")
            return True

        if self.stop_sim:
            if self.verbose: print("/!\ Simulation stopped")
            return True
        
        if self.controller.diverged:
            if self.verbose: print("/!\ Controller diverged")
            return True
        
        return False
        
    def run(self,
            simulation_time: float = -1.,
            use_viewer: bool = True,
            visual_callback_fn: Callable = None,
            **kwargs,
            ) -> None:
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            - simulation_time (float, optional): Simulation time in second.
            Unlimited if -1. Defaults to -1.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
            - viewer (bool, optional): Use viewer. Defaults to True.
            - verbose (bool, optional): Print timing informations.
            - stop_on_collision (bool, optional): Stop the simulation when there is a collision.
        """
        real_time = kwargs.get("real_time", use_viewer)
        self.verbose = kwargs.get("verbose", True)
        self.stop_on_collision = kwargs.get("stop_on_collision", False)
        self.visual_callback_fn = visual_callback_fn
        
        record_video = kwargs.get("record_video", False)
        if record_video:
            video_save_path = kwargs.get("video_save_path", "test.mp4")
            fps = kwargs.get("fps", 30)
            playback_speed = kwargs.get("playback_speed", 1.0)
            frame_height = kwargs.get("frame_height", 1080)
            frame_width = kwargs.get("frame_width", 1920)
            
            renderer = mujoco.Renderer(self.robot.mj_model, height=frame_height, width=frame_width)
            frames_count = 0
            VideoWriter = cv2.VideoWriter(
                video_save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (renderer.width, renderer.height)
            )
            
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance, cam.azimuth, cam.elevation = 1.35, -130, -20
            cam.lookat[0], cam.lookat[1], cam.lookat[2] = 0.0, 0.0, 0.2
        
        if self.verbose:
            print("-----> Simulation start")
        
        self.sim_step = 0
        
        # With viewer
        if use_viewer:
            with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
                
                  # Enable wireframe rendering of the entire scene.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                
                viewer.sync()
                sim_start_time = time.time()
                while (viewer.is_running() and
                       (simulation_time < 0. or
                        self.sim_step < simulation_time * (1 / self.sim_dt))
                       ):
                    self._simulation_step_with_timings(real_time)
                    self.update_visuals(viewer)
                    viewer.sync()
                    
                    if self._stop_sim():
                        break

        # No viewer
        else:
            sim_start_time = time.time()
            while (simulation_time < 0. or self.sim_step < simulation_time * (1 / self.sim_dt)):
                self._simulation_step_with_timings(real_time)
                
                if record_video and frames_count < self.robot.mj_data.time * fps / playback_speed:
                    renderer.update_scene(self.robot.mj_data, cam)
                    frames_count += 1
                    image = renderer.render()
                    # print(image.shape)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    VideoWriter.write(image)
                
                if self._stop_sim():
                    break
    
        if self.verbose:
            print(f"-----> Simulation end\n")
            sum_step_time = sum(self.simulation_it_time)
            mean_step_time = sum_step_time / len(self.simulation_it_time)
            total_sim_time = time.time() - sim_start_time
            print(f"--- Total optimization step time: {sum_step_time:.2f} s")
            print(f"--- Mean simulation step time: {mean_step_time*1000:.2f} ms")
            print(f"--- Total simulation time: {total_sim_time:.2f} s")

        # Reset flags
        self._reset()
        
        if record_video:
            VideoWriter.release()
        
    def update_visuals(self, viewer) -> None:
        """
        Update visuals according to visual_callback_fn.
        
        Args:
            viewer (fn): Running MuJoCo viewer.
        """
        if self.visual_callback_fn != None:
            try:
                self.visual_callback_fn(viewer, self.sim_step, self.q, self.v, self.robot.mj_data)
                
            except Exception as e:
                if self.verbose:
                    print("Can't update visual geometries.")
                    print(e)