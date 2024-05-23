import numpy as np
import os
from typing import Tuple

class SteppingStonesEnv:
    CACHE_DIR = "cache"
    DEFAULT_XLM_FILE = "stepping_stones.xml"
    DEFAULT_STONE_SHAPE = "box"               # box or cylinder
    DEFAULT_STONE_HEIGHT = 0.1                # m
    DEFAULT_STONE_RGBA = [0.2, 0.2, 0.25, 1.]  # [R, G, B, A]
    
    def __init__(self,
                 grid_size: Tuple[int, int] = (9, 9),
                 spacing: Tuple[float, float] = (0.2, 0.2),
                 size_ratio: Tuple[float, float] = (0.6, 0.6),
                 randomize_pos_ratio: float = False,
                 randomize_height_ratio: float = 0.,
                 **kwargs) -> None:
        """
        Define stepping stones locations on a grid. 

        Args:
            - grid_size (Tuple[int, int], optional): Number of stepping stones node (x, y).
            - spacing (Tuple[float, float], optional): Spacing of the center of the stones (x, y).
            - size_ratio (Tuple[float, float], optional): Size ratio of the stepping 
            stone and the spacing.
            size_ratio[0] * spacing and size_ratio[1] * spacing. Defaults to False.
            - randomize_pos (float, optional): Randomize stepping stone location within it's area 
            without collision. Ratio to the max displacement. Defaults to 0, no displacement.
            - randomize_height_ratio (float, optional): Randomize height between [(1-ratio)*h, (1+ratio)*h].
        """
        self.grid_size = grid_size
        self.randomize_pos_ratio = randomize_pos_ratio
        self.spacing = list(spacing)
        self.size_ratio = size_ratio
        self.randomize_height_ratio = randomize_height_ratio
        
        # Optional args
        self.shape = None
        self.height = None
        optional_args = {
            "shape" : SteppingStonesEnv.DEFAULT_STONE_SHAPE,
            "height" : SteppingStonesEnv.DEFAULT_STONE_HEIGHT,
            "rgba" : SteppingStonesEnv.DEFAULT_STONE_RGBA,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        self.I = self.grid_size[0]
        self.J = self.grid_size[1]
        self.N = self.I * self.J
        self.id_to_remove = np.array([], dtype=np.int32)
        
        self._init_center_location()  
        self._init_size()
        self._randomize_center_location()
        
    def _init_center_location(self) -> None:
        """
        Initialize the center locations of the stepping stones.
        """        
        ix = np.arange(self.I) - self.I // 2
        iy = np.arange(self.J) - self.J // 2
        z = np.full(((self.N, 1)), self.height)
        z += (np.random.rand(self.N, 1) - 0.5) * 2 * self.randomize_height_ratio * self.height

        nodes_xy = np.dstack(np.meshgrid(ix, iy)).reshape(-1, 2)
        stepping_stones_xy = nodes_xy * np.array([self.spacing])
        self.stepping_stones = np.hstack((stepping_stones_xy, z))

        
    def _init_size(self) -> None:
        """
        Init the size of the stepping stones.
        """
        size_ratio = np.random.uniform(
            low=self.size_ratio[0],
            high=self.size_ratio[1],
            size=self.N
            )
        self.size = size_ratio * max(self.spacing)
        
    def _randomize_center_location(self) -> None:
        """
        Randomize the center of the stepping stones locations.
        """
        max_displacement_x = self.spacing[0] - self.size
        max_displacement_y = self.spacing[1] - self.size
        
        dx = np.random.rand(self.N) * max_displacement_x * self.randomize_pos_ratio
        dy = np.random.rand(self.N) * max_displacement_y * self.randomize_pos_ratio

        self.stepping_stones[:, 0] += dx
        self.stepping_stones[:, 1] += dy
        
    def remove_random(self, N_to_remove: int, keep: list[int] = []) -> None:
        """
        Randomly remove stepping stones.
        
        Args:
            N_to_remove (int): Number of box to remove.
            keep (list[int]): id of the stepping stones to keep
        """
        # 0 probability for id in keep
        probs = np.ones((self.N,))
        probs[keep] = 0.
        probs /= np.sum(probs)
        
        self.id_to_remove = np.random.choice(self.N, N_to_remove, replace=False, p=probs)
        
    def _xml_single_geom_string(self, id: int) -> str:
        """
        Returns the xml string in mujoco format for the stone <id>.

        Args:
            id (int): id of the stepping stone.
        """
        assert self.shape in ["box", "cylinder"], "Stepping stone shape should be 'box' or 'cylinder'"

        if self.shape == "box":
            size_x, size_y, size_z = self.size[id]/2., self.size[id]/2., self.stepping_stones[id, 2]/2.
            r, g, b, a = self.rgba
            x, y, z = self.stepping_stones[id, 0], self.stepping_stones[id, 1], self.stepping_stones[id, 2]/2.
            string = f"""<geom type="box" name="box_{id}" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" pos="{x:.3f} {y:.3f} {z:.3f}" rgba="{r} {g} {b} {a}"/>"""
            
        elif self.shape == "cylinder":
            size_radius, size_length = self.size[id]/2., self.stepping_stones[id, 2]/2.
            r, g, b, a = self.rgba
            x, y, z = self.stepping_stones[id, 0], self.stepping_stones[id, 1], self.stepping_stones[id, 2]/2.
            string = f"""
            <geom type="cylinder" name="cylinder_{id}" size="{size_radius:.3f} {size_length:.3f}" pos="{x:.3f} {y:.3f} {z:.3f}" rgba="{r} {g} {b} {a}"/>"""
        string = "\t\t" + string + "\n"
        return string
        
    def _xml_geom_string(self) -> str:
        """
        Genrate xml string corresponding to the stepping stones configuration
        in MuJoCo format.
        """
        string = ""
        for id in filter(
            lambda id : id not in self.id_to_remove,
            range(self.N)
            ):
            string += self._xml_single_geom_string(id)
            
        return string

    def mj_xml_string(self) -> str:
        """
        Return the full xml string for the stepping stones environment
        in MuJoCo format.
        """
        
        xml_string =\
        f"""<mujoco model="stepping stones scene">
            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="1. 1. 1." specular="0.2 0.2 0.2" active="1"/>
                <global azimuth="-130" elevation="-20"/>
            </visual>
            <worldbody>\n\t<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n{self._xml_geom_string()}\t    </worldbody>
        </mujoco>"""
        xml_string.replace("        ", "")
        
        return xml_string
    
    def save_xml(self, saving_path: str = "") -> None:
        """
        Save the xml string at <saving path>

        Args:
            saving_path (str): path to save the xml description file at.
        """
        if saving_path == "":
            current_dir, _ = os.path.split(__file__)
            saving_path = os.path.join(
                current_dir,
                SteppingStonesEnv.CACHE_DIR,
                SteppingStonesEnv.DEFAULT_XLM_FILE
                )
        
        # Check extension
        XML_EXT = ".xml"
        _, ext = os.path.splitext(saving_path)
        if ext == '':
            saving_path += XML_EXT
        elif ext != XML_EXT:
            saving_path.replace(ext, XML_EXT)
        
        # Erase file if exists
        if os.path.exists(saving_path):
            os.remove(saving_path)
            
        # Save file
        file = open(saving_path, 'w')
        xml_string = self.mj_xml_string()
        file.write(xml_string)
        file.close()
        
    def include_env(self, mj_scene: str) -> str:
        """
        Include the stepping stones environment in a predefined scene.

        Args:
            mj_scene (str): path or string of the predifined xml scene.

        Returns:
            str: modfied MuJoCo scene as a xml string.
        """
        # If xml path
        if os.path.exists(mj_scene):
            file = open(mj_scene)
            lines = file.readlines()
            file.close()
        # If xml string
        else:
            lines = mj_scene.split('\n')
            
        # Find <mujoco> balise
        for i, line in enumerate(lines):
            if "<mujoco" in line:
                break
        
        # Add <include> line after mujoco balise
        self.save_xml() # Generate new xml file
        current_dir, _ = os.path.split(__file__)
        mj_stepping_stones_path = os.path.join(
            current_dir,
            SteppingStonesEnv.CACHE_DIR,
            SteppingStonesEnv.DEFAULT_XLM_FILE
            )
        
        include_str = f"""   <include file="{mj_stepping_stones_path}"/>"""
        lines.insert(i + 1, include_str)
        xml_string = '\n'.join(lines)
        
        return xml_string
    
    def get_closest(self, positions_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the indices and positions of the stepping stones closest to 
        each position in <position_xyz>. 

        Args:
            positions_xyz (np.ndarray): array of N 3D positions [N, 3].
        """
        # Calculate squared distances between each point of positions_xyz
        squared_distances = np.sum(
            (self.stepping_stones[:, np.newaxis] - positions_xyz)**2,
            axis=2)

        # Find the indices of the closest points
        closest_indices = np.argmin(squared_distances, axis=0)
        
        # Extract the closest points from stepping stones
        closest_points = self.stepping_stones[closest_indices]

        return closest_indices, closest_points
        
    
if __name__ == "__main__":
    ### TESTING
    stones_env = SteppingStonesEnv(
        randomize_height_ratio=0.2,
        shape="cylinder"
    )
    stones_env.remove_random(10, keep=[22, 43])
    
    stones_env.save_xml()
    xml_string = stones_env.include_env("/home/project/example/environment/default_scene.xml")
    
    
    import mujoco
    from mujoco import viewer

    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    mj_data = mujoco.MjData(mj_model)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while (viewer.is_running()):
            viewer.sync()