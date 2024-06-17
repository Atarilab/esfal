import os
from typing import List
import numpy as np
import pinocchio as pin
import torch
from torch.utils.data import Dataset, DataLoader

def transform_points(b_T_W, points_w):
    # Add a fourth homogeneous coordinate (1) to each point
    ones = np.ones((points_w.shape[0], 1))
    points_w_homogeneous = np.hstack((points_w, ones))
    # Apply the transformation matrix
    points_b_homogeneous = b_T_W @ points_w_homogeneous.T
    # Convert back to 3D coordinates
    points_b = points_b_homogeneous[:3, :].T
    return points_b

ID_DATA = {
    "pos" : [0,1,2],   # pos
    "q" : [3,4,5,6],   # q
    "v" : [7,8,9],   # v
    "w" : [10,11,12],   # w
    "qj" : list(range(13, 25)),   # qj
    "vj" : list(range(25, 37)),   # vj
}

class CollisionClassifierDataset(Dataset):
    def __init__(self,
                 data_path,
                 exclude: str = "pos",
                 binary: bool = False,
                 use_height: bool = False,
                 noise_std : float = 0.):
        
        data = np.load(data_path)
        self.exclude = exclude
        self.binary = binary
        self.use_height = use_height
        self.noise_std = noise_std
        
        # State when robots lands [x, y, z, quat, v, w, qj, vj]
        self.state = data["state"]
        # Contact positions where robots land [4, 3]
        self.feet_contact = data["contact"]
        # Next target positions [4, 3]
        self.target_contact = data["target"]
        # Current jump collision state [5] (collision [FL_knee, FR_knee, RL_knee, RR_knee, other])
        self.collision = data["collision"]
        
        

        collision_sum = np.sum(self.collision, axis=-1)
        self.id_success = np.where(collision_sum == 0)[0]
        
        new_column = (collision_sum == 0).astype(int)
        self.collision = np.column_stack((new_column, self.collision))
            
        self._process_data()
        
    def _process_data(self):
        
        ### Express contact positions in base frame
        for i in range(len(self.state)-1):
            # b_T_W is world pose in base frame
            b_T_W = pin.XYZQUATToSE3(self.state[i, :7]).inverse()
            self.feet_contact[i] = transform_points(b_T_W, self.feet_contact[i])
            self.target_contact[i] = transform_points(b_T_W, self.target_contact[i])
        
        self.state = torch.tensor(self.state, dtype=torch.float32)
        self.feet_contact = torch.tensor(self.feet_contact, dtype=torch.float32).reshape(-1, 12)
        self.target_contact = torch.tensor(self.target_contact, dtype=torch.float32).reshape(-1, 12)
        self.collision = torch.tensor(self.collision, dtype=torch.float32)
        
        all_id = np.arange(self.state.shape[1])
        id_to_exclude = []
        for name in self.exclude.split("_"):
            id_to_exclude.extend(ID_DATA[name])
        id_to_keep = torch.from_numpy(np.setdiff1d(all_id, id_to_exclude)).long().unsqueeze(0)
        self.state = torch.take_along_dim(self.state, id_to_keep, dim=-1)
        
        
    def __len__(self):
        return len(self.id_success)

    def __getitem__(self, idx):
        
        i = self.id_success[idx]
        input = torch.concatenate(
            (self.state[i], self.feet_contact[i], self.target_contact[i])
        )
        
        if self.binary:
            target = self.collision[i + 1, 0].unsqueeze(-1)
        else:
            target = self.collision[i + 1]
            
        # Add noise to the input data
        if self.noise_std > 0:
            input += torch.rand_like(input) * self.noise_std

        item = {
            "input": input,
            "target": target
        }
        return item

class StateEstimationDataset(Dataset):
    def __init__(self,
                 data_path,
                 exclude: str = "pos_vj",
                 use_height: bool = False,
                 noise_std : float = 0.):
        
        self.exclude = exclude
        self.use_height = use_height
        self.noise_std = noise_std

        data = np.load(data_path)
        self.state = data["state"]
        self.feet_contact = data["contact"]
        self.next_feet_contact = np.empty_like(self.feet_contact)
        self.target_contact = data["target"]
        collision = data["collision"]

        # Filter out the samples with collisions
        collision_sum = np.sum(collision, axis=-1)
        id_success = np.where(collision_sum == 0)[0]
        # id_failure = np.where(collision_sum > 0)[0]
        # id_first_jump = np.concatenate(([0], id_failure + 1), axis=0)
        
        # Valid id are jump that didn't fail and that are not first jump
        self.valid_id = id_success
        
        self._process_data()
        
    def _process_data(self):

        ### Express contact positions in base frame
        for i in range(len(self.state)-1):
            # b_T_W is world pose in base frame
            b_T_W = pin.XYZQUATToSE3(self.state[i, :7]).inverse()
            self.feet_contact[i] = transform_points(b_T_W, self.feet_contact[i])
            self.next_feet_contact[i] = transform_points(b_T_W, self.feet_contact[i+1])
            self.target_contact[i] = transform_points(b_T_W, self.target_contact[i])
        
        self.state = torch.tensor(self.state, dtype=torch.float32)
        self.feet_contact = torch.tensor(self.feet_contact, dtype=torch.float32).reshape(-1, 12)
        self.next_feet_contact = torch.tensor(self.next_feet_contact, dtype=torch.float32).reshape(-1, 12)
        self.target_contact = torch.tensor(self.target_contact, dtype=torch.float32).reshape(-1, 12)
                
        all_id = np.arange(self.state.shape[1])
        id_to_exclude = []
        for name in self.exclude.split("_"):
            id_to_exclude.extend(ID_DATA[name])
        id_to_keep = torch.from_numpy(np.setdiff1d(all_id, id_to_exclude)).long().unsqueeze(0)
        self.state = torch.take_along_dim(self.state, id_to_keep, dim=-1)
        
        
    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, idx):
        
        i = self.valid_id[idx]
        input = torch.concatenate(
            # current state, current contact, next contact reached (all in CURRENT base frame)
            (self.state[i], self.next_feet_contact[i], self.feet_contact[i])
        ).unsqueeze(0)
        
        target = torch.concatenate(
            # next state, input to reach next contact (all in CURRENT base frame)
            (self.state[i+1], self.target_contact[i])
        ).unsqueeze(0)
        
        # Add noise to the input data
        if self.noise_std > 0:
            input += torch.rand_like(input) * self.noise_std

        item = {
            "input": input,
            "target": target
        }
        
        return item

class DatasetFactory():
    DATASETS = {
        "classifier": CollisionClassifierDataset,
        "regressor": StateEstimationDataset,
    }

    @staticmethod
    def get(training_mode:str, params:dict):
        dataloader_factory = DatasetFactory.DATASETS.get(training_mode.lower())
        if not dataloader_factory:
            available_str = " ".join(DatasetFactory.DATASETS.keys())
            raise ValueError(f"Invalid training mode. Should be in {available_str}")
        return dataloader_factory(**params)
    
def get_dataloaders(name : str, batch_size : int, **kwargs):
    """
    Function called to load dataloader in main.
    """
    data_path = kwargs.get("data_path", "")
    assert data_path != "", "data_path argument not specified"
    
    train_data_path = os.path.join(data_path, "train", "data.npz")
    test_data_path = os.path.join(data_path, "test", "data.npz")
    
    train_dataset, test_dataset = None, None
    if os.path.exists(train_data_path):
        kwargs["data_path"] = train_data_path
        train_dataset = DatasetFactory.get(name, kwargs)
        print("Train samples:", len(train_dataset))
    if os.path.exists(test_data_path):
        kwargs["data_path"] = test_data_path
        if kwargs.get("noise_std", None) is not None:
            kwargs["noise_std"] = 0.
        test_dataset = DatasetFactory.get(name, kwargs)
        print("Test samples:", len(test_dataset))

    train_dataloader, test_dataloader = None, None
    if train_dataset:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # None if no test dataset
    
    batch = next(iter(train_dataloader))
    
    print("Train batch shape:")
    for (k, v) in batch.items():
        print(k, ":", list(v.shape))
    
    return train_dataloader, test_dataloader