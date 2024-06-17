import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Toy dataset of length <N> where each sample are <L> points
    from a circle. There are <C> different circle radius.
    """
    def __init__(self,
                 N:int=10000,
                 L:int=3,
                 C:int=2,
                 conditioned:str="radius", # radius or theta
                 randomized:bool=True
                 ) -> None:
        super().__init__()
        self.N = N
        self.L = L
        self.C = C
        self.points, self.radius, self.theta = self.sample_circle(N, L, C)
        self.conditioned = conditioned
        self.randomized = randomized
    
    def sample_circle(self, n_sample, n_per_sample, n_radius):
        radius = (torch.randint(n_radius, (n_sample, 1)) + 1.) / n_radius
        radius_exp = radius.expand(-1, n_per_sample)
        theta = 2 * torch.pi * torch.rand(n_sample, n_per_sample)

        x = radius_exp * torch.cos(theta)
        y = radius_exp * torch.sin(theta)
        points = torch.stack([x, y], dim=-1)

        return points, radius, (theta - torch.pi) / torch.pi
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        if self.randomized:
            point, radius, theta = self.sample_circle(1, self.L, self.C)


            if self.conditioned == "radius":
                item = {
                    "data" : point[0],
                    "condition" : radius[0], 
                }
            elif self.conditioned == "theta":
                item = {
                    "data" : point[0],
                    "condition" : theta[0], 
                }
            else:
                raise ValueError("Choose the conditionning between radius or theta")
            
        else:
            if self.conditioned == "radius":
                item = {
                    "data" : self.points[index],
                    "condition" : self.radius[index], 
                }
            elif self.conditioned == "theta":
                item = {
                    "data" : self.points[index],
                    "condition" : self.theta[index], 
                }
            else:
                raise ValueError("Choose the conditionning between radius or theta")
            
        return item
    
def get_dataloaders(batch_size:int, points_per_circle:int=3, n_radius:int=2, conditioned:str="theta"):
    """
    Function called to load dataloader in main.
    """
    train_dataset = MyDataset(L=points_per_circle, C=n_radius, conditioned=conditioned)
    test_dataset = MyDataset(N=300, L=points_per_circle, C=n_radius, conditioned=conditioned, randomized=False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # None if no test dataset
    
    batch = next(iter(train_dataloader))
    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader