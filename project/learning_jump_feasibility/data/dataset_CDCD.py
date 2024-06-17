import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Toy dataset of length <N> where each sample are <C*L> points
    from circles with <C> different circle radius. <L> points have the same
    argument theta.
    """
    def __init__(self,
                 N:int=10000,
                 L:int=5,
                 C:int=2,
                 ) -> None:
        super().__init__()
        self.N = N
        self.L = L
        self.C = C
        self.points, self.radius, self.theta = self.sample_circle(N, L, C)
    
    def sample_circle(self, n_sample, n_per_sample, n_radius):
        radius = (torch.arange(1, n_radius + 1) / n_radius).reshape(1, 1, n_radius)
        radius_exp = radius.expand(n_sample, -1, -1)
        theta = 2 * torch.pi * torch.rand((n_sample, n_per_sample, 1))

        x = radius_exp * torch.cos(theta)
        y = radius_exp * torch.sin(theta)
        points = torch.stack([x, y], dim=-1)

        theta_norm = (theta - torch.pi) / torch.pi
        theta_norm = theta_norm.squeeze()
        return points, radius, theta_norm
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        
        points = self.points[index]
        theta = self.theta[index]
        theta_index = torch.randint(0, self.L, (1,)).long()
        radius_index = torch.randint(0, self.C, (1,)).long()
        theta_goal = theta[theta_index]
        condition = torch.cat([points.reshape(-1), theta_goal], dim=-1)

        item = {
            "data" : points[theta_index, radius_index],
            "condition" : condition, 
            "index" : self.C * theta_index + radius_index,
        }

        return item
    
def get_dataloaders(batch_size:int, points_per_circle:int=3, n_radius:int=2):
    """
    Function called to load dataloader in main.
    """
    train_dataset = MyDataset(L=points_per_circle, C=n_radius)
    test_dataset = MyDataset(N=300, L=points_per_circle, C=n_radius)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # None if no test dataset
    
    batch = next(iter(train_dataloader))
    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    dataloader, _ = get_dataloaders(2, 5, 2)

    batch = next(iter(dataloader))

    data = batch["data"]
    condition = batch["condition"]
    index = batch["index"]

    points_2d = condition[:, :-1]
    theta_goal = condition[:, -1]

    fig, ax = plt.subplots(1)
    ax.scatter(*torch.split(data.squeeze(), 1, dim=-1), s=50)
    ax.scatter(*torch.split(condition[:, :-1].reshape(-1, 2), 1, dim=-1), s=5, c ="k")

    for theta in theta_goal:
        print(theta)
        theta *= torch.pi
        theta += torch.pi

        ax.arrow(0., 0., torch.cos(theta).data, torch.sin(theta).data)

    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])

    limits = 1.5 #1.1 * intermediate_steps.max()
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)

    plt.show()