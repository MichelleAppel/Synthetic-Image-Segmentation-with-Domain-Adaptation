import torch

class DomainTransfer:
    def __init__(self, generator_class, checkpoint_path):
        """
        Initialize DomainTransfer.
        
        Parameters:
            generator_class (Type[torch.nn.Module]): The generator model class.
            checkpoint_path (str): Path to the checkpoint file for the generator model.
        """
        # Initialize the generator model.
        self.generator = generator_class
        
        # Load checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint['state_dict'])
        
        # Set the generator to evaluation mode.
        self.generator.eval()

    def apply_transfer(self, batch):
        """
        Applies domain transfer to a batch of images.
        
        Parameters:
            batch (Tuple[torch.Tensor, ...]): The original batch from dataloader.
        
        Returns:
            Tuple[torch.Tensor, ...]: The batch with domain transferred images.
        """
        with torch.no_grad():
            transferred_images = self.generator(batch[0])
        return (transferred_images, *batch[1:])

    def generate(self, dataloader):
        """
        Yields batches of domain transferred images from a data loader.

        Parameters:
            dataloader (torch.utils.data.DataLoader): The original dataloader.
        
        Yields:
            torch.Tensor: The domain-transferred batch.
        """
        for batch in dataloader:
            yield self.apply_transfer(batch)