import torch
import torchvision
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageNetColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, image_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a grayscale image.
            target_transform (callable, optional): Optional transform to be applied
                on the color image.
            image_size (tuple): Desired size of the images after resizing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image_rgb = Image.open(img_name).convert('RGB')


        image_gray = image_rgb.convert('L')

        # Apply resizing
        image_gray = image_gray.resize(self.image_size)
        image_rgb = image_rgb.resize(self.image_size)

        # Apply transformations if any
        if self.transform:
            image_gray = self.transform(image_gray)
        if self.target_transform:
            image_rgb = self.target_transform(image_rgb)

        return image_gray, image_rgb


transform_gray = transforms.Compose([
    transforms.ToTensor(),  # Convert grayscale image to tensor
])

transform_rgb = transforms.Compose([
    transforms.ToTensor(),  # Convert RGB image to tensor
])

# Define the dataset and dataloader
root_dir = '/home/pragay/image_colorization/example image/archive'
dataset = ImageNetColorizationDataset(root_dir=root_dir, 
                                      transform=transform_gray, 
                                      target_transform=transform_rgb)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


for i, (grayscale, color) in enumerate(data_loader):
    print(f"Batch {i}: Grayscale shape: {grayscale.shape}, Color shape: {color.shape}")
    # Convert the first grayscale and color image of the batch to NumPy arrays for visualization
    grayscale_img = grayscale[0].squeeze().numpy()  # Remove channel dimension from grayscale (shape: H, W)
    color_img = np.transpose(color[0].numpy(), (1, 2, 0))  # Transpose to (H, W, C) from (C, H, W)
    
    # Plot grayscale and color images side by side
    plt.figure(figsize=(10, 5))
    
    # Grayscale image
    plt.subplot(1, 2, 1)
    plt.title("Grayscale Image")
    plt.imshow(grayscale_img, cmap='gray')
    plt.axis('off')
    
    # Color image
    plt.subplot(1, 2, 2)
    plt.title("Color Image")
    plt.imshow(color_img)
    plt.axis('off')
    
    # Show the plot
    plt.savefig(f"output_image_{i}.png")
    
'''    # Break after showing one pair of images
    break'''

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))