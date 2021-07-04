import torch
import echonet
from echonet.utils.segmentation_rv import run as run1
from echonet.utils.video_rv import run as run2
from tqdm import tqdm
import numpy as np
# Set device for computations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set up model
print(device)
#model = echonet.utils.model.unet3d(is_dilated=True, is_off_unit=True,is_2d1d=False)
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load("/workspace/data/NAS/RV-Milos/output/segmentation-rv/unet-3d-dilated-off_random/best.pt", map_location=device)
#model.load_state_dict(checkpoint['state_dict'])
#model.eval()

## RUn the test once the gpu frees, this will maybe give you the segmented videos or..
# RUN 1st model
#run(num_epochs=50,
#        modelname="unet-3d-dilated-off_random_0409",
#        pretrained=True,
#        output="/workspace/data/NAS/RV-Milos/output/segmentation-rv/unet-3d-dilated-off_random_0409",
#        device=device,
#        n_train_patients=None,
#        num_workers=2,
#        batch_size=16,
#        seed=0,
#        lr_step_period=None,
#        save_segmentation=True,
#        block_size=1024,
#        run_test=True,
#        test_only=True)

# Run 2nd model
run2(num_epochs=45,
        modelname="r2plus1d_18",
        tasks="EF",
        frames=32,
        period=2,
        pretrained=True,
        output="/workspace/data/NAS/RV-Milos/output/",
        device=device,
        n_train_patients=None,
        num_workers=5,
        batch_size=8,
        seed=0,
        lr_step_period=15,
        run_test=False)

#dataset = echonet.datasets.Echo_RV_Video(split="train", length=30)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
#n = 0  
#s1 = 0.
#s2 = 0.
#for (x,mask) in tqdm(dataloader):
#        x = x.transpose(0, 1).contiguous().view(3, -1)
#        n += x.shape[1]
#        s1 += torch.sum(x, dim=1).numpy()
#        s2 += torch.sum(x ** 2, dim=1).numpy()
#        mean = s1 / n  # type: np.ndarray
#        std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray
#
#        mean = mean.astype(np.float32)
#        std = std.astype(np.float32)
#        break 
#print(mask)