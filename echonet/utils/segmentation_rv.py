"""Functions for training and running RV segmentation."""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import echonet.utils.model
#import nonechucks as nc
import pdb

import echonet


def run(num_epochs=50,
        modelname="unet-3d-dilated-off",
        pretrained=False,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=4,
        batch_size=16,
        seed=0,
        lr_step_period=None,
        save_segmentation=False,
        block_size=1024,
        run_test=False):
    """Trains/tests segmentation model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        modelname (str, optional): Name of segmentation model. ``unet-3d-dilated-off''
            Name: unet-[3d/2d1d]-[dilated/]-[off/], baseline model ``unet-3d''
            Defaults to ``unet-3d-dilated-off''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/segmentation/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to ``None''.
        save_segmentation (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        block_size (int, optional): Number of frames to segment simultaneously when saving
            videos with segmentation (this is used to adjust the memory usage on GPU; decrease
            this is GPU memory issues occur).
            Defaults to 1024.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
    """
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation-rv", "{}_{}".format(modelname, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = echonet.utils.model.unet3d(is_dilated=('dilated' in modelname),
                   is_off_unit=('off' in modelname),
                   is_2d1d=('2d1d' in modelname))
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    # mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    # tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    # kwargs = {"target_type": tasks,
    #           "mean": mean,
    #           "std": std
    #           }

    # Set up datasets and dataloaders
    #train_dataset = nc.SafeDataset(echonet.datasets.Echo_RV(split="train", fuzzy_aug=True))
    #val_dataset = nc.SafeDataset(echonet.datasets.Echo_RV(split="val"))
    train_dataset = echonet.datasets.Echo_RV(split="train", fuzzy_aug=True)
    val_dataset = echonet.datasets.Echo_RV(split="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        pdb.set_trace()
        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                loss_seg, loss_cons, inter, union = echonet.utils.segmentation_rv.run_epoch(model, dataloaders[phase], phase == "train", optim, device)
                overall_dice = 2 * inter.sum() / (union.sum() + inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss_seg,
                                                                    loss_cons,
                                                                    overall_dice,
                                                                    time.time() - start_time,
                                                                    inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss_seg,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss_seg < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss_seg

            if epoch > 1:
                break

        # Load best weights
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo_RV(split=split)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss_seg, loss_cons, inter, union = echonet.utils.segmentation_rv.run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * inter / (union + inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall\n")
                    for (filename, overall) in zip(dataset.subj_list_all[:len(overall_dice)], overall_dice):
                        g.write("{},{}\n".format(filename, overall))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(inter, union, echonet.utils.dice_similarity_coefficient)))
                f.flush()

    # Saving videos with segmentations
    dataset = echonet.datasets.Echo_RV(split="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    print('End 03/23')
    pdb.set_trace()
    if save_segmentation and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.subj_list_all):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + block_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.circle(int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10)), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation rv.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total_seg = 0.
    total_cons = 0.
    n = 0

    pos = 0
    neg = 0

    model.train(train)

    inter = 0
    union = 0
    inter_list = []
    union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for iter, sample in enumerate(dataloader):
                video = sample['video']
                mask = sample['mask']
                mask_idx = sample['mask_idx']
                # Count number of pixels in/out of human segmentation
                pos += (mask == 1).sum()
                neg += (mask == 0).sum()
                # to gpu
                video = video.to(device)
                mask = mask.to(device)
                mask_idx = mask_idx.to(device)
                # Run prediction for diastolic frames and compute loss
                pred_logit = model(video)
                pred_logit = pred_logit.squeeze(1)
                pred_logit_sel = []
                for bs in range(pred_logit.shape[0]):
                    pred_logit_sel.append(pred_logit[bs:bs+1, mask_idx[bs], ...])
                pred_logit_sel = torch.cat(pred_logit_sel, 0)

                loss_seg = torch.nn.functional.binary_cross_entropy_with_logits(pred_logit_sel, mask, reduction="mean")
                loss_cons = echonet.utils.segmentation_rv.compute_consistency_loss(pred_logit)

                # Compute pixel intersection and union between human and computer segmentations
                inter += np.logical_and(mask.detach().cpu().numpy() > 0., pred_logit_sel.detach().cpu().numpy() > 0.).sum()
                union += np.logical_or(mask.detach().cpu().numpy() > 0., pred_logit_sel.detach().cpu().numpy() > 0.).sum()
                inter_list.extend(np.logical_and(mask.detach().cpu().numpy() > 0., pred_logit_sel.detach().cpu().numpy() > 0.).sum((1, 2, 3)))
                union_list.extend(np.logical_or(mask.detach().cpu().numpy() > 0., pred_logit_sel.detach().cpu().numpy() > 0.).sum((1, 2, 3)))

                # Take gradient step if training
                loss = loss_seg +  loss_cons
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total_seg += loss_seg.item()
                total_cons += loss_cons.item()
                n += video.size(0)
                p = pos / (pos + neg)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} {:.4f} / {:.4f}, {:.4f}".format(total_seg / n, total_cons / n, -p * math.log(p) - (1 - p) * math.log(1 - p), 2 * inter / (union + inter)))
                pbar.update()

                if iter > 10:
                    break

    inter_list = np.array(inter_list)
    union_list = np.array(union_list)

    return (total_seg / n,
            total_cons / n,
            inter_list,
            union_list
            )

def compute_consistency_loss(pred_logit):
    num_pred = pred_logit.shape[1]
    pred_logit_bi = (pred_logit > 0).float()
    pred_1 = pred_logit_bi[:,1:num_pred,...]
    pred_2 = pred_logit_bi[:,0:num_pred-1,...]
    diff = torch.abs(pred_1 - pred_2)
    return diff.mean()

def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
