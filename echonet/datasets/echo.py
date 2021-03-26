"""EchoNet-Dynamic Dataset."""

import pathlib
import os
import collections
import pandas

import numpy as np
import skimage.draw
import torchvision
import echonet
import h5py
import glob

import pdb

class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        super().__init__(root, target_transform=target_transform)

        if root is None:
            root = echonet.config.DATA_DIR

        self.root = pathlib.Path(root)
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(self.root / "FileList.csv") as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(self.root / "Videos"))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), self.root / "Videos"))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(self.root / "Videos" / sorted(missing)[0])

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(self.root / "VolumeTracings.csv") as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

class Echo_RV(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic RV Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR_LIST`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 1.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 512.
        fuzzy_aug (boolean, optional): augmentation by shifting mask indexes
            Defaults to False.
        test_mode (False, optional): cut videos into overlapping clips
            Defaults to False.
        test_overlap_stride (int, optional): mitigate inconsistent segmentation, stride step for window-sliding of clips
            Defaults to 8
    """

    def __init__(self, root=None,
                 split="train",
                 mean=0., std=1.,
                 length=32, max_length=512,
                 period=1,
                 fuzzy_aug=False,
                 test_mode=False,
                 test_overlap_stride=8):
        super().__init__(root)

        if root is None:
            root = echonet.config.DATA_DIR_LIST
        # self.root = pathlib.Path(root)
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.fuzzy_aug = fuzzy_aug
        self.test_mode = test_mode
        self.test_overlap_stride = test_overlap_stride

        # preprocess dataset, save in h5 of the same folder
        for data_path in self.root:
            self.preprocess_dataset(data_path)

        # load h5 file
        self.data_list = []
        self.subj_list = []
        self.subj_list_all = []
        self.subj_list_count = []
        self.subj_list_count_acc = [0]
        for data_path in self.root:
            self.data_list.append(h5py.File(os.path.join(data_path, 'preprocessed.h5'), 'r'))
            self.subj_list.append(np.load(os.path.join(data_path, 'subj_list.npy'), allow_pickle=True).item()[split])
            self.subj_list_count.append(len(self.subj_list[-1]))
            self.subj_list_count_acc.append(self.subj_list_count_acc[-1]+self.subj_list_count[-1])
            self.subj_list_all.extend(self.subj_list[-1])
        print('Total number of videos', self.__len__())

    def __len__(self):
        return np.sum(self.subj_list_count)

    def preprocess_dataset(self, data_path):
        # check whether have preprocessed h5 file
        h5_path = os.path.join(data_path, 'preprocessed.h5')
        if os.path.exists(h5_path):
            print('Exist preprocessed.h5 for ', data_path)
            return
        video_path = os.path.join(data_path, 'Videos')
        mask_path = os.path.join(data_path, 'Tracings')
        file_path = os.path.join(data_path, 'FileList.csv')
        if not os.path.exists(video_path):
            raise ValueError('Missing video folder')
        if not os.path.exists(mask_path):
            print('Missing mask folder')
            os.mkdir(mask_path)

        # prepare h5 file in dict
        data_dict = {}    # subj_id: split, mask index, mask names, video name
        # with FileList.csv of name and split
        if os.path.exists(file_path):
            file = pd.read_csv(file_path)
            for i in tqdm.tqdm(range(len(file))):
                subj_id = file.loc[i, 'FileName']
                split = file.loc[i, 'Split'].lower()
                subj_mask_paths = glob.glob(os.path.join(mask_path, subj_id + '*.png'))
                subj_video_path = os.path.join(video_path, subj_id + '.avi')
                if os.path.exists(subj_video_path) and len(subj_mask_paths) > 0:  # exist video and mask
                    if subj_id in data_dict:
                        print(subj_id)
                        continue
                    subj_list[split].append(subj_id)
                    mask_idxes = []
                    for j, subj_mask_path in enumerate(subj_mask_paths):
                        mask_idx = int(os.path.basename(subj_mask_path).strip('.png').split('_')[1])
                        mask_idxes.append(mask_idx)
                    data_dict[subj_id] = {'split': split, 'mask_paths': subj_mask_paths, 'video_path': subj_video_path, 'mask_idx': mask_idxes}
        # don't have FileList.csv
        else:
            subj_video_paths = glob.glob(os.path.join(video_path, '*.avi'))
            for i, subj_video_path in enumerate(subj_video_paths):
                #pdb.set_trace()
                subj_id = os.path.basename(subj_video_path).split('.')[0]
                if self.test_mode:
                    split = 'test'
                else:
                    if np.random.rand() < 0.2:
                        split = 'val'
                    else:
                        split = 'train'
                subj_mask_paths = glob.glob(os.path.join(mask_path, subj_id + '*.png'))
                if len(subj_mask_paths) > 0:
                    subj_list[split].append(subj_id)
                    mask_idxes = []
                    for j, subj_mask_path in enumerate(subj_mask_paths):
                        mask_idx = int(os.path.basename(subj_mask_path).strip('.png').split('_')[1])
                        mask_idxes.append(mask_idx)
                    data_dict[subj_id] = {'split': split, 'mask_paths': subj_mask_paths, 'video_path': subj_video_path, 'mask_idx': mask_idxes}
        print('Finished preparing', data_path)
        print('Number of videos', len(data_dict))

        # save as h5 file
        h5_file = h5py.File(h5_path)
        mean = np.array([28.951515,28.914696,28.896002], dtype = np.float32)
        std = np.array([47.857174,47.831146,47.798138], dtype = np.float32)
        size = (112, 112)

        for subj_id in tqdm.tqdm(data_dict):
            masks = []
            for subj_mask_path in data_dict[subj_id]['mask_paths']:
                mask = sci.imread(subj_mask_path)
                if len(mask.shape) == 3:
                    mask = mask[np.newaxis,:,:,0]
                masks.append(mask)
            if len(masks) == 0:
                masks = np.zeros((0,112,112))
                is_mask = False
            else:
                masks = np.concatenate(masks, 0)
                masks = masks / 255.0
                is_mask = True

            video = echonet.utils.loadvideo(data_dict[subj_id]['video_path']).astype(np.float32)
            video_norm = (video - mean.reshape(3,1,1,1)) / std.reshape(3,1,1,1)

            if video_norm.shape[2:] != size or masks.shape[1:] != size:
                print(subj_id, ' video or mask do not have the expected size')
                continue

            h5_file.create_dataset(subj_id+'/masks', data=masks)
            h5_file.create_dataset(subj_id+'/video', data=video_norm)
            h5_file.create_dataset(subj_id+'/mask_idx', data=data_dict[subj_id]['mask_idx'])
            h5_file.create_dataset(subj_id+'/is_mask', data=is_mask)

            np.save(os.path.join(data_path, 'subj_list.npy'), subj_list)
        print('Finished saving preprocessed h5 file', data_path)

    def __getitem__(self, idx):
        for dataset_idx, data_path in enumerate(self.root):
            if idx < self.subj_list_count_acc[dataset_idx+1]:
                break
        subj_id = self.subj_list_all[idx]
        # subj_id = self.subj_list[dataset_idx][idx-self.subj_list_count_acc[dataset_idx]]
        video = np.array(self.data_list[dataset_idx][subj_id]['video'])
        masks = np.array(self.data_list[dataset_idx][subj_id]['masks'])
        mask_idx = np.array(self.data_list[dataset_idx][subj_id]['mask_idx'])
        try:
            is_mask = np.array(self.data_list[dataset_idx][subj_id]['is_mask'])
        except:
            is_mask = (masks.shape[0] > 0)

        if self.test_mode:
            video_length = video.shape[1]
            if video_length < self.length:
                pad = np.tile(video[:,-1:,...], (1,self.length-video_length,1,1))
                video = np.concatenate([video, pad], axis=1)
                video_length = self.length
            # get all clips
            video_clip, masks_selected, mask_idx_selected, idx_list = self.generate_all_clips(video, masks, mask_idx)
            video_clip = np.stack(video_clip, 0)
            idx_list = np.stack(idx_list, 0)
            return {'video': video_clip, 'mask': masks_selected, 'mask_idx': mask_idx_selected,
                    'idx_list': idx_list, 'video_length':video.shape[1], 'is_mask':is_mask, 'subj_id':subj_id}
        else:
            # select video period
            if self.split == 'train':
                start = np.random.choice(self.period)
            else:
                start = 0
            video = video[:,start::self.period]
            mask_idx = (mask_idx / self.period).astype(int)
            video_length = video.shape[1]
            mask_idx = np.where(mask_idx>=video_length, video_length-1, mask_idx)

            if video_length < self.length:
                pad = np.tile(video[:,-1:,...], (1,self.length-video_length,1,1))
                video = np.concatenate([video, pad], axis=1)
                video_length = self.length

            # sort masks
            masks, mask_idx = self.sort_mask(masks, mask_idx)

            # get a random clip, keep 2 masks in each clip
            idx_selected, masks_selected, mask_idx_selected = self.select_random_clip(video_length, masks, mask_idx)
            video_clip = video[:, idx_selected, ...]

            # fuzzy augmentation on mask_idx
            if self.fuzzy_aug:
                for i, m_idx in enumerate(mask_idx_selected):
                    if m_idx > 0 and m_idx < video_clip.shape[1]-1:
                        mask_idx_selected[i] = m_idx + np.random.choice([-1,0,1], 1)

            return {'video': video_clip, 'mask': masks_selected, 'mask_idx': mask_idx_selected}

    def sort_mask(self, masks, mask_idx):
        idx_sort = np.argsort(mask_idx)
        masks = masks[idx_sort, ...]
        mask_idx = mask_idx[idx_sort]
        return masks, mask_idx

    def select_random_clip(self, video_length, masks, mask_idx):
        # randomly select a mask to be included in the clip
        mask_idx_idx = np.random.choice(len(mask_idx))
        mask_idx_selected = mask_idx[mask_idx_idx]
        mask = masks[mask_idx_idx]

        # randomly select a start idx for this clip
        start_idx_set = np.arange(max(0, mask_idx_selected-self.length+1), min(mask_idx_selected+1, video_length-self.length+1))
        start_idx = np.random.choice(start_idx_set)
        idx_list = np.arange(start_idx, start_idx+self.length)

        # ensure 2 masks in the clip
        masks_clip = []
        mask_idx_clip = []
        for i, mask_idx_i in enumerate(mask_idx):
            if len(masks_clip) == 2:
                break
            if mask_idx_i in idx_list:
                mask_idx_clip.append(mask_idx_i)
                masks_clip.append(masks[i])
        # if only one clip, make it twice
        if len(masks_clip) == 1:
            masks_clip.append(masks_clip[-1])
            mask_idx_clip.append(mask_idx_clip[-1])
        masks_clip = np.stack(masks_clip, 0)
        mask_idx_clip = np.array(mask_idx_clip) - start_idx
        return idx_list, masks_clip, mask_idx_clip

    def generate_all_clips(self, video, masks, mask_idx):
        video_length = video.shape[1]
        # cut if longer than max_length
        if video_length > self.max_length:
            video = video[:,:self.max_length,...]
            for m in range(len(mask_idx)):
                if mask_idx[m] >= self.max_length:
                    mask_idx = np.delete(mask_idx, m)
                    masks = np.delete(masks, m)
        video_clips = []
        start_idx = 0
        idx_list = []
        while(True):
            if start_idx >= video_length:
                break
            if start_idx + self.length > video_length:
                pad = np.zeros((video.shape[0], start_idx+self.length-video_length, video.shape[2], video.shape[3])).astype(np.float32)
                video = np.concatenate([video, pad], axis=1)
            video_clip = video[:, start_idx:start_idx+self.length, ...]
            video_clips.append(video_clip)
            idx_list.append([np.linspace(start_idx, start_idx+self.length-1, self.length).astype(int)])
            start_idx += self.test_overlap_stride
        return video_clips, masks, mask_idx, idx_list


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
