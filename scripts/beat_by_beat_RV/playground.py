from utils import *
from tqdm import tqdm
ef_dic = {}

model_predictions_csv = "/workspace/Milos/dynamic/data/output/segmentation-rv/unet-3d-dilated-off_random_0409/size_test.csv"

volume_thru_frames = {}
with open(model_predictions_csv,'r') as f:
	next(f)
	for row in f:

		file_name,frame,area,human_large,human_small,computer_small= row.split(',')
		if(file_name in volume_thru_frames):
			volume_thru_frames[file_name].append(int(area))
		else:
			volume_thru_frames[file_name] = [int(area)]

ef_dic = {}
for i in tqdm(volume_thru_frames):
	#if quality_check(read_video(f"{mydir}/videos_test/{i}.avi"),pixel_threshold=pix_t,volume_change_tolerance=v_t):
	ar = np.array(volume_thru_frames[i])
	EF = BBEF(ar,limit_peaks=False,persistence=100)
	ef_dic[i] = EF
	preds = ef_dic

ef_dic = {}