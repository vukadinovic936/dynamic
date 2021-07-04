from utils import *
import numpy as np
import skimage as skimage
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def mae(y,y_pred):
    return np.mean(np.abs(y-y_pred))

if __name__ == "__main__":
	""" We are hyperoptimizing quality check
		l1 contains a list of all the possible min_pixel values
			min_pixel is the minimum number of pixels that a custer of pixels needs to contain to be considered as a cluster
			we use it to see if the segmentation contains 2 different clusters
		l2 contains a list of all the possible max_speed values
			where max_speed is the maximum area difference of the segmentation from frame to frame
	"""
	l1 = [20]
	l2 = [0.15]

	model_predictions_csv = "/workspace/Milos/dynamic/data/output/segmentation-rv/unet-3d-dilated-off_random_0409/size_test.csv"

	data_dir = "/workspace/data/NAS/RV-Milos/RV_data/Tracings"
	csv_path = "/workspace/data/NAS/RV-Milos/RV_data/FileList.csv"

	perform_check=False
	for pix_t in l1:
		for v_t in l2:

			####### TRAIN ##########
			volume_thru_frames = {}
			mydir = '/workspace/Milos/dynamic/data/output/segmentation-rv/unet-3d-dilated-off_random_0409/'
			with open(f'{mydir}/size.csv', newline='') as csvfile:
				spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
				next(spamreader)
				for row in spamreader:
					entry = row[0].split(',')
					file_name = entry[0].split('.')[0]
					if(file_name in volume_thru_frames):
						volume_thru_frames[file_name].append(int(entry[2]))
					else:
						volume_thru_frames[file_name] = [int(entry[2])]

			ef_dic = {}
			for i in tqdm(volume_thru_frames):
				if perform_check:
					if quality_check(read_video(f"{mydir}/videos/{i}.avi"), pixel_threshold=pix_t, volume_change_tolerance=v_t):
						ar = np.array(volume_thru_frames[i])
						EF = BBEF(ar,limit_peaks=False,persistence=100)
						ef_dic[i] = EF
				else:
						ar = np.array(volume_thru_frames[i])
						EF = BBEF(ar,limit_peaks=False,persistence=100)
						ef_dic[i] = EF

			with open('RV_EF_predictions.csv', 'w') as f:
				for key in ef_dic.keys():
					f.write("%s,%s\n"%(key,ef_dic[key]))

			preds = csv_to_dic("RV_EF_predictions.csv")
			truth = csv_to_dic("../../RV_EF_VAL_ground_truth.csv")

			file_names = list(truth.keys())
			Y_pred = []
			Y = []
			for name in file_names:
				if(name in preds):
					Y.append(float(truth[name]))
					Y_pred.append(float(preds[name]))

			Y_pred = np.array(Y_pred)
			Y = np.array(Y)
			nans = np.where(Y!=Y)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)
			nans = np.where(Y_pred>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)

			nans = np.where(np.abs(Y)>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)
			# I wanna make Y_pred be more similar to Y
			#model = LinearRegression().fit(Y_pred.reshape(-1,1),Y)
			degree=2
			model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
			model.fit(Y_pred.reshape(-1,1),Y)
			####### TEST ##############
			print("Getting the volumes of RV from model predictions")
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
				if perform_check:
					if quality_check(read_video(f"{mydir}/videos_test/{i}.avi"),pixel_threshold=pix_t,volume_change_tolerance=v_t):
						ar = np.array(volume_thru_frames[i])
						EF = BBEF(ar,limit_peaks=False,persistence=100)
						ef_dic[i] = EF
				else:
						ar = np.array(volume_thru_frames[i])
						EF = BBEF(ar,limit_peaks=False,persistence=100)
						ef_dic[i] = EF

			preds = ef_dic

			ef_dic = {}
			print("Getting the volumes of RV from human tracings")
			with open(csv_path,'r') as f:
				next(f)
				for row in tqdm(f):
					if(len(row.split(','))==5):
						file_name,number_tracing,split,frame1,frame2 = row.split(',')
						frame1=int(frame1.split("'")[1])
						frame2=int(frame2.split("'")[1])
						if(split=='test' and number_tracing == '2'):
							file1 = f'{data_dir}/{file_name}_{frame1}.png'
							file2 = f'{data_dir}/{file_name}_{frame2}.png'

							v1 = calculate_volume_from_mask(file1)
							v2 =  calculate_volume_from_mask(file2)
							EF  = np.abs(v1 - v2)/max(v1,v2)
							ef_dic[file_name] = EF
			truth = ef_dic

			Y_pred = []
			Y = []
			for name in truth.keys():
				if(name in preds):
					Y.append(float(truth[name]))
					Y_pred.append(float(preds[name]))

			Y_pred = np.array(Y_pred)
			Y = np.array(Y)
			nans = np.where(Y!=Y)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)
			nans = np.where(Y_pred>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)

			nans = np.where(np.abs(Y)>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)

			# if we have an additional model to use, we can use that here
			#Y = model.predict(Y.reshape(-1,1))
			Y_pred = model.predict(Y_pred.reshape(-1,1))

			nans = np.where(Y_pred>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)

			nans = np.where(np.abs(Y)>0.90)
			Y = np.delete(Y,nans)
			Y_pred = np.delete(Y_pred,nans)
			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Y_pred, Y)
			Y_lin = lambda x: (slope)*x+intercept
			print(f"R^2 is {r_value**2}")
			print(f"MAE is {mae(Y,Y_pred)}")

			plt.plot(np.arange(len(Y_pred)),Y_pred-Y,'o',alpha=0.6)
			plt.plot(np.arange(0,len(Y_pred)), np.repeat(np.nanmean(Y_pred-Y),len(Y_pred)) ,'k--')
			plt.plot(np.arange(0,len(Y_pred)), np.repeat(np.nanmean(Y_pred-Y)+1.96*np.nanstd(Y_pred-Y),len(Y_pred)),'k--', alpha=0.6)
			plt.plot(np.arange(0,len(Y_pred)), np.repeat(np.nanmean(Y_pred-Y)-1.96*np.nanstd(Y_pred-Y),len(Y_pred)),'k--', alpha=0.6)
			plt.title(f"Auto vs Man n={len(Y_pred)}")
			plt.ylabel("Difference in prediction")
			plt.xlabel("Sample ID")
			plt.grid()
			plt.savefig('scatter.png')			
			plt.clf()	

			plt.plot(Y_pred,Y,'o',alpha=0.6)
			plt.title(f"Ejection Fraction Auto vs Men (n={len(Y_pred)}) R^2={r_value**2}, MAE={mae(Y,Y_pred)}")
			plt.xlabel("Predictions")
			plt.ylabel("Ground Truth")
			plt.plot(Y_lin(Y_pred),Y_pred,label="best fit")
			plt.plot(Y,Y,label="Y=X")
			plt.grid()
			plt.legend()
			plt.savefig('regression.png')			

			#with open("logs.csv",'a') as f:
				#f.write(f"{pix_t},{v_t},{R_val}")
