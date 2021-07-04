from utils import *

# %ls

model_predictions_csv = "/workspace/Milos/dynamic/data/output/segmentation-rv/unet-3d-dilated-off_random_0409/size_test.csv"
data_dir = "/workspace/data/NAS/RV-Milos/RV_data/Tracings"
csv_path = "/workspace/data/NAS/RV-Milos/RV_data/FileList.csv"

# Getting the volumes of RV from model predictions

# +
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
# -

# Returns a dictionary preds that contains FAC for each filename

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

# Returns a dictionary truth that contains human predicted FAC for each filename

# +
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

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Y, Y_pred)
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

plt.plot(Y,Y_pred,'o',alpha=0.6)
plt.title(f"Ejection Fraction Auto vs Men (n={len(Y_pred)}) R^2={r_value**2}, MAE={mae(Y,Y_pred)}")
plt.xlabel("Predictions")
plt.ylabel("Ground Truth")
plt.plot(Y,Y_lin(Y),label="best fit")
plt.plot(Y,Y,label="Y=X")
plt.grid()
plt.legend()
plt.show()
plt.savefig('regression.png')
# -




