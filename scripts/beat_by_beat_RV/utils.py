import csv
import numpy as np
import matplotlib.pyplot as plt
from persistence1d import RunPersistence
from reconstruct1d import RunReconstruction
import os
from PIL import Image
import cv2 as cv
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import skimage
import skimage.measure
from matplotlib import animation, rc

def FindPeaks(ar,plot=False, persistence=100):
    """
    Finds peaks for each cycle of systole and diastole
    Args:
        ar numpy_array
            contains right ventricle volume in each frame of one video
    Return:
        peaks
            numpy array consisted of values of all local extrema interchagably [min,max,min...]
    """

    #~ This simple call is all you need to compute the extrema of the given data and their persistence.
    ExtremaAndPersistence = RunPersistence(ar)

    #~ Keep only those extrema with a persistence larger than 10.
    Filtered = [t for t in ExtremaAndPersistence if t[1] > persistence]
    Filtered = np.array(np.array(Filtered), dtype="int")
    
    if plot:
        plt.plot(ar)
        plt.scatter(Filtered[:,0],ar[Filtered[:,0]])
        plt.grid()
        plt.show()

    peaks = ar[sorted(Filtered[:,0])]
    min_p = peaks[sorted(np.argsort(peaks)[:4])] 
    max_p = peaks[sorted(np.argsort(-peaks)[:4])]
    #peaks = np.array([min_p[0],max_p[0],min_p[1],max_p[1],min_p[2],max_p[2],min_p[3],max_p[3]])
    return peaks

def BBEF(ar,limit_peaks=False,persistence=100):
    """
    Beat-By-Beat Ejection Fraction calculator
    Takes the mean of the change happening in all beats and calculates Ejection Fraction
    Args:
        ar numpy array
            contains right ventricle volume in each frame of one video
    Retrn:
        EF float
            ejection fraction
    """
    peaks = FindPeaks(ar,False,persistence)
    pts=0
    if limit_peaks:
        pts = min( len(peaks)//2, 4)
    else:
        pts = len(peaks)//2
    EFs = []
    for i in range(0,len(peaks)-1,2):
        # calculate EF for each min max pair
        v1 = peaks[i+1]
        v2 = peaks[i]
        EFs.append( (v1-v2)/v1 )
    meanEF = np.mean(sorted(EFs)[-pts:])
    return meanEF

def keep_digits(s):
    new_s=""
    for i in s:
        if(i.isdigit()):
            new_s+=i
    return new_s

def calculate_volume_from_mask(image_path):
    img = cv.imread(image_path,0)
    v = np.sum(img==255)
    return v

def csv_to_dic(file_path):
    dic = {}
    with open(file_path) as f:
        for line in f:
            name,EF = line.split(',')
            dic[name]=EF
    return dic

def get_R(X,Y):
    return 1 - ( np.sum((X-Y)**2)) / np.sum( (Y - np.mean(Y))**2 ) 

def linear_regression(X,Y,save=False):

    """
    Creates a linear regression and saves to pickle
    Args:
        X numpy array
        Y numpy array
    Return model
    """

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    #print(f"RBEFORE {r_value**2}")
    #print(slope)
    model = LinearRegression().fit( X.reshape(-1,1), Y)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Y,model.predict(X.reshape(-1,1)))
    #print(f"AFTER {r_value**2}")
    #print(slope)
    if save:
        filename = 'lin_weights.sav'
        pickle.dump(model, open(filename, 'wb'))
        print(f"SAVED LEARNABLE PARAMS AT {filename}")
    
    return model

def load_linear(model_path):
    model = pickle.load(open(f"{model_path}/lin_weights.sav", 'rb'))
    return model

def read_video(video_path):
    """
    Returns a 4d numpy array of shape
    T x H x W x C 
        T - time in frames
        H - height
        W - width
        C - channels
    """
    cap = cv.VideoCapture(video_path)
    total=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(1,1)
    ret, frame = cap.read()
    res = []
    for i in range(total):
        cap.set(1,i)
        ret, frame = cap.read()
        res.append(frame)
    res = np.array(res)
    return res

def create_video(orig_image):
    fig, ax = plt.subplots()
    plt.close()
    def animator(N): # N is the animation frame number
        ax.imshow(orig_image[N])
        return ax
    PlotFrames = range(0,orig_image.shape[0],1)
    anim = animation.FuncAnimation(fig,animator,frames=PlotFrames,interval=100)
    rc('animation', html='jshtml') # embed in the HTML for Google Colab
    return anim

def quality_check(seg,pixel_threshold=50,volume_change_tolerance=0.2):
    """
    Takes a 4d numpy array representing a segmented video
    Return:
        True/False
        
    """

    # Criterium 1, can only have 1 connected component
    # Criterium 2, discard images where the change in area is abrupt
    # Criterium 3, discard noisy images (perhaps)
    
    pixel_thres = pixel_threshold
    blue_lr = np.array([100,50,50])
    blue_ur = np.array([140,255,255])
    area=-1
    

    for t in range(seg.shape[0]):
        img=seg[t]
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        seg_t = cv.inRange(hsv, blue_lr, blue_ur)

        cc, n_cc = skimage.measure.label(seg_t == 255, return_num=True)
        count_cc = 0
        for i in range(1, n_cc + 1):
            binary_cc = (cc == i)
            if np.sum(binary_cc) > pixel_thres:
                # If this connected component has more than certain pixels, count it.
                count_cc += 1
        if count_cc >= 2:
            print('The segmentation has at least two connected components with more than {0} pixels '
                  'at time frame {1}.'.format(pixel_thres, t))
            return False
        
        new_area = np.sum(seg_t==255)
        if(area!=-1):
            if np.abs(new_area-area)/max(new_area,area)>volume_change_tolerance:
                print("The segmentation has sharp changes from one frame to another")
                return False
        area=new_area
    return True
def mae(y,y_pred):
    return np.mean(np.abs(y-y_pred))