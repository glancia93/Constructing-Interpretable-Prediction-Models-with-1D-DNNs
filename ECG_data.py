import pandas as pd
import numpy as np
####
import scipy.io
import scipy.stats as stats
import scipy.signal as signal
####
import emd
import pywt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tslearn.piecewise import PiecewiseAggregateApproximation

class ECG_Dataset:

    """A python class to load ECG data"""

    def __init__(self):
        
        self.hello = "HELLO!"
        return
        
    def plot_Atrial_Fibrillation(self, kind= "regular"):
        
        #
        root = "/Users/pinguino/Documents/ECG Categorization/Data"
        recods = open(root+"/Records")

        ###
        count = 0
        count_instance = 0
        recodrs_lines = recods.readlines()
        
        ### black boxes to fill
        age = []
        sex = []
        wave_feat = []
        data = []
        
        if kind== "regular":
            rythm= 426783006
        elif kind== "myocardial_infarction":
            rythm= 164865005
        elif kind=="atrial_fibrillation":
            rythm= 164889003
        elif kind=="sinus_bradicardya":
            rythm= 426177001
        
        print(rythm)
        
        ###sex dictionary
        sex_dict = {'F':0, 'M':1}

        ###Butter filter
        ecg_butter = signal.butter(N=3, btype='lowpass', Wn=.25)

        ### COLLECT DATA
        for item in recodrs_lines:

            path = root+"/"+item.rstrip('\n')
            path_records = root+"/"+item.rstrip('\n')+"Records"

            ###name mat_files
            name_file = pd.read_csv(path_records, header=None).values

            ###
            for name_path in name_file:
                File_Name = path+name_path[0]

                ##get the rawdata
                data_ = scipy.io.loadmat(File_Name)['val']

                File_Read = open(File_Name+'.hea')
                File_Read_List = File_Read.readlines()

                count += 1

                ###get all the "wave-features"
                try:
                    wave_features = File_Read_List[15][5:].split(',')
                    wave_features = [int(item.rstrip('\n')) for item in wave_features]
                    wave_feat.append(wave_features)
                except: 
                    wave_features = ['000']
                    wave_feat.append(wave_features)
                
                if np.any(np.isin(wave_features, rythm) == True):
                   
                    return data_.T
        
        
    def denoise_signal(self, X, dwt_transform, cutoff_low, cutoff_high):
        
        coeffs = pywt.wavedec(X, dwt_transform, level=None, mode= "zero")  

        # scale 0 to cutoff_low 
        for ca in range(0,cutoff_low):
            coeffs[ca]=np.multiply(coeffs[ca],[0.0])

        # scale cutoff_high to end
        for ca in range(cutoff_high, len(coeffs)):
            coeffs[ca]=np.multiply(coeffs[ca],[0.0])
        Y = pywt.waverec(coeffs, dwt_transform) # inverse wavelet transform
        
        return Y      

    def zero_mean(self, X):
        return X-X.mean()


    
    def Arithmia_dataset(self, downsampling= 1):
        
        """Just load the data with Arithmia against the healty ECG"""
        
        #
        root = "/Users/pinguino/Documents/ECG Categorization/Data"
        recods = open(root+"/Records")

        ###
        count = 0
        recodrs_lines = recods.readlines()
        
        ### black boxes to fill
        age = []
        sex = []
        wave_feat = []
        data = []

        ###sex dictionary
        sex_dict = {'F':0, 'M':1}

        ###Butter filter
        ecg_power_line_interference = signal.butter(N= 3, btype= 'lowpass', Wn= .2)
        ecg_baseline_wander = signal.butter(N= 3, btype= 'highpass', Wn= 0.002)


        ### COLLECT DATA
        for item in recodrs_lines:

            path = root+"/"+item.rstrip('\n')
            path_records = root+"/"+item.rstrip('\n')+"Records"

            ###name mat_files
            name_file = pd.read_csv(path_records, header=None).values

            ###
            for name_path in name_file:
                File_Name = path+name_path[0]

                ##get the rawdata
                data_ = scipy.io.loadmat(File_Name)['val']

                ####make the signal zero-mean
                #data_ = np.apply_along_axis(self.zero_mean, axis=-1, arr= data_)
    
                ####
                #data_ = np.apply_along_axis(self.denoise_signal, 
                #                            axis= 1, 
                #                            arr= data_, 
                #                            dwt_transform= 'db8', 
                #                            dlevels= 13, 
                #                            cutoff_low= 1, 
                #                            cutoff_high= 10)
                
                ###butter filter -- Remove all frequency higher than 50Hz
                data_ = signal.filtfilt(ecg_power_line_interference[0], 
                                        ecg_power_line_interference[1], 
                                        data_, axis= 1)
                
                ####
                data_ = signal.filtfilt(ecg_baseline_wander[0], 
                                        ecg_baseline_wander[1], 
                                        data_, axis= 1)
                
                
                ###Savitzy-Golay Remove off-set
                #wandering_trend = np.apply_along_axis(signal.savgol_filter, axis= -1, arr= data_, window_length=4001, polyorder=7)
                #data_ = data_-wandering_trend
                

                #### ATTENTION! We are decimating the Time-Series
                data_ = signal.decimate(x= data_, q= downsampling, axis= 1) 
                ###use 5-second insted of 10-second TS (sampling Frequency 50Hz)
                data.append(data_.round(0).T)

                ###
                File_Read = open(File_Name+'.hea')
                File_Read_List = File_Read.readlines()

                count += 1

                #if count % 1000 == 0:
                #    print(count)

                ###collect age
                try:
                    age.append(float(File_Read_List[13][5:8]))
                except:
                    age.append(np.nan)

                ###collect sex
                try:
                    sex.append(sex_dict[File_Read_List[14][6:7]])
                except:
                    sex.append(np.nan)

                ###get all the "wave-features"
                try:
                    wave_features = File_Read_List[15][5:].split(',')
                    wave_features = [int(item.rstrip('\n')) for item in wave_features]
                    wave_feat.append(wave_features)
                except: 
                    wave_features = ['000']
                    wave_feat.append(wave_features)


        ### Get Arithmia Labels
        LABLES = []
        WAVE_FEATURES = []
        buck = 0
        for item in wave_feat:
            
            WAVE_FEATURES.append(item)
            
            try:
                ###atriaal fibbrillation
                binary_var = np.any(np.array(item)==164889003)
                
                ###
                control_var_binary = np.any(np.array(item)==426783006)
                #test_element = np.unique(np.array([270492004, 195042002, 54016002, 28189009, 27885002,
                #                         39732003, 47665007, 233917008, 428417006, 
                #                         164909002, 164909002, 164909002, 164873001, 
                #                         446358003, 89792004, 11157007, 75532003, 13640000, 
                #                         17338001, 195060002, 251180001, 426177001, 
                #                         164889003, 427084000, 164890007, 427393009, 426761007, 
                #                         713422000, 233896004, 233897008, 195101003]))
                #binary_var = np.isin(item , test_element)
                                
                if binary_var:
                    LABLES.append(1)
                elif control_var_binary :
                    LABLES.append(0)
                else:
                    LABLES.append(-1)
                    
            except ValueError as error:
                buck +=1
                print(error)
                LABLES.append(-1)

        #####
        print(len(LABLES), len(data))
        
        # Simple Imputer with Age and SeX
        #### #### ####
        age = SimpleImputer(strategy='most_frequent').fit_transform(np.array(age).reshape(-1, 1)).ravel()
        sex = SimpleImputer(strategy='most_frequent').fit_transform(np.array(sex).reshape(-1, 1)).ravel()

        ###select only those with age > 18    
        return {"data": np.array(data)[age>=18], 
                     "labels": np.array(LABLES)[age>=18], 
                     "age": age[age>=18], 
                     "sex": sex[age>=18], 
                      "name_features": np.array(["Lead-I", "Lead-II", "Lead-III", "Lead-aVR", "Lead-aVL", "Lead-aVF", 
                                        "Lead-V1", "Lead-V2", "Lead-V3", "Lead-V4", "Lead-V5", "Lead-V6"]), 
                      "wave_features": WAVE_FEATURES}
    
    
    
    
    def Myocardial_infarction_dataset(self, downsampling= 1):
        
        """Just load the data with Arithmia against the healty ECG"""
        
        #
        root = "/Users/pinguino/Documents/ECG Categorization/Data"
        recods = open(root+"/Records")

        ###
        count = 0
        recodrs_lines = recods.readlines()
        
        ### black boxes to fill
        age = []
        sex = []
        wave_feat = []
        data = []

        ###sex dictionary
        sex_dict = {'F':0, 'M':1}

        ###Butter filter
        ecg_power_line_interference = signal.butter(N= 3, btype= 'lowpass', Wn= .2)
        ecg_baseline_wander = signal.butter(N= 3, btype= 'highpass', Wn= 0.002)


        ### COLLECT DATA
        for item in recodrs_lines:

            path = root+"/"+item.rstrip('\n')
            path_records = root+"/"+item.rstrip('\n')+"Records"

            ###name mat_files
            name_file = pd.read_csv(path_records, header=None).values

            ###
            for name_path in name_file:
                File_Name = path+name_path[0]

                ##get the rawdata
                data_ = scipy.io.loadmat(File_Name)['val']

                ####make the signal zero-mean
                #data_ = np.apply_along_axis(self.zero_mean, axis=-1, arr= data_)
    
                ####
                #data_ = np.apply_along_axis(self.denoise_signal, 
                #                            axis= 1, 
                #                            arr= data_, 
                #                            dwt_transform= 'db8', 
                #                            dlevels= 13, 
                #                            cutoff_low= 1, 
                #                            cutoff_high= 10)
                
                ###butter filter -- Remove all frequency higher than 50Hz
                data_ = signal.filtfilt(ecg_power_line_interference[0], 
                                        ecg_power_line_interference[1], 
                                        data_, axis= 1)
                
                ####
                data_ = signal.filtfilt(ecg_baseline_wander[0], 
                                        ecg_baseline_wander[1], 
                                        data_, axis= 1)
        

                #### ATTENTION! We are decimating the Time-Series
                data_ = signal.decimate(x= data_, q= downsampling, axis= 1) 
                ###use 5-second insted of 10-second TS (sampling Frequency 50Hz)
                data.append(data_.round(0).T)

                ###
                File_Read = open(File_Name+'.hea')
                File_Read_List = File_Read.readlines()

                count += 1

                #if count % 1000 == 0:
                #    print(count)

                ###collect age
                try:
                    age.append(float(File_Read_List[13][5:8]))
                except:
                    age.append(np.nan)

                ###collect sex
                try:
                    sex.append(sex_dict[File_Read_List[14][6:7]])
                except:
                    sex.append(np.nan)

                ###get all the "wave-features"
                try:
                    wave_features = File_Read_List[15][5:].split(',')
                    wave_features = [int(item.rstrip('\n')) for item in wave_features]
                    wave_feat.append(wave_features)
                except: 
                    wave_features = ['000']
                    wave_feat.append(wave_features)


        ### Get Arithmia Labels
        LABLES = []
        WAVE_FEATURES = []
        buck = 0
        for item in wave_feat:
            
            WAVE_FEATURES.append(item)
            
            try:
                ###atriaal fibbrillation
                binary_var = np.any(np.array(item)==164865005)
                
                ###
                control_var_binary = np.any(np.array(item)==426783006)
                #test_element = np.unique(np.array([270492004, 195042002, 54016002, 28189009, 27885002,
                #                         39732003, 47665007, 233917008, 428417006, 
                #                         164909002, 164909002, 164909002, 164873001, 
                #                         446358003, 89792004, 11157007, 75532003, 13640000, 
                #                         17338001, 195060002, 251180001, 426177001, 
                #                         164889003, 427084000, 164890007, 427393009, 426761007, 
                #                         713422000, 233896004, 233897008, 195101003]))
                #binary_var = np.isin(item , test_element)
                                
                if binary_var:
                    LABLES.append(1)
                elif control_var_binary :
                    LABLES.append(0)
                else:
                    LABLES.append(-1)
                    
            except ValueError as error:
                buck +=1
                print(error)
                LABLES.append(-1)

        #####
        print(len(LABLES), len(data))
        
        # Simple Imputer with Age and SeX
        #### #### ####
        age = SimpleImputer(strategy='most_frequent').fit_transform(np.array(age).reshape(-1, 1)).ravel()
        sex = SimpleImputer(strategy='most_frequent').fit_transform(np.array(sex).reshape(-1, 1)).ravel()

        ###select only those with age > 18    
        return {"data": np.array(data)[age>=18], 
                     "labels": np.array(LABLES)[age>=18], 
                     "age": age[age>=18], 
                     "sex": sex[age>=18], 
                      "name_features": np.array(["Lead-I", "Lead-II", "Lead-III", "Lead-aVR", "Lead-aVL", "Lead-aVF", 
                                        "Lead-V1", "Lead-V2", "Lead-V3", "Lead-V4", "Lead-V5", "Lead-V6"]), 
                      "wave_features": WAVE_FEATURES}    
    
    def Sinus_Bradycardia_dataset(self, downsampling= 1):
        
        """Just load the data with Arithmia against the healty ECG"""
        
        #
        root = "/Users/pinguino/Documents/ECG Categorization/Data"
        recods = open(root+"/Records")

        ###
        count = 0
        recodrs_lines = recods.readlines()
        
        ### black boxes to fill
        age = []
        sex = []
        wave_feat = []
        data = []

        ###sex dictionary
        sex_dict = {'F':0, 'M':1}

        ###Butter filter
        ecg_power_line_interference = signal.butter(N= 3, btype= 'lowpass', Wn= .2)
        ecg_baseline_wander = signal.butter(N= 3, btype= 'highpass', Wn= 0.002)


        ### COLLECT DATA
        for item in recodrs_lines:

            path = root+"/"+item.rstrip('\n')
            path_records = root+"/"+item.rstrip('\n')+"Records"

            ###name mat_files
            name_file = pd.read_csv(path_records, header=None).values

            ###
            for name_path in name_file:
                File_Name = path+name_path[0]

                ##get the rawdata
                data_ = scipy.io.loadmat(File_Name)['val']

                ####make the signal zero-mean
                #data_ = np.apply_along_axis(self.zero_mean, axis=-1, arr= data_)
    
                ####
                #data_ = np.apply_along_axis(self.denoise_signal, 
                #                            axis= 1, 
                #                            arr= data_, 
                #                            dwt_transform= 'db8', 
                #                            dlevels= 13, 
                #                            cutoff_low= 1, 
                #                            cutoff_high= 10)
                
                ###butter filter -- Remove all frequency higher than 50Hz
                data_ = signal.filtfilt(ecg_power_line_interference[0], 
                                        ecg_power_line_interference[1], 
                                        data_, axis= 1)
                
                ####
                data_ = signal.filtfilt(ecg_baseline_wander[0], 
                                        ecg_baseline_wander[1], 
                                        data_, axis= 1)
        

                #### ATTENTION! We are decimating the Time-Series
                data_ = signal.decimate(x= data_, q= downsampling, axis= 1) 
                ###use 5-second insted of 10-second TS (sampling Frequency 50Hz)
                data.append(data_.round(0).T)

                ###
                File_Read = open(File_Name+'.hea')
                File_Read_List = File_Read.readlines()

                count += 1

                #if count % 1000 == 0:
                #    print(count)

                ###collect age
                try:
                    age.append(float(File_Read_List[13][5:8]))
                except:
                    age.append(np.nan)

                ###collect sex
                try:
                    sex.append(sex_dict[File_Read_List[14][6:7]])
                except:
                    sex.append(np.nan)

                ###get all the "wave-features"
                try:
                    wave_features = File_Read_List[15][5:].split(',')
                    wave_features = [int(item.rstrip('\n')) for item in wave_features]
                    wave_feat.append(wave_features)
                except: 
                    wave_features = ['000']
                    wave_feat.append(wave_features)


        ### Get Arithmia Labels
        LABLES = []
        WAVE_FEATURES = []
        buck = 0
        for item in wave_feat:
            
            WAVE_FEATURES.append(item)
            
            try:
                ###Sinus Brachicardia
                binary_var = np.any(np.array(item)==426177001)
                
                ###
                control_var_binary = np.any(np.array(item)==426783006)
                #test_element = np.unique(np.array([270492004, 195042002, 54016002, 28189009, 27885002,
                #                         39732003, 47665007, 233917008, 428417006, 
                #                         164909002, 164909002, 164909002, 164873001, 
                #                         446358003, 89792004, 11157007, 75532003, 13640000, 
                #                         17338001, 195060002, 251180001, 426177001, 
                #                         164889003, 427084000, 164890007, 427393009, 426761007, 
                #                         713422000, 233896004, 233897008, 195101003]))
                #binary_var = np.isin(item , test_element)
                                
                if binary_var:
                    LABLES.append(1)
                elif control_var_binary :
                    LABLES.append(0)
                else:
                    LABLES.append(-1)
                    
            except ValueError as error:
                buck +=1
                print(error)
                LABLES.append(-1)

        #####
        print(len(LABLES), len(data))
        
        # Simple Imputer with Age and SeX
        #### #### ####
        age = SimpleImputer(strategy='most_frequent').fit_transform(np.array(age).reshape(-1, 1)).ravel()
        sex = SimpleImputer(strategy='most_frequent').fit_transform(np.array(sex).reshape(-1, 1)).ravel()

        ###select only those with age > 18    
        return {"data": np.array(data)[age>=18], 
                     "labels": np.array(LABLES)[age>=18], 
                     "age": age[age>=18], 
                     "sex": sex[age>=18], 
                      "name_features": np.array(["Lead-I", "Lead-II", "Lead-III", "Lead-aVR", "Lead-aVL", "Lead-aVF", 
                                        "Lead-V1", "Lead-V2", "Lead-V3", "Lead-V4", "Lead-V5", "Lead-V6"]), 
                      "wave_features": WAVE_FEATURES}
    
    
    
    
    def Multiclass_dataset(self, downsampling= 1):
        
        """Just load the Multiclass ECG dataset"""
        
        #
        root = "/Users/pinguino/Documents/ECG Categorization/Data"
        recods = open(root+"/Records")

        ###
        count = 0
        recodrs_lines = recods.readlines()
        
        ### black boxes to fill
        age = []
        sex = []
        wave_feat = []
        data = []

        ###sex dictionary
        sex_dict = {'F':0, 'M':1}

        ###Butter filter
        ecg_power_line_interference = signal.butter(N= 3, btype= 'lowpass', Wn= .2)
        ecg_baseline_wander = signal.butter(N= 3, btype= 'highpass', Wn= 0.002)


        ### COLLECT DATA
        for item in recodrs_lines:

            path = root+"/"+item.rstrip('\n')
            path_records = root+"/"+item.rstrip('\n')+"Records"

            ###name mat_files
            name_file = pd.read_csv(path_records, header=None).values

            ###
            for name_path in name_file:
                File_Name = path+name_path[0]

                ##get the rawdata
                data_ = scipy.io.loadmat(File_Name)['val']

                ####make the signal zero-mean
                #data_ = np.apply_along_axis(self.zero_mean, axis=-1, arr= data_)
    
                ####
                #data_ = np.apply_along_axis(self.denoise_signal, 
                #                            axis= 1, 
                #                            arr= data_, 
                #                            dwt_transform= 'db8', 
                #                            dlevels= 13, 
                #                            cutoff_low= 1, 
                #                            cutoff_high= 10)
                
                ###butter filter -- Remove all frequency higher than 50Hz
                data_ = signal.filtfilt(ecg_power_line_interference[0], 
                                        ecg_power_line_interference[1], 
                                        data_, axis= 1)
                
                ####
                data_ = signal.filtfilt(ecg_baseline_wander[0], 
                                        ecg_baseline_wander[1], 
                                        data_, axis= 1)
        

                #### ATTENTION! We are decimating the Time-Series
                data_ = signal.decimate(x= data_, q= downsampling, axis= 1) 
                ###use 5-second insted of 10-second TS (sampling Frequency 50Hz)
                data.append(data_.round(0).T)

                ###
                File_Read = open(File_Name+'.hea')
                File_Read_List = File_Read.readlines()

                count += 1

                #if count % 1000 == 0:
                #    print(count)

                ###collect age
                try:
                    age.append(float(File_Read_List[13][5:8]))
                except:
                    age.append(np.nan)

                ###collect sex
                try:
                    sex.append(sex_dict[File_Read_List[14][6:7]])
                except:
                    sex.append(np.nan)

                ###get all the "wave-features"
                try:
                    wave_features = File_Read_List[15][5:].split(',')
                    wave_features = [int(item.rstrip('\n')) for item in wave_features]
                    wave_feat.append(wave_features)
                except: 
                    wave_features = ['000']
                    wave_feat.append(wave_features)


        ### Get Arithmia Labels
        LABLES = []
        WAVE_FEATURES = []
        buck = 0
        for item in wave_feat:
            
            WAVE_FEATURES.append(item)
            
            intra_lables = []
            
            try:
                ###atriaal fibbrillation
                binary_var_af = np.any(np.array(item)==164889003)
                binary_var_mi = np.any(np.array(item)==164865005)
                binary_var_sb = np.any(np.array(item)==426177001)
                                
                if binary_var_af:
                    intra_lables.append(1)
                elif binary_var_mi:
                    intra_lables.append(2)
                elif binary_var_sb:
                    intra_lables.append(3)
                else:
                    intra_lables.append(0)
                    
                if len(intra_lables) > 1:
                    print(intra_lables)
                    
                LABELS.append(intra_lables)
                    
            except ValueError as error:
                buck +=1
                print(error)
                LABLES.append(-1)

        #####
        print(len(LABLES), len(data))
        
        # Simple Imputer with Age and SeX
        #### #### ####
        age = SimpleImputer(strategy='most_frequent').fit_transform(np.array(age).reshape(-1, 1)).ravel()
        sex = SimpleImputer(strategy='most_frequent').fit_transform(np.array(sex).reshape(-1, 1)).ravel()

        ###select only those with age > 18    
        return {"data": np.array(data)[age>=18], 
                     "labels": np.array(LABLES)[age>=18], 
                     "age": age[age>=18], 
                     "sex": sex[age>=18], 
                      "name_features": np.array(["Lead-I", "Lead-II", "Lead-III", "Lead-aVR", "Lead-aVL", "Lead-aVF", 
                                        "Lead-V1", "Lead-V2", "Lead-V3", "Lead-V4", "Lead-V5", "Lead-V6"]), 
                      "wave_features": WAVE_FEATURES}    
    