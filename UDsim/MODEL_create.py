import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2 
from tensorflow.keras.applications import ResNet50
import pathlib

#Paths to Training Data directories
path = 'H:/CARLA_0.9.10-Pre_Win/TrainingData'
standard_path = 'H:/CARLA_0.9.10-Pre_Win/TrainingData_Simulator2'
recovery_path = 'H:/CARLA_0.9.10-Pre_Win/TrainingData/Recovery'
udacity_path = "H:/CARLA_0.9.10-Pre_Win/TrainingData_Simulator"
track2_path = 'H:/CARLA_0.9.10-Pre_Win/TrainingData/Track2'
img_path = path + '/IMG'
models_path = "./models" 

def load_csv(file_path, col_names, remove_header=False):
    csv = pd.read_csv(file_path, header=None, names=col_names)
    if remove_header:
        csv = csv[1:]
    
    return csv 

pd.options.display.max_colwidth=100

# Define our column headers
col_header_names = ["Center", "Left", "Right", "Steering Angle", "Throttle", "Brake","Speed"] 

# Let's load our standard driving dataset, where we drive in both directions
standard_csv = load_csv(standard_path + "/driving_log.csv", col_header_names)
standard_csv["Steering Angle"] = standard_csv["Steering Angle"].astype(float) 
print("Standard dataset has {0} rows".format(len(standard_csv))) 

# Let's load the data set of driving in recovery mode (aka virtual drink driving dataset)
recovery_csv = load_csv(recovery_path + "/driving_log.csv", col_header_names)
recovery_csv["Steering Angle"] = recovery_csv["Steering Angle"].astype(float) 
print("Recovery dataset has {0} rows".format(len(recovery_csv)))

# Let's load dataset from track 2
track2_csv = load_csv(track2_path + "/driving_log.csv", col_header_names)
track2_csv["Steering Angle"] = track2_csv["Steering Angle"].astype(float) 
print("Track 2 dataset has {0} rows".format(len(track2_csv)))

# Finally let's load the udacity dataset
udacity_csv = load_csv(udacity_path + "/driving_log.csv", col_header_names, remove_header=True)
udacity_csv["Steering Angle"] = udacity_csv["Steering Angle"].astype(float) 
print("Standard dataset has {0} rows".format(len(udacity_csv)))

def get_steering_angles(data, st_column, st_calibrations, filtering_f=None):
    """
    Returns the steering angles for images referenced by the dataframe
    The caller must pass the name of the colum containing the steering angle 
    along with the appropriate steering angle corrections to apply
    """
    cols = len(st_calibrations)
    print("CALIBRATIONS={0}, ROWS={1}".format(cols, data.shape[0]))
    angles = np.zeros(data.shape[0] * cols, dtype=np.float32)
    
    i = 0
    for indx, row in data.iterrows():        
        st_angle = row[st_column]
        for (j,st_calib) in enumerate(st_calibrations):  
            angles[i * cols + j] = st_angle + st_calib
        i += 1
    
    # Let's not forget to ALWAYS clip our angles within the [-1,1] range
    return np.clip(angles, -1, 1)

st_angle_names = ["Center", "Left", "Right"]
st_angle_calibrations = [0, 0.25, -0.25]

standard_st_angles_without_calibrations = get_steering_angles(standard_csv, "Steering Angle", [0])
recovery_st_angles_without_calibrations = get_steering_angles(recovery_csv, "Steering Angle", [0])
track2_st_angles_without_calibrations = get_steering_angles(track2_csv, "Steering Angle", [0])
udacity_st_angles_without_calibrations = get_steering_angles(udacity_csv, "Steering Angle", [0])

frames = [recovery_csv, udacity_csv, track2_csv]
ensemble_csv = pd.concat(frames)
len(ensemble_csv)

validation_csv = standard_csv
len(validation_csv)

ensemble_st_angles_without_calibrations = get_steering_angles(ensemble_csv, "Steering Angle", [0])

def read_img(img_full_path, img_dir="/IMG"):
    prefix_path = udacity_path + img_dir

    if "Dataset_2" in img_full_path:
        prefix_path = standard_path + img_dir
    elif "Recovery_Driving" in img_full_path:
        prefix_path = recovery_path + img_dir
    elif "Track_2" in img_full_path:
        prefix_path = track2_path + img_dir
    
    img_path = "{0}/{1}".format(prefix_path, img_full_path.split("\"")[-1]) 
    #img_path = prefix_path+ img_full_path.split("/")[-1]
    #img = cv2.imread(img_path)
    img = cv2.imread(img_full_path)

    #print("prefix_path::",prefix_path)
    #print("img_full_path::",img_full_path.split("/")[-1]) 
    #print("img_path::",img_path)
    
    # OpenCV reads images in BGR format, we are simply converting and returning the image in RGB format
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #return img

def fliph_image(img):
    """
    Returns a horizontally flipped image
    """
    return cv2.flip(img, 1)

def blur_image(img, f_size=5):
    """
    Applies Gaussir Blur to smoothen the image.
    This in effect performs anti-aliasing on the provided image
    """
    img = cv2.GaussianBlur(img,(f_size, f_size),0)
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8) 

# Read more about it here: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
    """
    Shifts the image right, left, up or down. 
    When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
    """
    rows, cols = (img.shape[0], img.shape[1])
    translation_x = np.random.randint(low_x_range, high_x_range) 
    translation_y = np.random.randint(low_y_range, high_y_range) 
    
    st_angle += translation_x * delta_st_angle_per_px

    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
    return img, st_angle 

def change_image_lightness(img, low, high):
    """
    Applies an offset in [low, high] interval to change the 'L' component of the supplied image in HSL format
    The returned image in converted back to RGB
    """
    # Convert to HSL (HLS in OpenCV!!)
    hls = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HLS)
    hls = hls.astype(int)    
    
    # Add an offset to light component 
    offset = np.random.randint(low, high=high)
    # Since the format is HLS and NOT HSL, it is the second component (index 1) that is modified
    #hls[:,:,1] += offset
    hls[:,:,1] = offset

    # Make sure our lightness component is in the interval [0, 255]
    np.clip(hls, 0, 255)
    
    # Convert back to uint
    hls = hls.astype(np.uint8)
    
    # Make sure we return image in RGB format
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) 

def change_image_brightness(img, low, high):
    """
    Applies an offset in [low, high] interval to change the 'V' component of the supplied image in HSV format
    The returned image in converted back to RGB
    """

    # Convert to HSV
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(int)    
    
    # Adding the offset to the v component
    offset = np.random.randint(low, high=high)
    hsv[:,:,2] += offset
    
    # Make sure our lightness component is in the interval [0, 255]
    np.clip(hsv, 0, 255)
    
    # Convert back to uint
    hsv = hsv.astype(np.uint8)
    
    # Make sure we return image in RGB format
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 

def add_random_shadow(img, w_low=0.6, w_high=0.85):
    """
    Overlays supplied image with a random shadow poligon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)

def shift_horizon(img, h_s=0.2):
    img = img.astype(np.float32)
    
    # randomly shift horizon
    height = img.shape[0]
    width = img.shape[1]
    horizon = h_s * height / 3
    v_shift = np.random.randint(-height/8,height/8)
    pts1 = np.float32([[0, horizon],[width, horizon], [0, height], [width, height]])
    pts2 = np.float32([[0, horizon + v_shift],[width, horizon + v_shift], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M,(width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return img.astype(np.uint8)

def change_gamma(img, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    img = img.astype(np.float32) 
    img /= 255.0
    
    #img[:,:100,:] = img[:,:100,:] ** invGamma
    #img *= 255
    #img = img.astype(np.uint8)
    img = cv2.bitwise_and(img * 255, img * 255, mask=img[:,:100,:] ** invGamma)
    return img
    
    #table = np.array([((i / 255.0) ** invGamma) * 255
    #for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def change_image_brightness_rgb(img, s_low=0.2, s_high=0.75):
    """
    Changes the image brightness by multiplying all RGB values by the same scalacar in [s_low, s_high).
    Returns the brightness adjusted image in RGB format.
    """
    img = img.astype(np.float32)
    s = np.random.uniform(s_low, s_high)
    img[:,:,:] *= s
    np.clip(img, 0, 255)
    return  img.astype(np.uint8)

def augment_image(img, st_angle, p=1.0):
    """
    Augment a given image, by applying a series of transformations, with a probability p.
    The steering angle may also be modified.
    Returns the tuple (augmented_image, new_steering_angle)
    """
    aug_img = img
    
    #if np.random.random_sample() <= 1.0:
        # Reduce aliasing via blurring
        #aug_img = blur_image(aug_img)
   
    if np.random.random_sample() <= 0.5: 
        # Horizontally flip image
        aug_img = fliph_image(aug_img)
        st_angle = -st_angle
     
    if np.random.random_sample() <= 0.5:
        # Change image brightness or lightness
        #low = -np.random.randint(25, high=100)
        #high = np.random.randint(25, high=100)
        #if np.random.random_sample() <= 0.5:
            # Try changes in lightness
            #aug_img = change_image_lightness(aug_img, low, high)
        #else:
            # Try changes in brightness
            #aug_img = change_image_brightness(aug_img, low, high)
        # TODO Maybe use change_brightness_rgb function instead??
        aug_img = change_image_brightness_rgb(aug_img)
    
    if np.random.random_sample() <= 0.5: 
        aug_img = add_random_shadow(aug_img, w_low=0.45)
    '''     
    if np.random.random_sample() <= 0.5:
        # Shift the image left/right, up/down and modify the steering angle accordingly
        aug_img, st_angle = translate_image(aug_img, st_angle, -60, 61, -20, 21, 0.35/100.0)
    '''
    # TODO Try adding slight rotations
        
    return aug_img, st_angle

def generate_images(df, target_dimensions, img_types, st_column, st_angle_calibrations, batch_size=100, shuffle=True, 
                    data_aug_pct=0.8, aug_likelihood=0.5, st_angle_threshold=0.05, neutral_drop_pct=0.25):
    """
    Generates images whose paths and steering angle are stored in supplied dataframe object df
    Returns the tuple (batch,steering_angles)
    """
    # e.g. 160x320x3 for target_dimensions
    batch = np.zeros((batch_size, target_dimensions[0],  target_dimensions[1],  target_dimensions[2]), dtype=np.float32)
    steering_angles = np.zeros(batch_size)
    df_len = len(df)
    
    while True:
        k = 0
        while k < batch_size:            
            idx = np.random.randint(0, df_len)       

            for img_t, st_calib in zip(img_types, st_angle_calibrations):
                if k >= batch_size:
                    break
                                 
                row = df.iloc[idx]
                st_angle = row[st_column]            
                
                # Drop neutral-ish steering angle images with some probability
                if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct :
                    continue
                    
                st_angle += st_calib                                                                
                img_type_path = row[img_t]  
                #print(img_type_path)
                img = read_img(img_type_path)                
                
                # Resize image
                    
                img, st_angle = augment_image(img, st_angle, p=aug_likelihood) if np.random.random_sample() <= data_aug_pct else (img, st_angle)
                batch[k] = img
                steering_angles[k] = st_angle
                k += 1
            
        yield batch, np.clip(steering_angles, -1, 1)            

def show_images(imgs, labels, cols=5, fig_size=(15, 5)):
    rows = len(imgs) // cols
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r,c]
            img = imgs[cols * r + c]
            lb = labels[cols * r + c]
            ax.imshow(img.astype(np.uint8))
            ax.axis('on')
            ax.set_aspect('equal')
            ax.set(title=lb)
    fig.tight_layout()
    plt.show()
    
b_divider = 2
# Multiplying by 3 since we have center, left and right images per row
b_size = len(ensemble_csv)  * 3 // b_divider
b_size

gen = generate_images(ensemble_csv, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=b_size)

# Don't augment validation data at all
#gen_validation = generate_images(validation_csv, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=b_size, data_aug_pct=0.0)
#print(ensemble_csv["Center"][1])

#b, s = next(gen)

b0 = b[7]
b0_l = change_image_brightness_rgb(b0, s_low=0.2, s_high=0.75)
plt.imshow(b0_l)

# Initial Setup for Keras
import tensorflow as tf
from tensorflow.keras.backend import resize_images
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# Let's also define the input shape
in_shape = (160, 320, 3)

def resize(img):
    return resize_images(img, [66, 200])

def nvidia_model():
    model = Sequential()
    # Cropping image
    model.add(Lambda(lambda imgs: imgs[:,80:,:,:], input_shape=(160, 320, 3)))
    # Normalise the image - center the mean at 0
    model.add(Lambda(lambda imgs: (imgs/255.0)))
    model.add(Lambda(resize))
    
    # TODO use Keras.Cropping instead of a lambda layer

    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))    
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    
    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu')) 
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu')) 
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    
    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(1164, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
    return model

def build_Resnet50_DeepLab():

    rs50 = ResNet50(include_top = False, pooling = 'avg', weights='imagenet',
        input_shape=(160, 320, 3))
    for layer in rs50.layers:
        print(layer, layer.trainable)

    for layer in rs50.layers[:]:
        layer.trainable = False

    #mbv2 = MobileNetV2(include_top=False)
    '''
    dlv3 =  Deeplabv3(input_shape=(512,512,3),backbone="mobilenetv2", classes=1) 
    for layer in dlv3.layers:
        print(layer, layer.trainable)
    '''

    model = Sequential()
    model.add(rs50)
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='linear'))

    return model

m = nvidia_model()
m.summary()

#gen_train = generate_images(ensemble_csv, in_shape, st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=2)

# Take 20% of the data
gen_val = generate_images(validation_csv, in_shape, st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=(b_size * b_divider) // 5, data_aug_pct=0.0)
x_val, y_val = next(gen_val)

m.fit(gen, validation_data=(x_val, y_val),batch_size=32,epochs=5,shuffle=True,verbose=1,steps_per_epoch=7000 )

m.save("vggX_drive.h5".format(models_path))

