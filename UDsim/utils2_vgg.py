import cv2, os
import numpy as np
import matplotlib.image as mpimg
import time


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 200, 300, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def keras_process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    #image = edge_detection(image)
    #image = rgb2yuv(image)
    #dst = np.zeros(shape=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS))
    #cv2.imshow("",image)
    #print("imagecaptured")
    cv2.waitKey(1)

    #norm_img=cv2.normalize(image,dst,0,255,cv2.NORM_L1)
    norm_img = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #normalize data
    #norm_img = cv2.resize((cv2.cvtColor(norm_img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return norm_img


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def edge_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * 2.5
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    low_threshold = 60
    high_threshold = 220
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    #cv2.imshow("",edges) 

    rho = 5  #pixels, distance in Hough Space grid
    theta = (np.pi/180)  #Angular resolution of grid in Hough Space
    threshold = 3   #intersections in a given grid cell to qualify as a line
    min_line_length = 4 #minimum length of line acceptable as output
    max_line_gap = 5        #max dist between segments  to allow to be connected into a single line
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    lines_1 = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on the blank
    if lines_1 is not None :
        for line in lines_1:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)


    #cv2.imshow("",line_image) 

    '''
    lines_2 = cv2.HoughLinesP(cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY), rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    line_image = np.copy(image)*0
    for line in lines_2:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    '''

    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    return combo


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]) #empty array (n,height,width,channels)
    steers = np.empty(batch_size) #empty array ()

    print("\nBATGEN Image_Paths type:",type(image_paths) )
    j=0
    while True: #inf steps per epoch,unless stated otherwies
        i = 0
        if j>=(image_paths.shape[0]-1):
            j=0
        for index in range(image_paths.shape[0]): #choose a random index
            if j>=(image_paths.shape[0]-1):
                j=0
            center, left, right = image_paths[j]
            steering_angle = steering_angles[j]
            print('steering_angle_raw:',steering_angle)
            #print("I:",j)
            '''
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
                print("IF")
            else:
                image = load_image(data_dir, center) 
                #cv2.imshow("",image)
                #cv2.waitKey(50)
                print("ELSE")
                
                #print(image)
            '''
            
            input_start = -1
            input_end =1
            output_start = 0
            output_end = 1
            norm_steer = output_start + ((output_end - output_start) / (input_end - input_start)) * (steering_angle - input_start)
            steering_angle = norm_steer
            '''
            if norm_steer>0.5:
                steering_angle = 1
            else:
                steering_angle = 0
            '''

            print('steering_angle_norm:',steering_angle)
            #image, steering_angle = augument(data_dir, center, left, right, steering_angle
            image = load_image(data_dir, center) 
            #image = edge_detection(image)
            # add the image and steering angle to the batch
            image = preprocess(image)
            #processed = keras_process_image(image)
            images[i] = image
            #print(images[i])
            steers[i] = steering_angle

           

            i += 1
            j += 1
            if i == batch_size:
                break
        print("batch_generator():")
        print(images.shape, type(images))
        #print(images[0])

        #cv2.imshow("",images[0])
        #cv2.waitKey(50)
        
        print("steer shape")
        print(steers.shape)
        #time.sleep(10)
        yield images, steers

