import numpy as np
import os


def process_file(line_B, directory, filename):
    '''
    creates 2d array from the csv files fed into it
    '''
    i = 0
    with open(os.path.join(directory, filename), "r") as f:
        for line in f:
            i += 1
            line_A = line[13:]
            line_A = line_A.rstrip()
            line_A = line_A.split(";")
            line_A = [i.replace(",", ".") for i in line_A]
            try:
                line_A = [float(i) for i in line_A]
                line_A = np.array(line_A)
                line_A = np.reshape(line_A, (1, 3))
                line_B = np.append(line_B, line_A, axis=0)
            except:
                print("line has data error, line number : %d" %i)
                print(line_A)

    return line_B


def compress_data(line_B, frequency):

    '''
    compresses measurements to 50hz
    '''

    if frequency == 200:
        comp_parameter = 4
    elif frequency == 500:
        comp_parameter = 10

    data_rows = np.shape(line_B)[0]
    line_C = np.empty((0, 3), float)
    for i in range(0, data_rows, comp_parameter):
        mean = np.mean(line_B[i:(i + comp_parameter), :], axis=0)
        mean = np.reshape(mean, (1, 3))
        line_C = np.append(line_C, mean, axis=0)

    return line_C


def create_data(line_C):

    '''
    transforms data to be fed to the convolutional network
    '''

    data = np.empty((convolution_size, 3, 0), float)
    num_rows = np.shape(line_C)[0]
    for i in range(convolution_size, (num_rows-1), stride):
        data = np.append(data, np.atleast_3d(line_C[(i-convolution_size):i, :]), axis=2)
    lenght = np.shape(data)[2]

    return data, lenght


def determine_data_type(filename):
    '''
    determines the processed files type recorder devices frequency and assigns a step number
    '''
    if filename.startswith('N'):    #Negative data
        file = 0
        step_num = 0
        filename_split = filename.split("_")
        frequency = filename_split[1]
        frequency = int(frequency)
    elif filename.startswith('L') or filename.startswith("T"):      # Normal walking data
        file = 1
        step_num = 20
        filename_split = filename.split("_")
        frequency = filename_split[1]
        frequency = int(frequency)
    elif filename.startswith('S'):      #Stair data
        file = 2
        step_num = 10
        filename_split = filename.split("_")
        frequency = filename_split[1]
        frequency = int(frequency)

    return file, step_num, frequency


def create_step_file(lenght, step_num):
    '''
    creates the slicing steps for cost calculation during training
    '''
    extra = np.remainder(lenght, step_num)
    step_size = np.floor_divide(lenght, step_num)
    u = 0
    step_data = np.empty((0, 1), int)
    for i in range(step_num):
        if u < extra:
            step_data = np.append(step_data, (np.atleast_2d(step_size + 1)), axis=0)
            u = u + 1
        else:
            step_data = np.append(step_data, np.atleast_2d(step_size), axis=0)

    return step_data


def step_coordinatoes (full_step_data):

    '''
    create coordinates for slicing, first row is the step sizes, the second is for the beginner coordinates
    '''

    lenght, _ = np.shape((full_step_data))
    coordinates = np.empty(((lenght), 2), int)

    for i in range((lenght)):
        if i < 1:
            coordinates[i, :] = full_step_data[i], 0
        else:
            coordinates[i, :] = full_step_data[i], np.add(coordinates[(i - 1), 0], coordinates[(i - 1), 1])
    return coordinates


def process_one_file(directory, filename):
    '''
    opens a single file then returns a 3D array for training plus the Y values for that array

    '''
    file, step_num, frequency = determine_data_type(filename)
    line_B = np.empty((0, 3), float)

    line_B = process_file(line_B, directory, filename)

    line_C = compress_data(line_B, frequency)
    data, lenght = create_data(line_C)
    step_data = create_step_file(lenght, step_num)
    y_values = np.zeros((lenght, 4))    #first three rows are the classification values last is the intensity
    if file == 0:                       # 0 == negative   1 == positive lin or turn    2 == positive stairs
        y_values[:, 0] = np.ones((lenght))
        y_values[:, 3] = np.zeros((lenght))
    elif file == 1:
        y_values[:, 1] = np.ones((lenght))
        y_values[:, 3] = np.ones((lenght)) * ((step_num * step_value)/lenght)
    elif file == 2:
        y_values[:, 2] = np.ones((lenght))
        y_values[:, 3] = np.ones((lenght)) * ((step_num * step_value) / lenght)

    return data, y_values, step_data


def process_test_directory(nametype):
    directory = test_directory
    file_count = 0
    training_data = np.empty((convolution_size, 3, 0), float)
    y_data = np.empty((0, 4), float)
    full_step_data = np.empty((0, 1), int)

    for filename in os.listdir(directory):
        data, y_values, step_data = process_one_file(directory, filename)
        training_data = np.append(training_data, data, axis=2)
        y_data = np.vstack((y_data, y_values))
        full_step_data = np.vstack((full_step_data, step_data))
        file_count = file_count + 1

    coordinates = step_coordinatoes(full_step_data)
    print("test:")
    print(file_count, "file count")
    np.save("test_data" + nametype, training_data)
    np.save("test_ys" + nametype, y_data)
    np.save("test_step" + nametype, coordinates)
    print(np.shape(y_data), "y data shape")
    print(np.shape(training_data), "data shape")
    print(np.shape(coordinates), "coordinates shape")
    print(np.shape(full_step_data), "full_step_data")


def process_train_directory(nametype):
    directory = train_directory
    file_count = 0
    training_data = np.empty((convolution_size, 3, 0), float)
    y_data = np.empty((0, 4), float)
    full_step_data = np.empty((0, 1), int)

    for filename in os.listdir(directory):
        data, y_values, step_data = process_one_file(directory, filename)
        training_data = np.append(training_data, data, axis=2)
        y_data = np.vstack((y_data, y_values))
        full_step_data = np.vstack((full_step_data, step_data))
        file_count = file_count + 1

    coordinates = step_coordinatoes(full_step_data)
    print("train:")
    print(file_count, "file count")
    np.save("train_data" + nametype, training_data)
    np.save("train_ys" + nametype, y_data)
    np.save("train_step" + nametype, coordinates)
    print(np.shape(y_data), "y data shape")
    print(np.shape(training_data), "data shape")
    print(np.shape(coordinates), "coordinates shape")


#Output parameters
convolution_size = 16
stride = 5
step_value = 400
info = "multiclass_stairs"   # information for the filename

"""
insert directory paths here
"""
test_directory = r""
train_directory = r""

process_train_directory(info)

process_test_directory(info)

