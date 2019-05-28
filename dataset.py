# USAGE
# python train.py --dir car_ims --annos annotations.csv --classnames class_names.csv

# This dataset.py file is used to group images into categories
# The images are retrieved from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
import argparse # required for parser
import csv # required for csv reader
import os # required for files/dirs manipulation
import shutil # required for files/dirs manipulation
import cv2 # opencv

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--annos", required=True,
	help="path to annotations file")
ap.add_argument("-c", "--classnames", required=True,
	help="path to class names file")
args = vars(ap.parse_args())

# read the class names from csv file and saved it to an array
with open(args['classnames']) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		class_names = row
		break

makes = ['Acura', 'AM General', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti', 'Buick', \
		 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge', 'Eagle', 'Ferrari', 'FIAT',
		 'Fisker', 'Ford', 'Geo', 'GMC', 'Honda', 'HUMMER', 'Hyundai', 'Infiniti', 'Isuzu',
		 'Jaguar', 'Jeep', 'Lamborghini', 'Land Rover', 'Lincoln', 'Maybach', 'Mazda', 'McLaren',
		 'Mercedes-Benz', 'MINI Cooper', 'Mitsubishi', 'Nissan', 'Plymouth', 'Porsche', 'Ram CV',
		 'Rolls-Royce', 'Scion', 'smart fortwo', 'Spyker', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen',
		 'Volvo']

# read the annotations from csv file and group images into its corresponding class
with open(args['annos']) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		# the relative path to the image file
		file_name = row[0]
		print(file_name)

		# extract the car object based on provided bounding box
		image = cv2.imread(file_name)
		x1 = int(row[1])
		y1 = int(row[2])
		x2 = int(row[3])
		y2 = int(row[4])
		car_image = image[y1:y2, x1:x2, :]

		# get file name from format car_ims/xxx.jpg
		pos = file_name.rfind("/")
		file_name = file_name[pos+1:]

		# get class name
		class_index = int(row[5]) - 1
		class_name = class_names[class_index]

		# remove year from class name
		pos = class_name.rfind(" ")
		class_name = class_name[0:pos]

		# create dir for each class (if not already created)
		# first we split class name into make_model format
		for make in makes:
			if class_name.startswith(make):
				class_name = class_name.replace(make, make+"_")
		# create dir
		classDir = "dataset/" + class_name
		if not os.path.exists(classDir):
			os.mkdir(classDir)

		# save car_image file to the above dir
		cv2.imwrite(classDir + "/" + file_name, car_image)