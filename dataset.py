# This dataset.py file is used to group images into categories
# The images are retrieved from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
import argparse # required for parser
import csv # required for csv reader
import os # required for files/dirs manipulation
import shutil # required for files/dirs manipulation

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
		file_name = row[0] # the relative path to the image file
		print(file_name)
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
		classDir = args['dir'] + "/" + class_name
		if not os.path.exists(classDir):
			os.mkdir(classDir)
		# move file to the above dir
		sourcePath = args['dir'] + "/" + file_name
		destPath = classDir + "/" + file_name
		if os.path.exists(sourcePath):
			os.rename(sourcePath, destPath)