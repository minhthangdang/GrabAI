# This dataset.py file is used to group images into categories
import argparse # required for parser
import csv # required for csv reader

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
		classnames = row
		break

# read the annotations from csv file and group images into its corresponding class
with open(args['classnames']) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		classnames = row
		break