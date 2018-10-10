# coding=utf-8

"""
	input: datafile is csv file
	output: None
	function: read file, clean the data, print the average salary for 18-34 year olds, and print list of the suburbs in which 18–34 year olds live
"""

def main(datafile):
	import program as pg
	data = pg.read_data(datafile) # read csv file
	data = pg.clean_data(data)  # clean data
	average_salary = pg.average_salary(data,10,34)  # calculate average salary in age bracket
	print("Average salary for 18-34 year olds: $"+str(round(average_salary,2))) # value with rounding
	location_count_dict = pg.location_age_counts(data,18,34) # achieve location count in age bracket
	print("18-34 year olds live in the following suburbs:")
	for suburb, count in sorted(location_count_dict.items()): # alphabetical order
		print(suburb + " (" + str(count) + ")")

main(data)
