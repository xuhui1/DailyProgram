# coding=utf-8

data = {
	"P0": {
		"age": "22",
		"salary": "41838.0",
		"suburb": "St. Kilda",
		"language": "Chinese"
	},
	"P1": {
		"suburb": "Flemington",
		"language": "English",
		"age": "68",
		"salary": "23242.0"
	},
	"P2": {
		"age": "eighty two",
		"language": "English",
		"suburb": "Toorak",
		"salary": "60196.0"
	},
	"P3":{
		"age": "49",
		"language": "Chinese",
		"suburb": "St. Kilda",
		"salary": "-16945514.0"
	},
	"P4":{
		"age": "54",
		"language": "Italian",
		"suburb": "Neverland",
		"salary": "49775.0"
	}
}
MAX_SALARY = 1000000
VALID_SUBURBS = ['Flemington','St. Kilda','Toorak']

"""
	input: data type is dict
	output: dict with clean
	function: clean data about 0<= salary <=MAX_SALARY, suburb exist in VALID_SUBURBS,  age>0 and mush be digit
"""
def clean_data(data):

	for id, item in sorted(data.items()):
		for key, value in item.items():
			if key == "age" and (not value.isdigit() or int(value)<0): # age>0 and mush be digit
				data[id][key] = None
			elif key == "salary" and (float(value)<0 or float(value) >MAX_SALARY): # 0<= salary <=MAX_SALARY
				data[id][key] = None
			elif key == "suburb" and (value not in VALID_SUBURBS): # suburb exist in VALID_SUBURBS
				data[id][key] = None

	return data

"""
	input: data is dict after clean, lower_age is age with int type, upper_age is age with int type
	output: float type
	function: calculate the average salary in lower_age <= age <= upper_age
"""
def average_salary(data, lower_age, upper_age):
	if lower_age>upper_age:
		return 0.0
	salary_total = 0
	salary_num = 0
	for id, item in sorted(data.items()):
		for key, value in sorted(item.items()): # age is in front
			if key == "age" and (value == None or int(value) < lower_age or int(value) > upper_age):
				break  # filter age == None and not in brackets
			elif key == "salary" and value != None:
				salary_total += float(data[id][key])
				salary_num += 1
	return salary_total/salary_num if salary_num>0 else 0.0  # if there are not individuals in the age bracket than return 0.0

"""
	input: data is dict after clean, n_bins is bin with int type, max_salary is salary with int type
	output: list type
	function: calculates the distribution of salaries greater than or equal to 0 and less than or equal to max_salary
"""
def wealth_distribution(data, n_bins, max_salary):
	bin_list = equal_size_bin(max_salary,n_bins)
	wealth_bin_count = [0]*n_bins  # init the wealth with every bin
	for id, item in sorted(data.items()):
		for key, value in sorted(item.items()): # age is in front
			if key == "salary" and value != None:
				if float(value)<bin_list[0]:  # judge the first value: 0<= value <bin_list[0]
					wealth_bin_count[0] += 1
					break
				if float(value)>=bin_list[-2] and float(value)<=bin_list[-1]: # judge the last value: bin_list[-2] <= value <=bin_list[-1]
					wealth_bin_count[-1] += 1
					break
				for i in range(len(bin_list)-2): # judge the middle value: bin_list[i] <= value <bin_list[i+1]
					if float(value)>=bin_list[i] and float(value) < bin_list[i+1]:
						wealth_bin_count[i+1] += 1
						break
	return wealth_bin_count
"""
	input: m is total number, n is bin
	output: list type
	function: divide the number into a number of equal-sized ‘bins’
	example: m=100,n=6 return [16, 32, 49, 66, 83, 100]
"""
def equal_size_bin(m,n):
	quotient = int(m / n)
	remainder = m % n
	if remainder > 0:
		bin_list = [quotient] * (n - remainder) + [quotient + 1] * remainder
	else:
		bin_list = [quotient] * n

	for i in range(1, len(bin_list)):
		bin_list[i] = bin_list[i] + bin_list[i - 1]
	return bin_list

"""
	input: data is dict after clean, lower_age is age with int type, upper_age is age with int type
	output: dict type
	function: returns a dictionary of the number of individuals in the given age bracket for each suburb.
"""
def location_age_counts(data, lower_age, upper_age):
	location_count_dict = {}
	for id, item in sorted(data.items()):
		for key, value in sorted(item.items()):
			if key == "suburb" and value != None:
				location_count_dict[value] = 0  # achieve all value of suburb
	if lower_age > upper_age:  #
		return location_count_dict
	for id, item in sorted(data.items()):
		for key, value in sorted(item.items()): # age is in front
			if key == "age" and (value == None or int(value) < lower_age or int(value) > upper_age):
				break  # age == None and age not in bracket than break
			elif key == "suburb" and value != None:
				location_count_dict[value] += 1  # number add 1
	return location_count_dict
