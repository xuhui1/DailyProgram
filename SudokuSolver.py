import time
import sys

# read sudo file to list
def read_sudo_file(in_file):
	original_sudo = []
	with open(in_file,'r',encoding='utf-8') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if not line:
				continue
			one_sudo = []
			for element in line:
				if element == '.':
					element = 0
				else: element = int(element)
				one_sudo.append(int(element))
			original_sudo.append(one_sudo)
	return original_sudo
# translate every line to sudo matrix
def line_to_matrix(one_line_sudo):
	matrix = []
	for i in range(9):
		row = one_line_sudo[i*9:i*9+9]
		matrix.append(row)
	return matrix
# translate every matrix to line to save in file
def matrix_to_line(sudo_matrix):
	line = ''
	for i in range(9):
		row = ''.join(str(one) for one in sudo_matrix[i])
		line += row
	return line
# every nine block number -> [3*3*3] list
def nine(data):
	nine_block = []
	for i in range(3):
		for j in range(3):
			x = data[3*i:3*(i+1)]
			block = []
			for _x in x:
				y = _x[3*j:3*(j+1)]
				block.append(y)
			nine_block.append(block)
	return nine_block
# get number set not find in row and column
def num_set(data,nine_block):
	pick_set = {}
	for i in range(9):
		for j in range(9):
			if data[i][j] == 0:
				pick_set[str(i) + str(j)] = not_find(data,i,j,nine_block)
	return pick_set
# get number set not find in row and column
def not_find(data,i,j,nine_block):
	row = data[i]
	col = []
	for one in data[:]:
		col.append(one[j])
	_find = set(row)
	for one in col:
		_find.add(one)
	for one in nine_block[j//3+3*(i//3)]:
		for _one in one:
			_find.add(_one)
	r = []
	all_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	for num in all_num:
		if num not in _find:
			r.append(num)
	return r
class ViolentSorted:
	"""
		1) Ergodic each non-zero position in sequence.
		2) Try to fill in the numbers 1 to 9 in the grid.
		3) Recursive judgment of whether the number of fillings conforms to the rules.
		4) When all the numbers are filled out, exit the loop
	"""
	# choose one of the few not find number to start search
	def check(self, data,x, y, value):  # check every row and column have the same number
		for row_item in data[x]:
			if row_item == value:
				return False
		for row_all in data:
			if row_all[y] == value:
				return False
		row, col = int(x / 3) * 3, int(y / 3) * 3
		row3col3 = data[row][col:col + 3] + data[row + 1][col:col + 3] + data[row + 2][col:col + 3]
		for row3col3_item in row3col3:
			if row3col3_item == value:
				return False
		return True
	def get_next(self,data, x, y):  # get next not fill number
		for next_soduku in range(y + 1, 9):
			if data[x][next_soduku] == 0:
				return x, next_soduku
		for row_n in range(x + 1, 9):
			for col_n in range(0, 9):
				if data[row_n][col_n] == 0:
					return row_n, col_n
		return -1, -1  # if not next fill number return -1

	def try_it(self, data, x, y):
		if data[x][y] == 0:
			for i in range(1, 10):  # from 1 to 9
				if self.check(data,x, y, i):
					data[x][y] = i  # Fill the qualified blanks in 0
					next_x, next_y = self.get_next(data,x, y)  # get next blanks
					if next_x == -1:
						return True
					else:
						end = self.try_it(data,next_x, next_y)
						if not end:
							data[x][y] = 0
						else:
							return True
	def search(self,data):
		if data[0][0] == 0:
			self.try_it(data,0, 0)
		else:
			x, y = self.get_next(data,0, 0)
			self.try_it(data,x, y)
		self.data = data
class CandidateSetSorted:
	"""
		1. Find out the number that can be filled in at the position of 0, and store the position and the
		number in the dictionary in the form of key and value.
		2. Sort the values in the dictionary in ascending order, select the first one.
		3. Record the filling process in the list.
		4. Update 1 and 2 steps, if there is a blank data to fill in, indicating that there is a problem with the choice
		of a previous step, go back, change the value, and then go back to step 1.
		5. When all the numbers are filled out, exit the loop
	"""
	# choose one of the few not find number to start search
	def search(self,data):
		insert_step = []
		while True:
			nine_block = nine(data)
			pick_set = num_set(data,nine_block)
			if len(pick_set) == 0: break
			pick_sort = sorted(pick_set.items(), key=lambda x: len(x[1]))
			item_min = pick_sort[0]
			key = item_min[0]
			value = list(item_min[1])
			insert_step.append((key, value))
			if len(value) != 0:
				data[int(key[0])][int(key[1])] = value[0]
			else:
				insert_step.pop()
				for i in range(len(insert_step)):
					huishuo = insert_step.pop()
					key = huishuo[0]
					insert_num = huishuo[1]
					if len(insert_num) == 1:
						data[int(key[0])][int(key[1])] = 0
					else:
						data[int(key[0])][int(key[1])] = insert_num[1]
						insert_step.append((key, insert_num[1:]))
						break
		self.data = data

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("usage: ./SudokuSolver.py <filename>")
		sys.exit(1)
	time1 = time.time()
	in_file = sys.argv[1]
	candidata_set_sorted = CandidateSetSorted()
	original_sudo = read_sudo_file(in_file)
	out_file = ''
	if in_file.startswith('euler'):
		out_file = 'Euler.out'
	elif in_file.startswith('magictour'):
		out_file = 'Magic Tour.out'
	else:
		print('The input name is invalid!')
		exit(1)
	with open(out_file,'w',encoding='utf-8') as o:
		for one_sudo in original_sudo:
			matrix = line_to_matrix(one_sudo)
			candidata_set_sorted.search(matrix)
			line = matrix_to_line(candidata_set_sorted.data)
			o.write(line+"\n")
	time2 = time.time()
	print('\nFinished! using time:', time2 - time1, 's')
