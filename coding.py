#!/opt/bin/python3
import sys
def encode(names):
	if isinstance(names, list):
		for txt_name in names:
			start_encoding(txt_name)
	else:
		start_encoding(names)
# program verifies that the file determined by the user is of the right format
def is_text_file(file_name):
	name = file_name.split(".")
	if not name[1].startswith("txt"):
		print("Please input txt file.")
		sys.exit()
# Function used to write the magic numbers to the mtf file.
def _magic_numbers(mtf):
	xx = bytearray()
	xx.append(0xfa)
	xx.append(0xce)
	xx.append(0xfa)
	xx.append(0xdf)
	mtf.write(xx)
# write strs to mtf file
def _mtf_write(mtf, strs):
	"""Function to write strings or ints to mtf file """
	if isinstance(strs, int):  # if the thing to write is an int
		if (strs <= 120 and strs >= 0):  # low codes
			mtf.write(chr(strs + 128).encode("latin-1"))
		elif (strs >= 121 and strs <= 375):  # higher codes
			mtf.write(chr(249).encode("latin-1"))
			mtf.write(chr(strs - 121).encode("latin-1"))
		elif (strs >= 376 and strs <= 65912):  # highest codes
			mtf.write(chr(250).encode("latin-1"))
			mtf.write(chr((strs - 376) // 256).encode("latin-1"))
			mtf.write(chr((strs - 376) % 256).encode("latin-1"))
	else:
		mtf.write(bytes(strs.encode("latin-1")))  # if it is not an int
# if word is on the list and returns the index. also it moves the element to the front
def _move_word(word, list_of_words):
	index = list_of_words.index(word)
	list_of_words = [list_of_words[index]] + list_of_words[0:index] + list_of_words[(index + 1):]
	return index, list_of_words
# Program flow for the encode
def start_encoding(txt_name):
	is_text_file(txt_name)
	mtf_name = txt_name.replace("txt", "mtf")
	txt_file = open(txt_name, "r")
	mtf_file = open(mtf_name, "wb")
	_magic_numbers(mtf_file)
	all_words = []
	total_number_of_words = 0
	for input_line in txt_file:
		words_in_line = input_line.split(" ")
		has_new_line = False
		words_are_the_same = False
		code = 0
		for current_word in words_in_line:
			if current_word.endswith("\n"):
				has_new_line = True
				current_word = current_word.replace("\n", "")
			try:
				word_index, all_words = _move_word(current_word,all_words)
				code = word_index + 1
				words_are_the_same = True
			except ValueError:
				words_are_the_same = False
			len_of_current_word = len(current_word)
			if words_are_the_same == False and len_of_current_word > 0:
				all_words.insert(0, current_word)
				total_number_of_words += 1
				code = total_number_of_words
			if (has_new_line == True and len_of_current_word > 0) or (has_new_line == False):
				_mtf_write(mtf_file, code)
			if words_are_the_same == False:
				_mtf_write(mtf_file, current_word)
			if has_new_line == True:
				_mtf_write(mtf_file, "\n")
	mtf_file.close()
	txt_file.close()
# -------------- mtf2text ----------------
# decode
def decode(names):
	if isinstance(names, list):
		for mtf_name in names:
			start_decoding(mtf_name)
	else:
		start_decoding(names)
# verifies that the input file has the .mtf termination
def _is_mtf_file(file_name):
	name = file_name.split(".")
	if not name[1].startswith("mtf"):
		print("Wrong format of file given. Expected .mtf.")
		sys.exit()
# check the first bytes of the mtf file are the magic numbers
def _mtf_file_start_magic(mtf):
	byte = mtf.read(1)
	if not byte.isdigit():
		print("This is not mtf file")
		exit(1)


# adding the new words to the list and writing the word to the file
def _word_to_file(word, list_of_words, txt):
	copy_of_new_word = word  # creates local copy of the word to be added to the list
	copy_of_word = word.replace("\n", "")  # takes out the new line so that the word in the list doesn't have it
	list_of_words.insert(0, copy_of_word)  # adds word to the list
	if "\n" not in word:  # adds a space if the word doesn't have a new line
		word += " "
	txt.write(word)  # writes word to file
# creates the next word from the file stream
def create_next_word(all_words, txt_file,mtf_file):
	new_word = ""  # string used to buid the word from the next bytes
	while True:
		current_byte = mtf_file.read(1)  # read next byte
		int_of_current_byte = int.from_bytes(current_byte, "big")  # get integer representation
		if int_of_current_byte < 128 and current_byte != b"":  # next byte is not a coding number and the byte is not NULL
			new_word += current_byte.decode("ISO-8859-1")  # add the next character to the word
		else:
			_word_to_file(new_word, all_words,txt_file)  # add the finished word to the list of words
			break
	return current_byte, int_of_current_byte, all_words, txt_file, mtf_file  # returns the modified variables
# repeated word, writes it to the file txt, and moves to the next byte in mtf file
def _repeated_word(int_of_byte, byte, txt, words, mtf):
	index = int_of_byte
	byte = mtf.read(1)
	word_to_write = words[index]
	if byte.decode("ISO-8859-1") != "\n":
		word_to_write += " "
	txt.write(word_to_write)
	return byte, index
# Program flow for the decode
def start_decoding(mtf_name):
	_is_mtf_file(mtf_name)
	txt_name = mtf_name.replace("mtf", "txt")
	txt_file = open(txt_name, "w")
	mtf_file = open(mtf_name, "rb")
	# _mtf_file_start_magic(mtf_file)
	current_byte = mtf_file.read(1)
	# current_byte = mtf_file.read(1)
	all_words = []
	list_of_ints = []

	while current_byte != b"":
		int_of_current_byte = int.from_bytes(current_byte, "big")
		if int_of_current_byte >= 128 and int_of_current_byte <= 248:
			int_of_current_byte -= 129
			if int_of_current_byte not in list_of_ints:
				list_of_ints.append(int_of_current_byte)
				current_byte, int_of_current_byte, all_words, txt_file, mtf_file = create_next_word(all_words, txt_file, mtf_file)
			else:
				current_byte, index = _repeated_word(int_of_current_byte, current_byte, txt_file, all_words,mtf_file)
				all_words = [all_words[index]] + all_words[0:index] + all_words[(index + 1):]  # move word to front
		elif (int_of_current_byte == 249):
			current_byte = mtf_file.read(1)
			int_of_current_byte = int.from_bytes(current_byte, "big") + 121 - 1
			if int_of_current_byte not in list_of_ints:  # the int is seen for the first time

				list_of_ints.append(int_of_current_byte)  # save the int value
				current_byte, int_of_current_byte, all_words, txt_file, mtf_file = create_next_word(all_words, txt_file,mtf_file)
			else:
				current_byte, index = _repeated_word(int_of_current_byte, current_byte, txt_file, all_words,mtf_file)
				all_words = [all_words[index]] + all_words[0:index] + all_words[(index + 1):]
		elif (int_of_current_byte == 250):
			current_byte = mtf_file.read(1)
			int_of_current_byte = int.from_bytes(current_byte, "big")
			quotient = int_of_current_byte
			current_byte = mtf_file.read(1)  # read next byte
			int_of_current_byte = int.from_bytes(current_byte, "big")  # get integer representation

			remainder = int_of_current_byte

			int_of_current_byte = 256 * quotient + remainder + 376 - 1

			if int_of_current_byte not in list_of_ints:  # the int is seen for the first time

				list_of_ints.append(int_of_current_byte)  # save the int value
				current_byte, int_of_current_byte, all_words, txt_file, mtf_file = create_next_word(all_words, txt_file,mtf_file)
			else:
				current_byte, index = _repeated_word(int_of_current_byte, current_byte, txt_file, all_words,mtf_file)
				all_words = [all_words[index]] + all_words[0:index] + all_words[(index + 1):]  # move word to front
		else:
			if current_byte.decode("ISO-8859-1") == "\n":
				txt_file.write("\n")
			current_byte = mtf_file.read(1)
	txt_file.close()
	mtf_file.close()
if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("usage: ./coding.py <filename>")
		sys.exit(1)
	name = sys.argv[1]
	if name.split(".")[1].startswith("mtf"):
		decode(sys.argv[1])
	elif name.split(".")[1].startswith("txt"):
		encode(sys.argv[1])
