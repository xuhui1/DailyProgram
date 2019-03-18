# hw3-CSPs
HW3: Sudoku (Constraint Satisfaction Problems)
1. if run 'python SudokuSolver.py euler.txt' and will output a file 'Euler.out' every line is the result of one sudoku.
    if run 'python SudokuSolver.py magictour.txt' and will output a file 'Magic Tour.out' every line is the result of one sudoku.
2. We use 2-dimensional(9 * 9) list represent a sudoku and character '.' replace with the number 0, for ease of calculation.
    Use dict structure to save the every cell candidate set.
3. The most common practice is violent search, from 1 to 9. The step is:
    1) Ergodic each non-zero position in sequence.
    2) Try to fill in the numbers 1 to 9 in the grid.
    3) Recursive judgment of whether the number of fillings conforms to the rules.
    4) When all the numbers are filled out, exit the loop
    We can statistical candidate sets for each lattice, and then sort the number of candidate sets. The step is:
    1) Find out the number that can be filled in at the position of 0, and store the position and the
    number in the dictionary in the form of key and value.
    2) Sort the values in the dictionary in ascending order, select the first one.
    3) Record the filling process in the list.
    4) Update 1 and 2 steps, if there is a blank data to fill in, indicating that there is a problem with the choice
    of a previous step, go back, change the value, and then go back to step 1.
    5) When all the numbers are filled out, exit the loop
4. The method of violent search: cost 3201.41s in data of magictour.txt and cost 9.77s in data of euler.txt
    The method of sort candidate set: cost 325.21s in data of magictour.txt and cost 1.5s in data of euler.txt
    we can find the method of violent search cost time is very bigger.
5. I think it's a valuable experience. From the simplest way to six times performance improvement, the structure hasn't changed much, it's amazing.
    But this method is hard to think of, just changing the search order.
6. Develop violent search about cost 1 day contains debug. and develop sort candidate set about cost 4 hour.
7. The above code is done by us independently.
