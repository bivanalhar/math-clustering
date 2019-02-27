"""
This program is intended to cluster the Math problem into
several categories (still unknown) based on their similarity
in the Math expression contained within the problem itself

The program itself consists of several sub-parts, considering
the type of the problem we're handling on right now (which con-
sists of korean alphabets together with mathematical formula)
"""

import csv

#Phase 1 : Defining the function to omit the korean keywords of the problem
#(we need to focus entirely on the Math formula itself instead of the details surrounding it)
def formula_parse(text_ocr):
	math_formula = ""
	for i in range(len(text_ocr)):
		if (ord(text_ocr[i]) >= 32) and (ord(text_ocr[i]) <= 126): #the text is latin alphabet (hence included)
			math_formula = math_formula + text_ocr[i] #change into space
		else:
			math_formula = math_formula + " " #keep the non-korean variable intact
	math_formula = math_formula.strip() #omit the collection of spaces in the beginning and the end
	math_formula = " ".join(math_formula.split()) #get the final version of the math expression

	return math_formula

#Phase 2 : Extracting the information from the csv file
database_math = [] #the list to contain all the problems from the barista file

with open("barista_data.csv") as barista_csv:
	barista_reader = csv.reader(barista_csv, delimiter = ",")
	
	#firstrow, secondrow = next(barista_reader), next(barista_reader)
	#print(firstrow[2], firstrow[13], firstrow[-1])
	
	for row in barista_reader:
		if (row[13] == "mathpresso_ocr"): #need to change after looking at the real csv file
			pair_store = (row[2], row[-1],  formula_parse(row[-1]))
			database_math.append(pair_store)

database_math = database_math[1:]
print("The number of Barista questions are", len(database_math))

#Phase 3 : Do the real clustering

#at this point, the variable database_math contains all the questions from Barista file.
#we need to split the list into 70:15:15, while 70% will be used for the training file,
#15% will be used for the validation file and the rest is for testing file

data_num = len(database_math)
data_70, data_85 = int(data_num * 0.70), int(data_num * 0.85)

database_train = database_math[:data_70]
database_val = database_math[data_70:data_85]
database_test = database_math[data_85:]
