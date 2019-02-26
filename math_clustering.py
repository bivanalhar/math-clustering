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
		if (ord(text_ocr[i]) > 44032) and (ord(text_ocr[i]) < 55203): #the text is korean alphabet (hence omitted)
			math_formula = math_formula + " " #change into space
		else:
			math_formula = math_formula + text_ocr[i] #keep the non-korean variable intact
	math_formula = math_formula.strip() #omit the collection of spaces in the beginning and the end
	math_formula = " ".join(math_formula.split()) #get the final version of the math expression

	return math_formula

#Phase 2 : Extracting the information from the csv file
database_math = [] #the list to contain all the problems from the barista file

with open("barista_data.csv") as barista_csv:
	barista_reader = csv.reader(barista_csv, delimiter = ",")
	for row in barista_reader:
		if (row[8] == "mathpresso_ocr"): #need to change after looking at the real csv file
			pair_store = (row[2], formula_parse(row[-1]))
			database_math.append(pair_store)

#Phase 3 : Do the real clustering
#(will think about it soon enough)

