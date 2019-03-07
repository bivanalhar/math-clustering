"""
This program is intended to cluster the Math problem into
several categories (still unknown) based on their similarity
in the Math expression contained within the problem itself

The program itself consists of several sub-parts, considering
the type of the problem we're handling on right now (which con-
sists of korean alphabets together with mathematical formula)
"""

import csv
import re

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

with open("math_eng.csv") as math_csv:
	math_reader = csv.reader(math_csv, delimiter = ",")	

	#firstrow, secondrow = next(barista_reader), next(barista_reader)
	#print(firstrow[2], firstrow[13], firstrow[-1])
	
	for row in math_reader:
		#if row[0] == "6747725":
		#	print(row[-1])
		row_splice = row[-1].split("\n")
		row_unite = " ".join(row_splice).strip()
		
		#if row[0] == 'box_id':
		#	print(row)

		if row[0] != 'box_id' and row_unite != "":
			row_store = [int(row[0]), int(row[1]), row[4], row_unite]
			database_math.append(row_store)

# database_sat_act = []
database_other = []
database_calculus = []
database_linalg = []
database_probstat = []
database_engineering = []
# database_numanalysis = []
# database_analysis = []

#database_math = database_math[1:]

for i in range(len(database_math)):
	# if "SAT" in database_math[i][2] or "ACT" in database_math[i][2] or "Sat" in database_math[i][2]:
	# 	database_sat_act.append(database_math[i])
	if "Calculus" in database_math[i][2] or "Div" in database_math[i][2]:
		database_calculus.append(database_math[i])
	elif "Linear Algebra" in database_math[i][2]:
		database_linalg.append(database_math[i])
	elif "Probability" in database_math[i][2] or "Statistic" in database_math[i][2]:
		database_probstat.append(database_math[i])
	elif "Engineering" in database_math[i][2]:
		database_engineering.append(database_math[i])
	# elif "Numerical" in database_math[i][2]:
 #                database_numanalysis.append(database_math[i])
	# elif "Analysis" in database_math[i][2]:
 #                database_analysis.append(database_math[i])
	else:
		database_other.append(database_math[i])
	#print(database_math[i])

# for i in range(len(database_engineering)):
# 	print(database_engineering[i])

# print("The number of English questions are", len(database_math))
# print("The number of SAT or ACT-style questions are", len(database_sat_act))
# print("The number of Calculus questions are", len(database_calculus))
# print("The number of Linear Algebra questions are", len(database_linalg))
# print("The number of Probability and Statistics questions are", len(database_probstat))
# print("The number of Engineering questions are", len(database_engineering))
# print("The number of Numerical Analysis questions are", len(database_numanalysis))
# print("The number of Analysis questions are", len(database_analysis))

with open('eng_calculus.csv', mode = 'w') as calculus_file:
	fieldnames = ['question_id', 'text_contain', 'label']
	writer = csv.DictWriter(calculus_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(database_calculus)):
		writer.writerow({'question_id': database_calculus[i][0], 'text_contain' : database_calculus[i][-1], 'label' : '0'})

with open('eng_linalg.csv', mode = 'w') as linalg_file:
	fieldnames = ['question_id', 'text_contain', 'label']
	writer = csv.DictWriter(linalg_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(database_linalg)):
		writer.writerow({'question_id': database_linalg[i][0], 'text_contain' : database_linalg[i][-1], 'label' : '1'})

with open('eng_probstat.csv', mode = 'w') as probstat_file:
	fieldnames = ['question_id', 'text_contain', 'label']
	writer = csv.DictWriter(probstat_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(database_probstat)):
		writer.writerow({'question_id': database_probstat[i][0], 'text_contain' : database_probstat[i][-1], 'label' : '2'})

with open('eng_engineering.csv', mode = 'w') as engineering_file:
	fieldnames = ['question_id', 'text_contain']
	writer = csv.DictWriter(engineering_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(database_engineering)):
		writer.writerow({'question_id': database_engineering[i][0], 'text_contain' : database_engineering[i][-1]})


