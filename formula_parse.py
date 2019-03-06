"""
This program is intended to parse the Mathematical formula
input : string (Latex-formatted formula)
output : list denoting the sequential importance of the formula feature
"""

def detect_end(formula, start):
	#to detect the end of the formula, given the starting point
	#start is the number where the string of formula starts
	detector, current_idx = 1, start
	while (detector > 0):
		current_idx += 1
		if formula[current_idx] == "{":
			detector += 1
		elif formula[current_idx] == "}":
			detector -= 1
	return current_idx

def inner_formula(formula, start):
	#to return the formula contained within the bigger formula
	return formula[start+1:detect_end(formula, start)]

def formula_parse(formula):
	#first phase is to detect the important words (current stage is just frac and int)

	form_length = len(formula)
	if formula[0] == "\\": #meaning it's started with the formula notation
		if formula[1:4] == "int": #only handle singular formula so far
			start = formula.index("{")
			return ["\\int", formula_parse(inner_formula(formula, start))]

		elif formula[1:5] == "frac": #handling the regular fraction
			start =  formula.index("{")
			end_1 = detect_end(formula, start) + 1
			print(end_1)
			return ["\\frac", formula_parse(inner_formula(formula, start)),  formula_parse(inner_formula(formula, end_1))]
		
	else: #handling some other formula
		return [formula]

print(inner_formula("\\frac{4x-2}{\int{2x^2-4x+1}dx}", 5))
print(formula_parse("\\int{\\frac{3x+1}{5x-3}}dx"))
