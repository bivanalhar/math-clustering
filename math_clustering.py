"""
This program is intended to cluster the Math problem into
several categories (still unknown) based on their similarity
in the Math expression contained within the problem itself

The program itself consists of several sub-parts, considering
the type of the problem we're handling on right now (which con-
sists of korean alphabets together with mathematical formula)
"""

#Phase 1 : Obtaining the complete text of the problem
#(for this one, I will try to use Mathpresso OCR only for now, as MP-OCR is better in parsing math formula than Google)

#Phase 2 : Omitting the korean keywords of the problem
#(we need to focus entirely on the Math formula itself instead of the details surrounding it)

#Phase 3 : Do the real clustering
#(will think about it soon enough)