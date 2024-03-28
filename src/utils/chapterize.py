# https://github.com/JonathanReeve/chapterize/tree/master

import re
import logging


def parse_document_in_chapters(document):
    lines = document.split("\n")
    headings = get_headings(lines)
    headings = ignore_TOC(headings)
    return get_text_between_headings(headings, lines)


def get_headings(lines):
    # Form 1: Chapter I, Chapter 1, Chapter the First, CHAPTER 1
    # Ways of enumerating chapters, e.g.
    arabicNumerals = '\d+'
    romanNumerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
    numberWordsByTens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty',
                            'seventy', 'eighty', 'ninety']
    numberWords = ['one', 'two', 'three', 'four', 'five', 'six',
                    'seven', 'eight', 'nine', 'ten', 'eleven',
                    'twelve', 'thirteen', 'fourteen', 'fifteen',
                    'sixteen', 'seventeen', 'eighteen', 'nineteen'] + numberWordsByTens
    numberWordsPat = '(' + '|'.join(numberWords) + ')'
    ordinalNumberWordsByTens = ['twentieth', 'thirtieth', 'fortieth', 'fiftieth', 
                                'sixtieth', 'seventieth', 'eightieth', 'ninetieth'] + \
                                numberWordsByTens
    ordinalNumberWords = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 
                            'seventh', 'eighth', 'ninth', 'twelfth', 'last'] + \
                            [numberWord + 'th' for numberWord in numberWords] + ordinalNumberWordsByTens
    ordinalsPat = '(the )?(' + '|'.join(ordinalNumberWords) + ')'
    enumeratorsList = [arabicNumerals, romanNumerals, numberWordsPat, ordinalsPat] 
    enumerators = '(' + '|'.join(enumeratorsList) + ')'
    form1 = 'chapter ' + enumerators

    # Form 2: II. The Mail
    enumerators = romanNumerals
    separators = '(\. | )'
    titleCase = '[A-Z][a-z]'
    form2 = enumerators + separators + titleCase

    # Form 3: II. THE OPEN ROAD
    enumerators = romanNumerals
    separators = '(\. )'
    titleCase = '[A-Z][A-Z]'
    form3 = enumerators + separators + titleCase

    # Form 4: a number on its own, e.g. 8, VIII
    arabicNumerals = '^\d+\.?$'
    romanNumerals = '(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$'
    enumeratorsList = [arabicNumerals, romanNumerals]
    enumerators = '(' + '|'.join(enumeratorsList) + ')'
    form4 = enumerators

    pat = re.compile(form1, re.IGNORECASE)
    # This one is case-sensitive.
    pat2 = re.compile('(%s|%s|%s)' % (form2, form3, form4))

    # TODO: can't use .index() since not all lines are unique.

    headings = []
    for i, line in enumerate(lines):
        if pat.match(line) is not None:
            headings.append(i)
        if pat2.match(line) is not None:
            headings.append(i)

    if len(headings) < 3:
        logging.info('Headings: %s' % headings)
        logging.error("Detected fewer than three chapters. This probably means there's something wrong with chapter detection for this book.")
        return headings

    endLocation = get_end_location(lines)

    # Treat the end location as a heading.
    headings.append(endLocation)

    return headings

def get_end_location(lines):
        """
        Tries to find where the book ends.
        """
        ends = ["End of the Project Gutenberg EBook",
                "End of Project Gutenberg's",
                "\*\*\*END OF THE PROJECT GUTENBERG EBOOK",
                "\*\*\* END OF THIS PROJECT GUTENBERG EBOOK"]
        joined = '|'.join(ends)
        pat = re.compile(joined, re.IGNORECASE)
        endLocation = None
        endLine = 'None'
        for line in lines:
            if pat.match(line) is not None:
                endLocation = lines.index(line)
                endLine = lines[endLocation]
                break

        if endLocation is None: # Can't find the ending.
            logging.info("Can't find an ending line. Assuming that the book ends at the end of the text.")
            endLocation = len(lines)-1 # The end
            endLine = 'None'

        logging.info('End line: %s at line %s' % (endLine, endLocation))
        return endLocation

def ignore_TOC(headings):
        """
        Filters headings out that are too close together,
        since they probably belong to a table of contents.
        """
        pairs = zip(headings, headings[1:])
        toBeDeleted = []
        for pair in pairs:
            delta = pair[1] - pair[0]
            if delta < 4:
                if pair[0] not in toBeDeleted:
                    toBeDeleted.append(pair[0])
                if pair[1] not in toBeDeleted:
                    toBeDeleted.append(pair[1])
        logging.debug('TOC locations to be deleted: %s' % toBeDeleted)
        for badLoc in toBeDeleted:
            index = headings.index(badLoc)
            del headings[index]
        return headings

def get_text_between_headings(headings, lines):
    chapters = []
    lastHeading = len(headings) - 1
    for i, headingLocation in enumerate(headings):
        if i != lastHeading:
            nextHeadingLocation = headings[i+1]
            chapters.append(lines[headingLocation+1:nextHeadingLocation])
    return chapters