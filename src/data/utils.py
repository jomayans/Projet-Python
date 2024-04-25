"""Utility functions for data cleaning.
"""
##############################################################################
############## DEFINE FUNCTIONS FOR DATA CLEANING ############################
##############################################################################


#######################################
### EXTRACT `NAMES` ONLY FROM LIST ####
#######################################


def extract_only_names(cell):
    """Select the element whose attribute is name in a list of json documents.

    Parameters
    ----------
        cell: list, list of json documents,

    Return
    ------
        list: list of names extarcted.

    Example
    -------
    cell = [
        {"id": 15, "name": "Drama"},
        {"id": 30, "name": "Thriller"},
        {"id": 45, "name": "Adventure"}
    ]
    cellnames = extract_only_names(cell)
    print(cellnames)
    >>> ['Drama', 'Thriller', 'Adventure']

    # Application to column `genres`
    movies_df["genres"] = movies_df["genres"].apply(extract_only_names)
    """

    if isinstance(cell, list):
        variable = []
        for element in cell:
            variable.append(element["name"])
    return variable


#######################################
### SELECT FIRST ELEMENT FROM LIST ####
#######################################


def select_first_element(row):
    """Return first element for a list element. Use to process variables with
    list elements.

    Parametersgit 
    ----------
        row: list, list

    Return
    ------
        string: first element of the list.
    """

    if isinstance(row, list):
        return str(row[0])
    else:
        return None
