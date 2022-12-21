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


def select_first_element(x):
    """Return first element for a list element. Use to process variables with
    list elements.
    
    Parameters
    ----------
        x: list, list 
    
    Return
    ------
        string: first element of the list.
    """
    
    if isinstance(x, list):
        return str(x[0])
    
    
# MAPPER.
country_to_continent = {
    'Spain': 'Europe',
    'Italy': 'Europe',
    'India': 'Asia',
    'China': 'Asia',
    'Hong Kong': 'Asia',
    'Belgium': 'Europe',
    'South Korea': 'Asia',
    'Ireland': 'Europe',
    'Denmark': 'Europe',
    'Mexico': 'North America',
    'Czech Republic': 'Europe',
    'New Zealand': 'Australia/Oceania',
    'Russia': 'Europe',
    'Argentina': 'South America',
    'Netherlands': 'Europe',
    'Brazil': 'South America',
    'Bulgaria': 'Europe',
    'South Africa': 'Africa',
    'Switzerland': 'Europe',
    'United Arab Emirates': 'Asia',
    'Austria': 'Europe',
    'Norway': 'Europe',
    'Thailand': 'Asia',
    'Sweden': 'Europe',
    'Hungary': 'Europe',
    'Luxembourg': 'Europe',
    'Finland': 'Europe',
    'Morocco': 'Africa',
    'Greece': 'Europe',
    'Poland': 'Europe',
    'Taiwan': 'Asia',
    'Namibia': 'Africa',
    'Aruba': 'North America',
    'Israel': 'Asia',
    'Slovakia': 'Europe',
    'Romania': 'Europe',
    'Indonesia': 'Asia',
    'Chile': 'South America',
    'Iceland': 'Europe',
    'Venezuela': 'South America',
    'Ecuador': 'South America',
    'Malaysia': 'Asia',
    'Philippines': 'Asia',
    'Belarus': 'Europe',
    'Turkey': 'Europe',
    'Estonia': 'Europe',
    'Ghana': 'Africa',
    'Colombia': 'South America',
    'Afghanistan': 'Asia',
    'Soviet Union': 'Europe',
    'Kuwait': 'Asia',
    'Libyan Arab Jamahiriya': 'Africa',
    'Iran': 'Asia',
    'Singapore': 'Asia', 
    'Uruguay': 'South America', 
    'Peru': 'South America',
    'Cambodia': 'Asia', 
    'Qatar': 'Asia'
}



