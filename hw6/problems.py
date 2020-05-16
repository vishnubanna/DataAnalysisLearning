import re

def problem1(searchstring):
    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    string = searchstring
    string = re.sub(r"[\(][\d]{,3}[\)][ ]{,1}[\d]{,3}[\-][\d]{,4}", "", searchstring)
    if string == "":
        return True
    #print(string)
    string = re.sub(r"[\d]{,3}[\-][\d]{,3}[\-][\d]{,4}", "", searchstring)
    if string == "":
        return True
    #print(string)
    string = re.sub(r"[\d]{,3}[\-][\d]{,4}", "", searchstring)
    if string == "":
        return True
    #print(string)
    return False
        
def problem2(searchstring):
    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    out = re.search(r"[\d]+[ ]\b([A-Z]+[\w\d']*)([\sA-Z]+[\w\d']*)*\b", searchstring)
    #print(out[0])
    output = re.sub(r"(St)|(Ave)|(Dr)|(Rd)","",out[0])
    output = re.sub(r"[\d]+[ ]","",output)

    return output
    
def problem3(searchstring):
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    string = problem2(searchstring)
    string2 = string[::-1]
    #print(string2)
    output = re.sub(f" {string}", f"{string2} ", searchstring)
    #print(output)
    return output


if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1(' 765-494-4600 ')) #False
    print(problem1('(765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))
