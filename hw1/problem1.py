#!/usr/bin/python3
year=2020
# Your code should be below this line

def leapyear(year):
    if year % 4 == 0:
        return True
    else:
        return False

if __name__ == "__main__":
    # years = [800, 2020, 583, 1100, 1994, 2015, 3132]
    #
    # for yearp in years:
    #     leap = leapdet(yearp)
        #print(year, leap)
    leap = leapyear(year)
    print(leap)
