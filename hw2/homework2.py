def histogram(data, n, l, h):
    # data is a list
    # n is an integer number of bins
    # l and h are floats data set range

    # Write your code here
    width = float(h - l)/n
    hist = [0]*n#list(range(int(n)))

    ## thought question can you d o it in o(n) with out sorting it, maybe use number to calculate most likely range
    data.sort()
    datlen = len(data)
    place = 0

    for i in range(n):
        for j in range(place, datlen):
            if data[j] >= (l + i * width) and data[j] < (l + (i + 1) * width):
                hist[i] += 1
            else:
                place = j
                break

    # return the variable storing the histogram
    return hist
    # Output should be a list

    pass


def addressbook(name_to_phone, name_to_address):
    #name_to_phone and name_to_address are both dictionaries

    # Write your code here
    adr = set(name_to_address.values())
    fin_book = dict();
    dif_num = dict();

    for adress in adr:
        fin_book[adress] = [[], "-1"]

    for name in name_to_address.keys():
        if fin_book[name_to_address[name]][1] == "-1":
            fin_book[name_to_address[name]][0].append(name)
            fin_book[name_to_address[name]][1] = name_to_phone[name]
            fin_book[name_to_address[name]] = tuple(fin_book[name_to_address[name]])
        elif fin_book[name_to_address[name]][1] != name_to_phone[name]:
            fin_book[name_to_address[name]][0].append(name)
            if fin_book[name_to_address[name]][0][0] not in dif_num:
                dif_num[fin_book[name_to_address[name]][0][0]] = [name]
            else:
                dif_num[fin_book[name_to_address[name]][0][0]].append(name)

    for name in dif_num.keys():
        names = ", ".join(dif_num[name])
        print(f" Warning: {names} has a different number for {name_to_address[name]} than {name}. Using the number for {name}.")


    return fin_book
