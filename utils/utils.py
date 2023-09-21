def max_list(l:list, key=lambda x:x):
    nl = []
    for e in l:
        if nl == []:
            nl.append(e)
        elif key(e) > key(nl[0]):
            nl = [e]
        elif key(e) == key(nl[0]):
            nl.append(e)
    return nl
