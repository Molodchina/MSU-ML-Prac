def find_modified_max_argmax(l, f):
    l = [f(x) for x in l if type(x) == int]
    return () if not l else (max(l), l.index(max(l)))
