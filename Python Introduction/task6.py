def check(s: str, filename: str):
    with open(filename, 'w+') as f:
        keys = dict(sorted(dict.fromkeys(s.lower().split(' '), 0).items()))
        for x in keys:
            keys[x] = s.lower().split(' ').count(x)
            f.write(f"%s %s%s" % (x, keys[x], '' if len(keys) <= 1 else '\n'))
