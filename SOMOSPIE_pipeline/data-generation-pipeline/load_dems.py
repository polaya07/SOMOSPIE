#!/usr/bin/python3
def load_data(file:str, out_dir:str)->str:
    import urllib.request
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(file,'r') as text:
        for line in text:
            if not os.path.exists(out_dir+line.split("/")[-1]):
                urllib.request.urlretrieve(line, out_dir+line.split("/")[-1])
    return out_dir
    