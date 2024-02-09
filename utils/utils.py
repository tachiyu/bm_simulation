from pathlib import Path
import pickle
import re

def sort_file_with_tSiz(paths):
    def get_tSiz(path):
        return int(re.search(r'table_width_(\d*)', path.name).group(1))
    return sorted(paths, key=get_tSiz)