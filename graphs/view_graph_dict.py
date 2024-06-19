import pickle

with open('graph_dictionary.pkl', 'rb') as f:
    d = pickle.load(f)

fs_min = 0
fs_max = 0
for graph in d:

    if d[graph]['min_preserved'] == False:
        fs_min += 1
    if d[graph]['max_preserved'] == False:
        fs_max += 1

print(f'{fs_min} graphs fail on negative')
print(f'{fs_max} graphs fail on positive')
