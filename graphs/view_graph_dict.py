import pickle

with open('graph_dictionary.pkl', 'rb') as f:
    d = pickle.load(f)

print(d)
