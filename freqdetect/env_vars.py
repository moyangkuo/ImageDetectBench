import json
with open("./PAIR_ABBREV.json", 'r') as fptr:
    PAIR_ABBREV = json.load(fptr)
with open("./PERTURB_TYPE_TO_VALS.json", 'r') as fptr:
    PERTURB_TYPE_TO_VALS = json.load(fptr)
with open("./DATASETNAME_TO_NUMBER.json", 'r') as fptr:
    DATASETNAME_TO_NUMBER = json.load(fptr)