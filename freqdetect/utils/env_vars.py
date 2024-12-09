import json
with open("./utils/PAIR_ABBREV.json", 'r') as fptr:
    PAIR_ABBREV = json.load(fptr)
with open("./utils/PERTURB_TYPE_TO_VALS.json", 'r') as fptr:
    PERTURB_TYPE_TO_VALS = json.load(fptr)
with open("./utils/DATASETNAME_TO_NUMBER.json", 'r') as fptr:
    DATASETNAME_TO_NUMBER = json.load(fptr)