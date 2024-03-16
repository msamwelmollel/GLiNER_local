import json


from model import GLiNER

# model = GLiNER.from_pretrained("urchade/gliner_base", local_files_only=True)

model = GLiNER.from_pretrained("C:/Users/msamwelmollel/GLiNER/gliner_mult/", local_files_only=True)

text = """
Mimi na baba yangu tunalima mazao mbalimbali toka mwaka 2020
"""

labels = ["person",  "date",  "agriculture",  "place", "ANIMAL" , "ORGANISM" ]

entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])


# Parsing JSON string into a Python list
labels = json.loads(label.json)

# Now 'labels' is a Python list of NER labels.
labels