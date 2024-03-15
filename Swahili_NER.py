from model import GLiNER

# model = GLiNER.from_pretrained("urchade/gliner_base", local_files_only=True)

model = GLiNER.from_pretrained("gliner_mult", local_files_only=True)

text = """
Mimi na baba yangu tunalima mazao mbalimbali toka mwaka 2020
"""

labels = ["person",  "date",  "agriculture",  "place", "ANIMAL" , "ORGANISM" ]

entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])