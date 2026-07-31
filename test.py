import spacy 
nlp = spacy.load("en_core_web_sm") 
sentences = [
    "My name is nusrat",
    "I'm introvert",
    "I love learning AI",
    "My career goal is to become an AI Engineer",
    'i want to work in Ai field'
]
for sentence in sentences:
    doc = nlp(sentence)
    print(sentence) 
    for token in doc:
        print(token.text, token.pos_)
        print()