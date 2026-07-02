import spacy # spacy import 
nlp = spacy.load("en_core_web_sm") # load english model and give a name npl
 # 5 sentence of list 
sentences =[
    " My name is nusrat",
    "I'm introvert",
    "I love learning AI",
    "My career goal is to become an AI Engineer",
    'i want to work in Ai field'
]
for sentence in sentences:
    doc =nlp(sentence) # jo sentence chle ga usko nlp (english ke process se guzro or phr result doc main rakheo)
    print(sentence) # show on screen 
    for token in doc: # mtlb us sentence ke har token (word) ke liye  neeche wala kaam kro  
        print(token.text,token.pos_) # Har word ke sath uska tag dekheo  mtlb behr wala loop har sentence ke lye hai or andr wala loop surf us sentence ke har word ke lye 
        print()
# EXAMPLE 
#token.text _>word ( love )
#token.pos
#  _> us word ka part of speech (verb,noun,pronoun)
