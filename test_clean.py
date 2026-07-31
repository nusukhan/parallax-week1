from clean import clean_html, clean_spaces, clean_special

assert clean_html("<p>hi</p>") == "hi"
assert clean_spaces("a    b") == "a b"
assert clean_special("&%hello$#@") == "hello"
assert clean_spaces("") == ""
assert clean_html("<b>AI</b> <i>rocks</i>") == "AI rocks"

print("All tests passed!")