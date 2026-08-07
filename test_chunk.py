from chunk import chunk_text

# Test 1: normal text makes correct chunks
result = chunk_text("A" * 1000)
assert len(result) == 2

# Test 2: very short text
result = chunk_text("hi")
assert len(result) == 1

# Test 3: text without clear sentences (no full stops)
result = chunk_text("a" * 600)
assert len(result) == 2

print("All chunking tests passed!")