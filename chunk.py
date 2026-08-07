# Function that breaks a long text into fixed-size chunks
def chunk_text(text):
    # Empty list to store the chunks
    chunks = []
    # Loop through the text in jumps of 500 characters
    for i in range(0, len(text), 500):
        # Take a 500-character piece starting at position i
        chunk = text[i:i+500]
        # Add this piece to the list
        chunks.append(chunk)
    # Return the list of all chunks
    return chunks

# This part runs only when the file is run directly (not when imported)
if __name__ == "__main__":
    # Create a long sample text for testing
    sample = "AI is great. " * 100
    # Break the sample into chunks
    result = chunk_text(sample)
    # Print how many chunks were made
    print("Total chunks:", len(result))
    # Print the first chunk
    print("First chunk:", result[0])