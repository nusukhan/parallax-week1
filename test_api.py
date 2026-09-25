# ============================================================
# test_api.py — Week 10 (Line 5): Concurrent request test
#
# WHAT THIS FILE DOES:
# Sends several requests to the running API AT THE SAME TIME
# (concurrently) to check the API stays stable under load and
# does not crash when many users hit it together.
#
# NOTE: The API must already be running (uvicorn api:app --reload)
# in another terminal before you run this file.
# ============================================================

# requests -> lets us send HTTP requests to our API
import requests
# threading -> lets us send many requests at the SAME time
import threading
# time -> measure how long all the requests take together
import time

# ---------- The API address ----------
# This is the /query endpoint of our running API.
URL = "http://127.0.0.1:8000/query"

# ---------- The test questions ----------
# We will send all of these at the same time.
questions = [
    "what is physics",
    "what is energy",
    "what is an atom",
    "how do stars form",
    "what is light"
]

# This list will collect the result (question, status code) of each request.
results = []

# ============================================================
# Function that sends ONE request to the API
# ============================================================
def send_request(question):
    try:
        # Send a POST request with the question as JSON, wait up to 60s
        response = requests.post(URL, json={"question": question}, timeout=60)
        # Save the question and the HTTP status code (200 = success)
        results.append((question, response.status_code))
        print("Done:", question, "-> status", response.status_code)
    except Exception as e:
        # If the request fails completely, record it as an ERROR
        results.append((question, "ERROR"))
        print("Failed:", question, "->", str(e))

# ============================================================
# Send ALL requests at the same time (concurrently)
# ============================================================
print("Sending", len(questions), "requests at the same time...")
start = time.time()

# A "thread" runs a task in parallel. We make one thread per question
# so all questions are sent together, not one after another.
threads = []
for q in questions:
    t = threading.Thread(target=send_request, args=(q,))
    threads.append(t)
    t.start()      # start this request (runs alongside the others)

# Wait for every thread (request) to finish before continuing.
for t in threads:
    t.join()

# Total time taken for all requests together
total_time = round(time.time() - start, 2)

# ============================================================
# Show the results
# ============================================================
print()
print("=" * 50)
print("CONCURRENT TEST RESULTS")
print("=" * 50)

# Count how many requests returned status 200 (success)
success = 0
for question, status in results:
    if status == 200:
        success = success + 1

print("Successful responses:", success, "/", len(questions))
print("Total time for all requests:", total_time, "seconds")

# If all succeeded, the API handled concurrent load without crashing.
if success == len(questions):
    print("API stayed stable!")
else:
    print("Some requests failed.")