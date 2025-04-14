try:
    # some code that may raise an exception
    print("In try block")
    # uncomment to raise an exception
    raise ValueError("An error occurred")
except Exception as e:
    print(f"Caught an exception: {e}")
    raise ValueError("abc")

print("This code runs if there was no unhandled exception.")
