# can-we-solve-every-leetcode-using-only-one-template
defenitely no
How It Works



Each problem is implemented once as a function
Functions are registered using:
@BANK.register("Problem Name")
When you call:
BANK.solve("Problem Name", args...)

→ it finds the correct function and runs it

Flow:
Input → title string
      → normalize (lowercase, remove symbols)
      → lookup in BANK.routes
      → call mapped function
      → return result
