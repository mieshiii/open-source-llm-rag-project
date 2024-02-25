import runpod

def is_even(job):
  job_input = job["input"]
  num = job_input["number"]

  if not isinstance(num, int):
    return {"error": "Wrong input type, input an integer."}
  
  if num % 2 == 0:
    return True
  
  return False

runpod.serverless.start({"handler": is_even})