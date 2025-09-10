# Smart Airline Passenger Complaint Analyzer

# Technologies used 

  Python \
  FastAPI \
  Hugging Face Transformers \
  PyTorch \
  Data Version Control - For model Versioning and productibility \
  Uvicorn


Steps to run 
Go to the command prompt
conda activate aai_gpu_env

set KMP_DUPLICATE_LIB_OK=TRUE
uvicorn main:app --reload --host 127.0.0.1 --port 8000

The backend run on
 http://127.0.0.1:8000

 Check the health 
Should display as
{"status":"ok","device":"cuda"}

