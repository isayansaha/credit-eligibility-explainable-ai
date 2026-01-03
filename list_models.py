import google.generativeai as genai

genai.configure(api_key="AIzaSyC24D_AiCIOPFZyfUDWh5NvmwKle6s-pPY")

for m in genai.list_models():
    print(m.name)
