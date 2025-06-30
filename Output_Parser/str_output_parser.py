from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")

template1 = PromptTemplate(template = " Provide Information about {topic}",
                           input_variables= ['topic'])

template2 = PromptTemplate(template=" Provide brief descriptio about provided next \n{text}",
                           input_variables=['text'])


prompt1 = template1.invoke({'topic':"Artificial Intelligence"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' :"Indian Politics"})
result1 = model.invoke(prompt2)
print(result1)



