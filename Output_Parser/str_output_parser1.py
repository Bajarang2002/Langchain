from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")



template1 = PromptTemplate(template = "Provide detail report on {topic}",
                           input_variables=['topic'])

template2 = PromptTemplate(template= "Provide 5 lines summary on \n {text}",
                           input_variables=['text'])

parser = StrOutputParser()

chain = template1|model|parser|template2|model|parser

result = chain.invoke({"topic": "Education system in India"})
print(result)