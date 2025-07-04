from langchain_google_genai import GoogleGenerativeAI
from  dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

prompt = PromptTemplate(template="Provide the 5 line poem from the document {poem}",
                        input_variables=['poem'])


loader = TextLoader("cricket.txt",encoding= 'utf-8')

docs = loader.load()

print(docs)
print(type(docs))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt |model| parser

result = chain.invoke({'poem': docs[0].page_content})
print(result)


    

