from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

template = PromptTemplate(template="Provide 5 fact about the {text}",
                          input_variables= ['text'])

chain = template|model|parser

result = chain.invoke({'text': "Education system in India"})

print(result)
print(type(result))
chain.get_graph().print_ascii()
