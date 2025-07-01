from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()


template1 = PromptTemplate(template="Provide the information about the given {topic}",
                           input_variables=['topic'])

template2 = PromptTemplate(template="Provide the 5 line summary about the {text}",
                           input_variables=['text'])


chain = template1|model|parser|template2|model|parser

result = chain.invoke({"topic":"Impact of AI on jobs"})
print(result)

chain.get_graph().print_ascii()