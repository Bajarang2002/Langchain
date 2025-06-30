from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")

parser = JsonOutputParser()

template = PromptTemplate(template="Provide me 5 fact about the {topic} \n {format_instruction}",
                          input_variables=['topic'],
                          partial_variables={'format_instruction': parser.get_format_instructions()})

chain =  template|model|parser

result = chain.invoke({'topic': "Indian Democrocy"})
print(result)
print(type(result))



