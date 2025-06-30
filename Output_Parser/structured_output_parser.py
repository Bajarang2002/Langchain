from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")


schema = [ResponseSchema(name= "fact1",description=" Provide detail explaination about fact1"),
          ResponseSchema(name = "fact2",description="Provide detail Explaination about fact2"),
          ResponseSchema(name= "fact3",description="Provide detail Explaination about fact3")]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template="Provide me  fact about the {topic} \n {format_instruction}",
                          input_variables=['topic'],
                          partial_variables={'format_instruction': parser.get_format_instructions()})

chain =  template|model|parser

result = chain.invoke({'topic': "Indian Democrocy"})
print(result)
print(type(result))



