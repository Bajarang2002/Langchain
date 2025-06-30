from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()
''
llm =HuggingFaceEndpoint(repo_id="openai-community/gpt2",
                         task= "Text Generation")

model = ChatHuggingFace(llm =llm)

class Person(BaseModel):
     name : str=Field(decription= "Name of the person")
     age :int= Field(gt=18,description="Age of the person")
     city :str= Field(description="Name of the city")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template="Provide name, age,city of the litral {country} \n {format_instruction}",
                          input_variables=['country'],
                          partial_variables={'format_instruction': parser.get_format_instructions()})

chain =  template|model|parser

result = chain.invoke({'country': "USA"})
print(result)
print(type(result))



