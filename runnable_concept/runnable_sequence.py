from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")


parser = StrOutputParser()

prompt1 = PromptTemplate(template="Generate the joke on the topic {topic}",
                        input_variables=['topic'])

promt2 = PromptTemplate(template="explain the following joke \n {text}",
                        input_variables=['text'])


chain = RunnableSequence(prompt1, model, parser,promt2,model,parser)


result = chain.invoke({'topic': 'Computer'})
print(result)

