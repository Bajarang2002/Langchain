from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch

load_dotenv()

def word_count(topic):
    return len(topic.split())


model = GoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(template="Generate the information about {topic}",
                        input_variables=['topic'])

prompt2 = PromptTemplate(template=" Summarize the following given topic {text}",
                        input_variables=['text'])


gen_info = RunnableSequence(prompt1,model,parser)

parallel_branch = RunnableBranch(
    (lambda x: len(x.split())>200,prompt2|model|parser),
    RunnablePassthrough()
)

chain = RunnableSequence(gen_info, parallel_branch)

result = chain.invoke({'topic':'Artificial Intelligence'})
print(result)

