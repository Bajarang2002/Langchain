from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableParallel


load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")


parser = StrOutputParser()

prompt1 = PromptTemplate(template="Generate the joke on the topic {topic}",
                        input_variables=['topic'])

prompt2 = PromptTemplate(template="explain the following joke \n {text}",
                        input_variables=['text'])

gen_joke = RunnableSequence(prompt1,model,parser)


parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(gen_joke,parallel_chain)


result = chain.invoke({'topic': 'Computer'})
print(result)

