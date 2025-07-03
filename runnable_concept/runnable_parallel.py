from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel


load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")


prompt1 = PromptTemplate(template="Generate the notes about the topic {topic}",
                        input_variables={'topic'})


prompt2 = PromptTemplate(template="Generate the the descriptive question about the topic {topic}",
                         input_variables=['topic'])

parser = StrOutputParser()

paralled_chain =  RunnableParallel({
    'notes': RunnableSequence(prompt1,model,parser),
    'qustion' : RunnableSequence(prompt2,model,parser)
})

result = paralled_chain.invoke({
    'topic' : 'Artificial Intelligence'
})

print(result)






