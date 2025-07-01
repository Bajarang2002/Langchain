from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")


class Feedback(BaseModel):

    sentiment: Literal['Positive', 'Negative'] = Field(..., description="Sentiment is either positive or negative")


parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(template="Provide the sentiment from the following feedback text either positive or negative \n {feedback}/n {format_instruction}",
                         input_variables=['Feedback'],
                         partial_variables={"format_instruction":parser.get_format_instructions()})


classifier_chain = prompt1|model|parser


prompt2 = PromptTemplate(template="Geneate appropriate response on the this positive feedback \n {feedback}",
                         input_variables=['feedback'])
prompt3 = PromptTemplate(template="Generate the appropriate response on  negative feedback \n {feedback}",
                         input_variables=['feedback'])


branch_chain = RunnableBranch(
    (lambda x : x.sentiment =='Positive',prompt2|model|parser),
    (lambda x : x.sentiment == 'Negative',prompt3|model|parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain|branch_chain


result = chain.invoke({"feedback":" This is very wonderfull place"})
print(result)

chain.get_graph().print_ascii()
