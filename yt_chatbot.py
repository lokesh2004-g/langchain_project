###########################yt_chatbot#######################################################
#library
import vectorstore
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

###################api key###################################################### 

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your token id"

##########################adding yt video transcript by using api #######################

str_transcript=""
transcript = YouTubeTranscriptApi().fetch(video_id="h0e2HAPTGF4")
str_transcript = " ".join(snippet.text for snippet in transcript)

###################################breaking in small parts#############################

spliter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks=spliter.create_documents([str_transcript])

##########################making Embeddings and storing them #############################
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
########################storing vector#########################################
#store to your disk
#vector_store=FAISS.from_documents(chunks,hf)
#vector_store.save_local("my_faiss_index")
vector_store = FAISS.load_local(
    "my_faiss_index",
    hf,
    allow_dangerous_deserialization=True
)
#print(vector_store.get_by_ids(['9bd2ba58-e5f3-47cd-816a-c90bdb6aac67']))
########################################retriever#####################################

retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

###################################now using llm by api#################################

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    max_new_tokens=10,
    task="text-generation",
    provider="hyperbolic",  # set your provider here
    # provider="nebius",
    # provider="together",
)
##################################prompt templete #################################
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question ="tell me about the clustering"
retriever_docs=retriever.invoke(question)
context= "\n\n".join(d.page_content for d in retriever_docs)
final_prompt=prompt.invoke({"context": context,"question": question})
##################################getting result#####################################
model = ChatHuggingFace(llm=llm)
result = model.invoke(final_prompt)
print(result.content)
