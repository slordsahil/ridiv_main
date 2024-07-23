from fastapi import FastAPI,File,UploadFile
import uvicorn
from pydantic import BaseModel

from typing import Annotated
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os 
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key='0b10c602-9637-428b-94d7-7cbd46104269')
embeddings=OpenAIEmbeddings(api_key=os.environ['api_key'],model="text-embedding-3-large")


my_app=FastAPI()

class QuestionRequest(BaseModel):
    question: str

file_chunks=[]

@my_app.post("/upload_file/")
async def create_file(file: Annotated[UploadFile, File()]):
    global file_chunks

    file_chunks_em=[]
    
    while True:
        data_chunk=await file.read(1000)
        if not data_chunk:
            break
        
        data_chunk_str = data_chunk.decode('utf-8')  
        file_chunks_em.append(embeddings.embed_query(data_chunk_str))
        file_chunks.append(data_chunk_str)
        
    id = [str(i) for i in range(1, len(file_chunks) + 1)]

   

    index_name='sahil'
    index = pc.Index("sahil")
    pc.describe_index(name=index_name)
    
    index.upsert(vectors=zip(id,file_chunks_em))
    
    return {"successfully uploaded"}

@my_app.post("/ask_me/")
async def create_file(request: QuestionRequest):
    global file_chunks
    index=pc.Index('sahil')
    matching_ans=index.query(vector=embeddings.embed_query(request.question),top_k=3,include_values=False)
    matching_ids = [match['id'] for match in matching_ans['matches']]
    similarity=list(map(lambda x:file_chunks[int(x)],matching_ids))
    prompt=PromptTemplate(input_variables=['query','history_data'],template="Write an response for  {query} where chat references are {history_data}")


    llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.9,max_tokens=225,api_key=os.environ["api_key"])
    gen=LLMChain(prompt=prompt,llm=llm)
    final_response=gen.run({"query": request.question, "history_data": similarity})
    
    return {"response":final_response,"similarity":similarity}


    
@my_app.post("/delete_my_pdf_data/")
async def delete_data():
    try:
        index_name='sahil'
        index = pc.Index("sahil")
        pc.describe_index(name=index_name)
        index.delete(delete_all=True)
        return {"status": "All data deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
    