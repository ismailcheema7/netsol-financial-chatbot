import os
import time
from typing import Sequence, Annotated, TypedDict
from operator import add as add_messages
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import pymongo
from pymongo.errors import AutoReconnect
from pymongo.operations import SearchIndexModel
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongo_uri)
collection = client["Netsol-Chatbot"]["Financial-Report"]


tavily_api_key = os.getenv("TAVILY_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Gemini 1.5 or Flash via LangChain wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.25,
    convert_system_message_to_human=True,
    verbose=True
)

"""2. Loading the Data as a PDF"""

if collection.count_documents({}) == 0:
    print("📥 No documents found. Generating embeddings and storing data...")
    path  = 'NetSol_Financial Statement_2024_Part 1.pdf'

    def pdf_exists(path):
        return os.path.exists(path) and path.endswith(".pdf")

    keywords = [
        "revenue", "income", "profit", "gross margin", "net income", "operating margin",
        "earnings", "eps", "ebitda", "cash flow", "expenses", "liabilities", "assets",
        "net profit", "total revenue", "cost of sales", "net earnings", "profitability",
    ]

    # Returning only the pages where we have some useful data 
    def is_useful(text):
        return (
            len(text.strip()) > 100 and
            any(word in text.lower() for word in keywords)
        )   

    # Try primary loader
    def try_load_pdf(path):
        if not pdf_exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")

        try:
            print("🔍 Trying PyPDFLoader...")
            loader = PyPDFLoader(path)
            return loader.load()     
        except Exception as e1:
            print(f"⚠️ PyPDFLoader failed: {e1}")
            try:
                print("🛠 Trying UnstructuredPDFLoader as fallback...")
                loader = UnstructuredPDFLoader(path)
                return loader.load()
            except Exception as e2:
                print(f"❌ Both loaders failed: {e2}")
                return []
            
    # Load PDF
    loaded_data = try_load_pdf(path) 

    # returns the entire text document containing the useful files 
    useful_pages = [p for p in loaded_data if is_useful(p.page_content)]
    print(f"Found {len(useful_pages)} useful pages")
    if useful_pages:
            print(f"Sample page content: {useful_pages[0].page_content[:200]}...")

    """3. Split the PDF data into chunks using RecursiveTextSplitter"""

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = recursive_splitter.split_documents(useful_pages)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---\n", chunk.page_content[:300])

    """4. Load huggingface embedding model and 5. Convert chunks to embeddings"""

    embedding_model_st = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', trust_remote_code=True)

    def get_embedding(data, precision="float32"):
        return embedding_model_st.encode(data, precision=precision).tolist()

    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk.page_content)
        embeddings.append(embedding)

    print(f"Number of embeddings: {len(embeddings)}")
    if embeddings:
        print(f"First embedding sample: {embeddings[0][:5]}...")  # Show first 5 values

    """6. Convert embeddings into a document format"""

    def create_embedding_docs(embeddings, data):
        docs = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, data)):    #each embedding matches to data according to index
            doc = {
            "text": chunk.page_content,
            "embedding": embedding
            }                                             #the content of each chunk is saved with its embedding 
            docs.append(doc)
        return docs    #-> list of dictionaries

    docs = create_embedding_docs(embeddings, chunks)

    if embeddings:
        print(f"Embedding dimension: {len(embeddings[0])}")  #check the dimensions

    collection.delete_many({})  #all data previously stored in mongodb is deleted 

    for attempt in range(3):
        try:
            collection.insert_many(docs, ordered=False)
            print(f"Successfully inserted {len(docs)} documents")
            break
        except AutoReconnect:
            print(f"Connection lost on attempt {attempt + 1}. Retrying in 3 seconds...")
            time.sleep(3)
        except Exception as e:
            print(f"Error inserting documents: {e}")
            break

    """8. Prepare Vector Search Index Model"""

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
            {
            "type": "vector",
            "path": "embedding",
            "numDimensions": len(embeddings[0]),
            "similarity": "dotProduct",
            "quantization": "scalar"
            }]
        },
        name="vector_index",
        type="vectorSearch"
    )

    index_name = "vector_index"

    # Check if the index already exists
    existing_indexes = list(collection.list_search_indexes())
    existing_index_names = [i['name'] for i in existing_indexes]

    if index_name not in existing_index_names:
        try:
            result = collection.create_search_index(model=search_index_model)
            print(f"✅ Created search index: {result}")
        except Exception as e:
            print(f"⚠️ Error creating index: {e}")
    else:
        print(f"ℹ️ Search index '{index_name}' already exists.")
else:
    print("✅ MongoDB already populated. Skipping data ingestion.")

predicate = lambda index: index.get("queryable") is True
max_attempts = 3 # 5 minutes max
attempts = 0

while attempts < max_attempts:
      try:
            index_name = "vector_index"
            indices = list(collection.list_search_indexes(index_name))
            if len(indices) and predicate(indices[0]):
                print("Search index is ready!")
                break
      except Exception as e:
            print(f"Error checking index status: {e}")

      attempts += 1
      time.sleep(5)

if attempts >= max_attempts:
   print("Warning: Search index may not be ready yet")

"""10. Prepare Vector Store Database"""

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="vector_index",
    text_key="text",   #field which contains the text 
    embedding_key="embedding" #field which contains the embedding
)

"""11. Prepare Retreiver Model"""

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 most similar documents
)

tavily = TavilyClient(api_key=tavily_api_key)

@tool
def retriever_tool(query: str) -> str:
  """
  This tool searches and returns information from the Netsol Annual Financial Report 2025.
  """

  retrieved_docs = retriever.invoke(query)

  if not retrieved_docs:
    return "I found no relevant information in the Netsol Annual Financial Report Document"

  results = []
  for i, doc in enumerate(retrieved_docs):
    results.append(f"Document {i+1}:\n{doc.page_content}")

  return "\n\n".join(results)

#This only returns the formateed documented chunks as a single string 

@tool
def tavily_tool(query: str) -> str:
    """
    This tool performs a web search to provide real-time information.
    This is only run if we have not gotten a proper response from either the LLM or the retriever tool
    """
    try:
        results = tavily.search(query)

        if not results or 'results' not in results:
            return "No relevant web results found."

        output = []
        for i, res in enumerate(results['results'][:3]):  # Limit to top 3 results
            output.append(f"Result {i+1}:\nTitle: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('content', 'N/A')}")

        return "\n\n".join(output)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

tools = [retriever_tool, tavily_tool]

llm =  llm.bind_tools(tools)

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]


# Returns True if the latest message from AI used tool calls
def should_continue(state:AgentState):
  """Check if the last message contains tool calls"""
  result = state['messages'][-1]  #the last message in the result from the AI 
  return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
If the question is related to Netsol, Use the retriever tool to give a detailed reponse based on the 2024 NetSol Financial Report. Write it in a normal paragraph like format. Not too long

If the user question is not related to Netsol, DO NOT USE THE RETRIEVER TOOL. Use the Tavily tool to give the asnwer and be very funny and creative in this case

If the user requests for a one word answer such as a specific value (total revenue or liquidity or main income stream), give only a ONE WORD ANSWER. Do not start telling multiple line stories

You are provided full chat history. Use it to remember what the user said earlier.

Always cite your sources: document snippets (for retriever) or URLs (for Tavily).
"""

def build_summary(messages):
    summary = []
    for m in messages:
        if isinstance(m, HumanMessage):
            summary.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            summary.append(f"Bot: {m.content}")
    return "\n".join(summary)


tools_dict = {tool.name: tool for tool in tools}   #dictionary of our tools
#{'retriever_tool': <retriever_tool function>,'tavily_tool': <tavily_tool function>}

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages']) 

    last_user_message = messages[-1].content.lower()
    if "last question" in last_user_message or "history" in last_user_message:
        summary = build_summary(messages[:-1])
        messages.insert(1, HumanMessage(content=f"Here's your past history:\n{summary}"))

    messages = [SystemMessage(content=system_prompt)] + messages #initial System Prompt message added to history of messages from state 

    print("\n🤖 Gemini sees this full history:")
    for m in messages:
        print(f"- {m.type}: {m.content}")

    message = llm.invoke(messages)  #message history sent to LLM 
    return {'messages': (state['messages'] + [message])}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls   # Get last LLM message's tool calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        # Makes sure the LLM is calling a valid tool 
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            #Run the tool Gemini requested, and pass it the query argument Gemini gave.
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        
        # Appends the Tool Message to the results for the LLM 
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue, #if the last llm call had tool calls, continue, otherwise end the fucntion, and wait for the next query 
    {True: "_agent", False: END}
)
graph.add_edge("_agent", "llm")  
graph.set_entry_point("llm")

rag_agent = graph.compile()
