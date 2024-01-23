import os
import sys
import lists
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import constants
import json
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader("data/report.txt") # Use this line if you only need data.txt
  # loader = DirectoryLoader("data/")
  data = loader.load()
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
jsonfile = open('./format.json','r')
format = json.load(jsonfile)
prompt = f"""
Find  any words or thir synonyms or forms from the two lists in the text ib the data folder
, count them and provide two json formatted lists of count for top 5 words.
just do it here. Do not mix list 1 and list 2 provide two separate jsons. Find words from both lists in text, do not provide code, answer in folowing format ```{format}```, only list top 10 occurences with word presented as json key and count of its occurences as json value:


give me json with key stats about the text in this format with properties like:
length, readability(estimate on scale 1-10), count of references to "RACI", "environment","TBC","out of scope","Risk","Distribution", "unit test","scrum","agile","Performance","UAT"

```{data}```,```{lists.list1}```,```{lists.list2}```

please test the output that it does not contain string "count" or "word" but has actual keys and values in it.

# give a very short suggestion on how to improve the text including specific instructions:
# If count of word RACI < 1 Then say "I didn't find a clear indication of who is responsible. Consider adding a responsibility matrix such as RACI (responsible, accountable, consulted, informed) to make it clearer who has to do what"
"""

result = chain({"question": prompt, "chat_history": chat_history})
print(result['answer'])

chat_history.append((query, result['answer']))

