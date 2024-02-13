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
# Define the path to your CV file
# cv_file_path = "path/to/your/CV.txt"

# # Read the content of the CV file
# with open(cv_file_path, 'r', encoding='utf-8') as file:
#     cv_text = file.read()

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader("data/CV.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  text = loader.load()
  # print(text)
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
Follow the instructions in steps, ensuring correctness at each stage.
Identify words, synonyms, or forms from the lists {lists.forest_entities} and {lists.redundant_rude_words}. Count them and provide two JSON-formatted lists of the top 5 words each.

Perform the analysis here using the provided {text} variable. Keep "forest_entities" and "redundant_rude_words" separate in the results. Only list the top 10 occurrences with the word as the JSON key and its count as the JSON value.

Provide a JSON with key stats about the text in the following format:
```json
{format}
Additionally, test the output to ensure it does not contain the strings "count" or "word," but actual keys and values.

Finally, offer a brief suggestion on improving the text:

If the count of the word "GDPR" is less than 1, suggest: "Even if you are not disclosing personal information, consider mentioning GDPR."
If the GDPR count is 1 or more, acknowledge: "Thanks for promoting GDPR."
"""

result = chain({"question": prompt, "chat_history": chat_history})
print(result['answer'])


chat_history.append((query, result['answer']))

