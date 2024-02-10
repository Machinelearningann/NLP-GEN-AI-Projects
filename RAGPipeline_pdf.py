import os
import tiktoken
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter,SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context


current_dir = os.getcwd()
print(current_dir)
os.environ['OPENAI_API_KEY'] = "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"
data = os.path.join(current_dir, 'data')
documents = SimpleDirectoryReader(input_dir=data).load_data()

text_splitter = TokenTextSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"],
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode

)

# node_parser = SimpleNodeParser(6
#     sentence_splitter=text_splitter
# )

node_parser = SimpleNodeParser.from_defaults(
  text_splitter=text_splitter
)

node_parser = SimpleNodeParser(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    chunk_size=1024,
    chunk_overlap=20,
    paragraph_separator ='\n\n\n',
    separator=' '
)

text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  paragraph_separator="\n\n\n",
  secondary_chunking_regex="[^,.;。]+[,.;。]?",
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode

)

llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)

embed_model = OpenAIEmbedding()

prompt_helper = PromptHelper(

  context_window=4096, 

  num_output=256, 

  chunk_overlap_ratio=0.1, 

  chunk_size_limit=None

)

service_context = ServiceContext.from_defaults(

  llm=llm,

  embed_model=embed_model,

  node_parser=node_parser,

  prompt_helper=prompt_helper

)

index = VectorStoreIndex.from_documents(
    documents, 
    service_context = service_context
    )

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What Interactions with other microservices include")
print(response)
