import argparse
import faiss
import os
import pickle

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

parser = argparse.ArgumentParser(description='Paepper.com Q&A')
parser.add_argument('question', type=str, help='Your question for Paepper.com')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

def get_qa_chain(vectorstore):
    # 稍微调整提示语格式
    question_template = """
    根据下面内容回答问题，无答案时如实告知，不允许编造：
    {context}
    
    问题：{question}
    答案：
    """
    qa_prompt = PromptTemplate(#回答模板
        template=question_template,
        input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={'k':4}),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        ),
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        return_source_documents=True
    )

if __name__ == "__main__":
    qa_chain = get_qa_chain(store)
    response = qa_chain({
        'question': args.question,
        'chat_history': []
    })

    source_urls = []
    for doc in response['source_documents']:
        url = doc.metadata['source']
        if url not in source_urls:
            source_urls.append(url)
    
    print(f"回答：{response['answer']}\n来源：{'&'.join(source_urls)}")
