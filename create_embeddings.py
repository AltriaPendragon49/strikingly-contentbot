import argparse
import pickle
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding website content')
    parser.add_argument('-m', '--mode', type=str, choices=['sitemap', 'zendesk'], default='sitemap',
                        help='Mode for data extraction: sitemap or zendesk')
    parser.add_argument('-s', '--sitemap', type=str, required=False,
                        help='URL to your sitemap.xml', default='https://www.paepper.com/sitemap.xml')
    parser.add_argument('-f', '--filter', type=str, required=False,
                        help='Text which needs to be included in all URLs which should be considered',
                        default='https://www.paepper.com/blog/posts')
    parser.add_argument('-z', '--zendesk', type=str, required=False,
                        help='URL to your zendesk api')
    args = parser.parse_args()

    if args.mode == 'sitemap':
        r = requests.get(args.sitemap)
        xml = r.text
        raw = xmltodict.parse(xml)

        pages = []
        for info in raw['urlset']['url']:
            # info example: {'loc': 'https://www.paepper.com/...', 'lastmod': '2021-12-28'}
            url = info['loc']
            if args.filter in url:
                pages.append({'text': extract_text_from(url), 'source': url})
    else:  # args.mode == 'zendesk'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        all_articles = []
        
        nextpage = args.zendesk#分页
        while nextpage:
            r = requests.get(nextpage, headers=headers)
            data = r.json()
            articles = data.get('articles', [])
            all_articles.extend(articles)
            nextpage = data.get('next_page')
        
        def clean_html(html):
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            text = '\n'.join(p.get_text() for p in soup.find_all('p'))
            return text.strip() or "No text content found"

        pages = [{
            "text": clean_html(article['body']),
            "source": article['html_url']
        } for article in all_articles]

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
