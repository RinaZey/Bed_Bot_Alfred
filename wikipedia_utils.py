import wikipedia

wikipedia.set_lang("ru")

def get_wiki_articles(query, max_articles=3):
    try:
        results = wikipedia.search(query, results=max_articles)
        articles = []
        for r in results:
            try:
                page = wikipedia.page(r)
                articles.append({'title': page.title, 'summary': page.summary[:300], 'url': page.url})
            except Exception:
                continue
        return articles
    except Exception:
        return []
