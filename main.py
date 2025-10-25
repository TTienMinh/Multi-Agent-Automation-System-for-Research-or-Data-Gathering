import arxiv
import json
from datetime import datetime

def arxiv_abstracts_sample(num_results=10, query="LLM"):
    """
    Sample code to fetch and print abstracts from ArXiv using the arxiv library.
    """
    
    # Initialize the ArXiv client and perform the search
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=num_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending # newest first
    )
    
    all_results = list(client.results(search))
    if not all_results:
        print("No results found.")
        return

    print(f"✅ Found and retrieved {len(all_results)} paper(s).")
    
    output_lines = []
    for i, paper in enumerate(all_results):
        # Take the paper main attributes
        abstract_clean = paper.summary.replace('\n', ' ')

        entry = (
            f"Paper {i+1}/{len(all_results)}:\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join([a.name for a in paper.authors])}\n"
            f"ID: {paper.entry_id}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"Abstract:\n{abstract_clean}\n"
            "--------------------------------------------------\n"
        )
        
        output_lines.append(entry)
        
        # Print a concise version to the console
        print(f"{i+1}. **{paper.title}**")
        print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
        print(f"   Abstract: {abstract_clean[:200]}... [Full abstract saved to file]\n")
    
    filename = "arxiv_abstracts.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"💾 Successfully stored all data in **{filename}**")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
    
if __name__ == "__main__":
    arxiv_abstracts_sample(num_results=10, query="RAG AND LLM")