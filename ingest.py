from pypdf import PdfReader

def load_pages(pdf_path):
    reader= PdfReader(pdf_path)
    pages=[]
    for page_number , page in enumerate(reader.pages,start=1):
        text= page.extract_text() or ""
        if text.strip()== "":
            continue
        pages.append((page_number,text))
    return pages
def join_pages(pages):
    full_text="" 
    page_starts=[]
    for page_number , text in pages:
        page_starts.append((len(full_text),page_number))
        full_text+= text + "\n"
    return full_text, page_starts
def find_page(position,page_starts):
    page=page_starts[0][1]
    for start,page_number in page_starts:
        if start<=position:
            page=page_number
        else:
            break
    return page
def chunk_with_page(full_text, page_starts,size=1500,overlap=500):
    chunks=[]
    start=0
    chunk_id=0
    while start<=len(full_text):
        end=start+size
        chunks.append({
            "id":chunk_id,
            "page": find_page(start,page_starts),
            "end_page": find_page(min(end, len(full_text)) - 1, page_starts),
            "text": full_text[start:end]
        })
        if end>=len(full_text):
            break
        start= end-overlap
        chunk_id+=1
    return chunks

if __name__ == "__main__":
    pages = load_pages("data/tesla.pdf")
    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    print("Total chunks:", len(chunks))
    for c in chunks[:4]:
        print(c["id"], "page", c["page"], "|", c["text"][:60].replace("\n", " "))
    middle = chunks[200]
    print("\nChunk 200 is on page", middle["page"])
    print(middle["text"][:300])