import os
import requests
from bs4 import BeautifulSoup

URLS = [
    "https://vnu.edu.vn/gioi-thieu/tong-quan/lich-su",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/su-mang-tam-nhin",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/chien-luoc-phat-trien",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/thi-dua-khen-thuong",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/so-lieu-thong-ke"
]

OUTPUT_DIR = "data/raw"

def clean_text(text):
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def crawl_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for url in URLS:
        try:
            print(f"Crawling: {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # The content of VNU sites is usually inside div with class 'post-content' or 'main-content'
            content_div = soup.find('div', class_='post-content')
            if not content_div:
                content_div = soup.find('div', class_='main-content')
                
            if not content_div:
                # Fallback to body
                content_div = soup.body
                
            # Remove scripts and styles
            for script in content_div(["script", "style"]):
                script.extract()
                
            text = content_div.get_text(separator='\n')
            clean_content = clean_text(text)
            
            # Save to file
            filename = url.strip('/').split('/')[-1] + '.txt'
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_content)
                
            print(f"Saved {filepath}")
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")

if __name__ == "__main__":
    crawl_and_save()
