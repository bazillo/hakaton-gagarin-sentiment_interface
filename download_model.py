from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    path = hf_hub_download(repo_id="ganjubas2008/absa-rubert-tiny", filename="rubert_tiny_2fc", local_dir='./final_solution/absa')
    print(path)

