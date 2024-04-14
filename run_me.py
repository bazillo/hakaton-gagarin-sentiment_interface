import json
import pathlib
import typing as tp

import requests

import final_solution

# import final_solution.solution_stupid


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def download_file_from_google_drive(url):
    # Making request to the download URL
    response = requests.get(url)

    # Save the content with name
    with open('./final_solution/absa/rubert_tiny_2fc', 'wb') as file:  # Ensure the correct filename extension
        file.write(response.content)

def main():
    texts = load_data()
    scores = final_solution.solution.score_texts(texts)
    save_data(scores)


if __name__ == '__main__':
    main()
