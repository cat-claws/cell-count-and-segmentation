import re
import os
import csv
import json
import requests

idRegex = re.compile(r'.*file/d/|/view.*|.*id=')

# All codes below are taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
	data = {'consep.zip':'https://drive.google.com/file/d/1CfeaVpKcG2EcRtLA4JZF2kiF3TgmPDZf/view?usp=sharing',
		'train.pkl':'https://drive.google.com/file/d/1yKWqNYAB_Ba1uij6KxaK5dDB0z2b5566/view?usp=sharing',
		'valid.pkl':'https://drive.google.com/file/d/1-15x3lpn8BjDgvVIH5QqPHJzGFQvdJij/view?usp=sharing'}
	
	for k in data:
		download_file_from_google_drive(idRegex.sub('', data[k]), k)
