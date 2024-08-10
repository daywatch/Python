## Docker process sample

<br>

* I built an masked token generation function using BERT-uncased; then I used FastAPI and Uvicorn to load to a local portal
* Then I used the dockerfile to create an image of the function based on a huggingface image and pushed to a container - it passed a simple test in test.py
* The container is called “tl5971/bert:latest” on my docker hub. After pulling, you should be able to see the function if you use "localhost:8080/docs" in your browser

<br>

## Kubernetes process sample (coming soon)