## Docker process sample
<br>
- I built an masked token generation function using BERT-uncased; then I used FastAPI and Uvicorn to load to a local portal
<br>
- Then I used the dockerfile to create an image of the function based on a huggingface image and pushed to a container - it passed the basic test using the test.py
<br>
- The container is called “” on my docker hub. After pulling, you should be able to see the function if you use “tl5971/bert:latest”
