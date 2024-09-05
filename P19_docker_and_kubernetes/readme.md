## Docker process sample

<br>

* I built an masked token generation function using BERT-uncased; then I used FastAPI and Uvicorn to load to a local portal
* Then I used the dockerfile to create an image of the function based on a huggingface image and pushed to a container - it passed a simple test in test.py
* The container is called “tl5971/bert:latest” on my docker hub. After pulling, you should be able to see the function if you use "localhost:8080/docs" in your browser

<br>

## Kubernetes process sample
* I provided a yaml template of a replicaset for the same docker container, which can be used for scaled up production
* Also I included study notes on Kubernetes, including basics (pods, deployments, RS and RC, services) and cloud adaptations, see [here](https://docs.google.com/document/d/1PUr68kKitIOHTpxrclEiyfXixcb6UkvMbwZH4hF5W7E/edit?usp=sharing)