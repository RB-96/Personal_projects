# Chart parser repository
This repository detect the chart type and extract the data from the chart. You can find the entire flow diagram [here](https://73strings.atlassian.net/wiki/spaces/AT1/pages/431980555/Chart+Parser?atlOrigin=eyJpIjoiM2Q4ZDY1NzRmZWU0NGMxNTg3MTg4OGM1NWIxZjBkMDkiLCJwIjoiYyJ9).

# How to setup the repository
## Local
1. Clone the repository
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Start the server
    ```bash
    python server.py
    ```
4. Now you can access the [server](http://0.0.0.0:5900/docs)

## Docker
1. Clone the repository
2. Build the docker image
    ```bash
    docker build -t chart-parser .
    ```
3. Run the docker image
    ```bash
    docker run -p 5900:5900 --env-file .env chart-parser
    ```
4. Now you can access the [server](http://0.0.0.0:5900/docs)