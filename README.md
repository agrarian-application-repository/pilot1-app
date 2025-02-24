![AGRARIAN](assets/agrarian.png)

# AGRARIAN: Herd Monitoring

## Setup
1.  Copy the project with
    ```
    git clone https://github.com/simonesarti/AGRARIAN.git
    ```
2. Enter the project with
    ```
    cd AGRARIAN
    ```
3. Create a file `.env` with the following content:
    ```
    DOCKER_USERNAME="$oauthtoken"
    DOCKER_PASSWORD="<your-ncvr-io-token>"
    ```


## Developing with virtual environments (conda)
4. Move to the `dev` folder with
    ```
    cd dev
   ``` 
   
5. Create the conda environment with
    ```
    bash create_conda_env.sh
    ``` 

6. Run experiments with:
    ```
   bash run_script.sh <script.py> [--arg1 arg1val ... --argN argNval]
    ```  

# Docker

## Health monitoring
1. Enter the project with
    ```
    cd AGRARIAN
    ```
   
2. Build the Docker image with
    ```bash
   docker build -t health_monitoring -f docker/health_monitoring/Dockerfile .
   ```
   
3. Verify the docker image was built succesfully
   ```
   docker images | grep health_monitoring
   ```

4. Run the containerized application with
    ```
   docker run --rm \
   -e <http://ip:port/video>
   -v </your/path/to/config_file.yaml>:/app/config.yaml \
   health_monitoring config.yaml
   ```
