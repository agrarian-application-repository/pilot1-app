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
3. Create a copy of file `.env_placeholder` called `.env` with
    ```
    cp .env_placeholder .env
    ```   
4. Fill the  `.env ` file with your personal tokens


## Developing with virtual environments (conda)
5. Move to the `dev` folder with
    ```
    cd dev
   ``` 
   
6. Create the conda environment with
    ```
    bash create_conda_env.sh
    ``` 

7. Run experiments with:
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
