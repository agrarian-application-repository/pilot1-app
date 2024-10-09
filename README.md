# AGRARIAN

## Setup
1.  Copy the project with
    ```
    git clone https://github.com/simonesarti/AGRARIAN.git
    ```
2. Enter the project with
    ```
    cd AGRARIAN
    ```
3. Rename file `.env_placeholder` to `.env` with
    ```
    mv .env_placeholder .env
    ```   
4. Fill the  ` env ` file with your personal tokens and paths

   
5. Build the Singularity Image with
    ```
    bash singularity_build.sh
    ``` 
   
6. Run experiments with
    ```
    bash singularity_run.sh <script>.py [--arg1 arg1v ... --argN argNv]
    ```  