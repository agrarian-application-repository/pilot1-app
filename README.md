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
3. Create a copy of file `.env_placeholder` called `.env` with
    ```
    cp .env_placeholder .env
    ```   
4. Fill the  `.env ` file with your personal tokens


## Working with virtual environments (conda)
5. Create the conda environment with
    ```
    bash create_conda_env.sh
    ``` 

6. Run experiments with:
    ```
   bash run_script.sh <script.py> [--arg1 arg1val ... --argN argNval]
    ```  


## Working with Singularity Containers
5. Build the Singularity Image with
    ```
    bash singularity_build.sh <absolute/path/to/project>
    ``` 
   
6. Run experiments interactively with
    ```
    bash singularity_run.sh <script>.py [--arg1 arg1val ... --argN argNval]
    ```  
   
7. Run experiments through the PBS scheduler with
    ```
    qsub -v 'ARGS="<script.py> [--arg1 arg1val ... --argN argNval]"' singularity_run.sh
    ```  

