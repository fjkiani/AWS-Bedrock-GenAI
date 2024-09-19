import logging
import pandas as pd
from pycelonis import get_celonis

# Define your constants
BASE_URL = 'https://51cdv7o2-2024-07-16.training.celonis.cloud'
API_TOKEN = 'MTNhYmIxNmUtN2Y5MC00MjhjLTliYmQtZDE3ZjY0NzNlNjRlOkFqZXpkdHYrQ0Q5MFlFdk82bFFxYUlVRU9ub2NCVGs3U1AyckxXQUZKdWFw'
DATA_POOL_NAME = 'test1'
TABLE_NAME = 'TestTable'

# Function to create dummy data
def create_dummy_data(file_path):
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'value': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

# Function to get or create the data pool
def get_or_create_data_pool(celonis, data_pool_name):
    data_pool = celonis.data_integration.get_data_pools().find(data_pool_name)
    if data_pool:
        logging.info(f"Found existing data pool: {data_pool_name}")
        return data_pool
    else:
        logging.info(f"Creating new data pool: {data_pool_name}")
        return celonis.data_integration.create_data_pool(name=data_pool_name)

# Function to push data to Celonis
def push_data_to_celonis(celonis, data_pool_name, table_name, df):
    try:
        pool = get_or_create_data_pool(celonis, data_pool_name)

        if not pool:
            logging.error(f"Unable to get or create data pool '{data_pool_name}'.")
            return

        # logging.info(f"Table schema for {table_name}:")
        # for column, dtype in data.dtypes.items():
        #     logging.info(f"- {column}: {dtype}")

        # Check if table exists, if not create it
        table = pool.get_tables().find(table_name)
        logging.info(f"tables: {table}")
        if table:
            # Push data to the table
            table.upsert(
                df,
                keys=['id']
            )
            logging.info(f"Data successfully pushed to {data_pool_name}.{table_name}")

        else:
            logging.info(f"Table '{table_name}' not found. Creating new table.")
            # column_config = {
            #     col: {"type": str(dtype), "length": 255 if dtype == 'object' else None}
            #     for col, dtype in data.dtypes.items()
            # }
            data_pool_table = pool.create_table(df, table_name)
            data_pool_table.append(df)
            
        
    except Exception as e:
        logging.error(f"Error pushing data: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

# Main function
def main():
    # Create dummy data
    file_path = 'dummy_data.csv'
    df = pd.read_csv(file_path)
    
    # Initialize Celonis connection
    celonis = get_celonis(
        base_url=BASE_URL,
        api_token=API_TOKEN
    )
    
    # Push data to Celonis
    push_data_to_celonis(celonis, DATA_POOL_NAME, TABLE_NAME, df)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()