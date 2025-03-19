# Snowflake AI Contract Intelligence

This Repository provides a document processing interface for contracts and legal documents

## Setup
All you need to do is to execute the following SQL statements in your Snowflake Account.


```sql
USE ROLE DEMO_ROLE;

-- Create a warehouse
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH WITH WAREHOUSE_SIZE='MEDIUM';

-- Create Database
CREATE DATABASE IF NOT EXISTS DOC_EXTRACTION_DB;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- Create the API integration with Github
CREATE OR REPLACE API INTEGRATION GITHUB_INTEGRATION_CONTRACT_INTELLIGENCE
    api_provider = git_https_api
    api_allowed_prefixes = ('https://github.com/doneyli/')
    enabled = true
    comment='Git integration with Doneyli De Jesus Github Repository.';

-- Create the integration with the Github demo repository
CREATE OR REPLACE GIT REPOSITORY GITHUB_REPO_CORTEX_CONTRACT_INTELLIGENCE
	ORIGIN = 'https://github.com/doneyli/snowflake-ai-contract-intelligence' 
	API_INTEGRATION = 'GITHUB_INTEGRATION_CONTRACT_PROCESSING' 
	COMMENT = 'Github Repository from Doneyli De Jesus to interact with contracts and legal documents in natural language';

-- Run the installation of the Streamlit App
EXECUTE IMMEDIATE FROM @DOC_EXTRACTION_DB.PUBLIC.GITHUB_REPO_CORTEX_CONTRACT_INTELLIGENCE/branches/main/setup.sql;


-- Run this script to set up the required Snowflake objects before deploying the Streamlit app
USE DATABASE DOC_EXTRACTION_DB;

-- Create stage for storing PDFs
CREATE OR REPLACE STAGE pdf_contracts_stage 
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') 
    DIRECTORY = (ENABLE = TRUE);

-- Create table for tracking client uploads
CREATE TABLE IF NOT EXISTS client_contract_uploads (
    upload_id VARCHAR(36) PRIMARY KEY,
    client_name VARCHAR(255),
    upload_timestamp TIMESTAMP_NTZ,
    num_files INTEGER,
    processed BOOLEAN DEFAULT FALSE
);

-- Create table for individual PDF files
CREATE TABLE IF NOT EXISTS contract_pdf_files (
    file_id VARCHAR(36) PRIMARY KEY,
    upload_id VARCHAR(36),
    original_filename VARCHAR(255),
    stored_filename VARCHAR(255),
    file_path VARCHAR(500),
    upload_timestamp TIMESTAMP_NTZ,
    file_size INTEGER,
    num_pages INTEGER,
    extracted_metadata VARIANT,
    parsed_document VARIANT,
    FOREIGN KEY (upload_id) REFERENCES client_contract_uploads(upload_id)
);

-- Lease metadata table
CREATE TABLE IF NOT EXISTS LEASE_METADATA AS
SELECT 
    FILE_ID,
    ORIGINAL_FILENAME,
    STORED_FILENAME,
    SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', 
        CONCAT($$Extract the following information from the lease in JSON format as follows. Just provide the JSON: 
                [METADATA {
                'commencement_date' : 'value',
                'location': 'value', 
                'street': 'value', 
                'city': 'value',
                'province': 'value', 
                'landlord': 'value', 
                'tenant': 'value' } ] $$, 
            PARSED_DOCUMENT ) ) METADATA
FROM 
    CONTRACT_PDF_FILES;


-- creating chunks and append metadata
CREATE TABLE IF NOT EXISTS LEASE_CHUNKS AS 
SELECT
    FILE_ID,
    ORIGINAL_FILENAME,
    STORED_FILENAME,
    PARSED_DOCUMENT,
    c.VALUE || (SELECT METADATA FROM LEASE_METADATA) AS chunk
FROM
    CONTRACT_PDF_FILES,
    LATERAL FLATTEN(
        SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER (
            parsed_document,
            'markdown',
            1200,
            120 ) ) c;

-- CREATE CORTEX SEARCH SERVICE
CREATE OR REPLACE CORTEX SEARCH SERVICE CORTEX_SEARCH_LEASES 
ON CHUNK 
ATTRIBUTES ORIGINAL_FILENAME, STORED_FILENAME
WAREHOUSE = DEMO_WH 
TARGET_LAG = '1 minute' 
AS ( SELECT * FROM LEASE_CHUNKS );
```



