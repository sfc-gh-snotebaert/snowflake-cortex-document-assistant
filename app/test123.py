import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.core import Root  # requires snowflake>=0.8.0
from snowflake.cortex import Complete
import os
import tempfile
import uuid
import hashlib
from datetime import datetime
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Contract PDF Processor",
    page_icon="📄",
    layout="wide"
)

# Get the active Snowflake session that's already established
session = get_active_session()
root = Root(session)

# Models for chat
MODELS = [
    "claude-3-5-sonnet",
    "mistral-large2",
]

# Function to calculate MD5 hash of a file
def calculate_md5(file_content):
    """Calculate MD5 hash of a file's content."""
    md5_hash = hashlib.md5()
    md5_hash.update(file_content)
    return md5_hash.hexdigest()

# Function to check if a file exists based on MD5 hash
def check_file_exists(md5_hash):
    """
    Comprehensive check if a file with the given MD5 hash (as file_id) already exists and has been processed.
    Checks all relevant tables: contract_pdf_files (including parsed_document), LEASE_METADATA, and LEASE_CHUNKS.
    
    Returns:
        tuple: (exists, is_parsed, metadata_exists, chunks_exist, result_data)
          - exists: True if the file exists in contract_pdf_files
          - is_parsed: True if the file has a parsed_document in contract_pdf_files
          - metadata_exists: True if the file has an entry in LEASE_METADATA
          - chunks_exist: True if the file has chunks in LEASE_CHUNKS
          - result_data: Information about the existing file
    """
    try:
        # Check if file exists in the database and if it has been parsed
        result = session.sql(f"""
        SELECT cf.file_id, cf.original_filename, cf.upload_id, cu.client_name,
               CASE WHEN cf.parsed_document IS NOT NULL THEN 1 ELSE 0 END as is_parsed
        FROM contract_pdf_files cf
        JOIN client_contract_uploads cu ON cf.upload_id = cu.upload_id
        WHERE cf.file_id = '{md5_hash}'
        """).collect()
        
        exists = len(result) > 0
        
        if exists:
            is_parsed = result[0]['IS_PARSED'] == 1
            
            # Check if metadata exists
            metadata_result = session.sql(f"""
            SELECT COUNT(*) as metadata_count
            FROM LEASE_METADATA
            WHERE FILE_ID = '{md5_hash}'
            """).collect()
            
            metadata_exists = metadata_result[0]['METADATA_COUNT'] > 0
            
            # Check if chunks exist
            chunks_result = session.sql(f"""
            SELECT COUNT(*) as chunk_count
            FROM LEASE_CHUNKS
            WHERE FILE_ID = '{md5_hash}'
            """).collect()
            
            chunks_exist = chunks_result[0]['CHUNK_COUNT'] > 0
            
            return exists, is_parsed, metadata_exists, chunks_exist, result
        else:
            return False, False, False, False, None
    except Exception as e:
        st.error(f"Error checking file existence: {str(e)}")
        return False, False, False, False, None

# Initialize chat messages
def init_messages():
    """
    Initialize the session state for chat messages. If the "messages" key is not 
    in the session state, initialize it as an empty list.
    """
    # Initialize messages if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Function to clear chat history
def clear_chat_history():
    """Clear the chat message history"""
    st.session_state.messages = []

# Initialize service metadata for Cortex Search
def init_service_metadata():
    """
    Initialize the session state for cortex search service metadata. Query the available
    cortex search services from the Snowflake session and store their names and search
    columns in the session state.
    """
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )

        st.session_state.service_metadata = service_metadata

# Function to generate completions
def complete(model, prompt):
    """
    Generate a completion for the given prompt using the specified model.

    Args:
        model (str): The name of the model to use for completion.
        prompt (str): The prompt to generate a completion for.

    Returns:
        str: The generated completion.
    """
    return Complete(model, prompt).replace("$", "\$")

# Query Cortex Search service
def query_cortex_search_service(query, columns=[], filter={}):
    """
    Query the selected cortex search service with the given query and retrieve context documents.
    Display the retrieved context documents in the sidebar if debug mode is enabled. Return the
    context documents as a string.

    Args:
        query (str): The query to search the cortex search service with.

    Returns:
        str: The concatenated string of context documents.
        list: The results list from the search.
    """
    db, schema = session.get_current_database(), session.get_current_schema()

    cortex_search_service = (
        root.databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.selected_cortex_search_service]
    )

    if 'num_retrieved_chunks' not in st.session_state:
        st.session_state.num_retrieved_chunks = 10

    context_documents = cortex_search_service.search(
        query, 
        columns=columns, 
        limit=st.session_state.num_retrieved_chunks
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                 if s["name"] == st.session_state.selected_cortex_search_service][0].lower()

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"

    if 'debug' in st.session_state and st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)

    return context_str, results

# Get chat history
def get_chat_history():
    """
    Retrieve the chat history from the session state limited to the number of messages specified
    by the user in the sidebar options.

    Returns:
        list: The list of chat messages from the session state.
    """
    if 'num_chat_messages' not in st.session_state:
        st.session_state.num_chat_messages = 5
        
    start_index = max(
        0, len(st.session_state.messages) - st.session_state.num_chat_messages
    )
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]

# Create chat history summary for better context
def make_chat_history_summary(chat_history, question):
    """
    Generate a summary of the chat history combined with the current question to extend the query
    context. Use the language model to generate this summary.

    Args:
        chat_history (str): The chat history to include in the summary.
        question (str): The current user question to extend with the chat history.

    Returns:
        str: The generated summary of the chat history and question.
    """
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """

    summary = complete(st.session_state.model_name, prompt)

    if 'debug' in st.session_state and st.session_state.debug:
        st.sidebar.text_area(
            "Chat history summary", summary.replace("$", "\$"), height=150
        )

    return summary

# Create prompt for the chat with Cortex Search
def create_prompt(user_question, additional_context=""):
    """
    Create a prompt for the language model by combining the user question with context retrieved
    from the cortex search service and chat history (if enabled). Format the prompt according to
    the expected input format of the model.

    Args:
        user_question (str): The user's question to generate a prompt for.
        additional_context (str): Any additional context to add (e.g., from lease documents)

    Returns:
        str: The generated prompt for the language model.
        list: The search results
    """
    prompt_context = ""
    results = []
    
    if not hasattr(st.session_state, 'service_metadata') or not st.session_state.service_metadata:
        # Fall back to using only additional context if no search services
        prompt_context = additional_context
    else:
        # Use Cortex Search services
        if hasattr(st.session_state, 'use_chat_history') and st.session_state.use_chat_history:
            chat_history = get_chat_history()
            if chat_history:
                question_summary = make_chat_history_summary(chat_history, user_question)
                prompt_context, results = query_cortex_search_service(
                    question_summary,
                    columns=["chunk"],
                )
            else:
                prompt_context, results = query_cortex_search_service(
                    user_question,
                    columns=["chunk"],
                )
        else:
            prompt_context, results = query_cortex_search_service(
                user_question,
                columns=["chunk"],
            )
    
    # Add any additional context (e.g., from lease documents)
    if additional_context:
        if prompt_context:
            prompt_context += "\n\nAdditional context:\n" + additional_context
        else:
            prompt_context = additional_context

    # Create the base prompt
    prompt = f"""
        [INST]
        You are an expert legal professional that analyzes contracts for a living. 

        Review the following question and the provided relevant materials to answer the question in a legally accurate manner. 

        Please pay extra attention to the date of the legal details. Always prioritize newer information if there is a contradiction to past information.

        
        <context>
        {prompt_context}
        </context>
        
        <question>
        {user_question}
        </question>
        
        Please provide a helpful, and accurate response based on the context provided.
        If the answer cannot be determined from the context, just say "I don't have enough information to answer that question."
        [/INST]
        Answer:
    """
    
    return prompt, results

# Check if parsed_document column exists
def check_parsed_document_column():
    try:
        columns_check = session.sql("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = CURRENT_SCHEMA() 
            AND table_name = 'CONTRACT_PDF_FILES'
            AND column_name = 'PARSED_DOCUMENT'
        """).collect()
        
        return len(columns_check) > 0
    except Exception:
        return False

# Function to process files with true parallel processing
def process_files_with_ai_parallel(file_records, client_name, upload_id):
    if not file_records:
        return False
    
    try:
        # Get all the file paths first
        list_result = session.sql(f"""
        LIST @pdf_contracts_stage/{client_name}/{upload_id}/ PATTERN='.*'
        """).collect()
        
        # Create a mapping of original filenames to their actual stored filenames
        file_mapping = {}
        for row in list_result:
            if 'name' in row:
                path = row['name']
                path_parts = path.split('/')
                
                # The structure is expected to be: 
                # pdf_contracts_stage/client/upload_id/original_filename/stored_filename
                if len(path_parts) >= 5:
                    original_dir = path_parts[-2]  # Original filename used as directory
                    stored_file = path_parts[-1]   # Stored filename
                    file_mapping[original_dir] = stored_file
        
        # Update the stored_filename in file_records based on the mapping
        for record in file_records:
            original_filename = record["original_filename"]
            if original_filename in file_mapping:
                record["stored_filename"] = file_mapping[original_filename]
        
        # Process files in batches for parallelism
        batch_size = 3  # Process up to 3 files at once in parallel
        total_files = len(file_records)
        processed_files = 0
        failed_files = 0
        
        # Show a progress indicator
        progress_bar = st.progress(0)
        processing_status = st.empty()
        processing_status.info(f"Processing {total_files} files in parallel...")
        
        # Create a single metadata table
        session.sql("""
        CREATE OR REPLACE TABLE LEASE_METADATA (
            FILE_ID VARCHAR,
            ORIGINAL_FILENAME VARCHAR,
            STORED_FILENAME VARCHAR,
            METADATA VARCHAR
        )
        """).collect()
        
        # Check the structure of LEASE_CHUNKS table
        columns_check = session.sql("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = CURRENT_SCHEMA() 
            AND table_name = 'LEASE_CHUNKS'
        """).collect()
        
        column_names = [col['COLUMN_NAME'] for col in columns_check]
        
        # Create or update chunks table with correct structure
        has_parsed_document = 'PARSED_DOCUMENT' in column_names
        
        if not has_parsed_document:
            # If table exists but doesn't have PARSED_DOCUMENT column
            if 'LEASE_CHUNKS' in [t['TABLE_NAME'] for t in session.sql("SHOW TABLES").collect()]:
                session.sql("""
                ALTER TABLE LEASE_CHUNKS 
                ADD COLUMN PARSED_DOCUMENT VARCHAR
                """).collect()
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = file_records[i:i+batch_size]
            
            # Step 1: Parse documents for the entire batch in parallel
            parse_futures = []
            for record in batch:
                file_id = record["file_id"]
                original_filename = record["original_filename"]
                stored_filename = record["stored_filename"]
                
                # Construct the correct path with original filename as folder and stored filename as file
                relative_path = f"{client_name}/{upload_id}/{original_filename}/{stored_filename}"
                
                # Build parse statement and execute asynchronously
                parse_stmt = f"""
                UPDATE contract_pdf_files
                SET parsed_document = (
                    SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
                        '@pdf_contracts_stage',
                        '{relative_path}',
                        {{'mode': 'OCR'}}
                    )
                )
                WHERE file_id = '{file_id}'
                """
                
                parse_futures.append({
                    'future': session.sql(parse_stmt).collect_nowait(),
                    'record': record
                })
            
            # Wait for all parsing operations to complete
            parse_successful_records = []
            for future_item in parse_futures:
                try:
                    future_item['future'].result()  # Wait for completion
                    parse_successful_records.append(future_item['record'])
                except Exception as e:
                    failed_files += 1
                    st.error(f"Error parsing {future_item['record']['original_filename']}: {str(e)}")
            
            # Step 2: Extract metadata for successfully parsed files in parallel
            metadata_futures = []
            for record in parse_successful_records:
                file_id = record["file_id"]
                
                metadata_stmt = f"""
                INSERT INTO LEASE_METADATA
                SELECT 
                    FILE_ID,
                    ORIGINAL_FILENAME,
                    STORED_FILENAME,
                    SNOWFLAKE.CORTEX.COMPLETE('llama3.3-70b',
                        'I am going to provide a document which will be indexed by a retrieval system containing many similar documents. I want you to provide key information associated with this document that can help differentiate this document in the index. Follow these instructions:

                    1. Do not dwell on low level details. Only provide key high level information that a human might be expected to provide when searching for this doc.

                    2. Do not use any formatting, just provide keys and values using a colon to separate key and value. Have each key and value be on a new line.

                    3. Only extract at most the following information. If you are not confident with pulling any one of these keys, then do not include that key:\n'
                    ||
                    ARRAY_TO_STRING(
                        ARRAY_CONSTRUCT(
                            'contract term', 'effective date', 'applications', 'termination', 'change order comparison', 
                            'change order summary', 'application management', 'pricing changes', 'application timeline', 
                            'contract sections', 'change order signatories', 'application transitions', 'service changes',
                            'application count', 'contract owner', 'terms and conditions', 'pcr details', 'contract modifications', 
                            'execution timeline', 'price adjustments', 'contract value', 'change order value', 'approval authority', 
                            'services scope', 'supplemental terms', 'order form comparison', 'licensed products', 'pricing summary', 
                            'agreement timeline', 'payment terms', 'termination terms', 'vendor compliance', 'contract comparison', 
                            'service provider', 'contract amendments', 'stakeholder approval', 'contract dates', 
                            'service scope changes', 'pricing structure', 'contract governance', 'vendor relationship'),
                        '\t\t* ')
                    ||
                    '\n\nDoc starts here:\n' || SUBSTR(PARSED_DOCUMENT, 0, 4000) || '\nDoc ends here\n\n') AS METADATA
                FROM 
                    CONTRACT_PDF_FILES
                WHERE 
                    FILE_ID = '{file_id}'
                """
                
                metadata_futures.append({
                    'future': session.sql(metadata_stmt).collect_nowait(),
                    'record': record
                })
            
            # Wait for all metadata operations to complete
            metadata_successful_records = []
            for future_item in metadata_futures:
                try:
                    future_item['future'].result()  # Wait for completion
                    metadata_successful_records.append(future_item['record'])
                except Exception as e:
                    failed_files += 1
                    st.error(f"Error extracting metadata for {future_item['record']['original_filename']}: {str(e)}")
            
            # Step 3: Create chunks for files with successful metadata in parallel
            chunk_futures = []
            for record in metadata_successful_records:
                file_id = record["file_id"]
                
                chunk_stmt = f"""
                INSERT INTO LEASE_CHUNKS (
                                    FILE_ID,
                                    ORIGINAL_FILENAME,
                                    STORED_FILENAME,
                                    PARSED_DOCUMENT,
                                    CHUNK
                                )
                    WITH SPLIT_TEXT_CHUNKS AS (
                        SELECT
                            FILE_ID,
                            ORIGINAL_FILENAME,
                            STORED_FILENAME,
                            PARSED_DOCUMENT,
                            C.VALUE AS CHUNK,
                        FROM
                        CONTRACT_PDF_FILES,
                        LATERAL FLATTEN( input => SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER (
                            PARSED_DOCUMENT,
                            'none',
                            1800, -- SET CHUNK SIZE
                            300 -- SET CHUNK OVERLAP
                        )) C
                    )
                    SELECT
                        C.FILE_ID,
                        C.ORIGINAL_FILENAME,
                        C.STORED_FILENAME,
                        C.PARSED_DOCUMENT,
                        CONCAT(M.METADATA, '\n\n', C.CHUNK) AS CONTEXTUALIZED_CHUNK,
                    FROM
                        SPLIT_TEXT_CHUNKS C
                    JOIN
                        LEASE_METADATA M ON C.FILE_ID = M.FILE_ID
                    WHERE 
                        C.FILE_ID = '{file_id}'
                """
                
                chunk_futures.append({
                    'future': session.sql(chunk_stmt).collect_nowait(),
                    'record': record
                })
            
            # Wait for all chunk creation operations to complete
            for future_item in chunk_futures:
                try:
                    future_item['future'].result()  # Wait for completion
                    processed_files += 1  # Only count as processed if all steps succeed
                except Exception as e:
                    failed_files += 1
                    st.error(f"Error creating chunks for {future_item['record']['original_filename']}: {str(e)}")
            
            # Update progress bar
            progress = ((processed_files + failed_files) / total_files)
            processing_status.info(f"Processing files... ({processed_files + failed_files}/{total_files})")
            progress_bar.progress(progress)
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        
        # Show final summary
        if processed_files == total_files:
            processing_status.success(f"✅ Successfully processed all {total_files} files")
        elif processed_files > 0:
            processing_status.warning(f"⚠️ Processed {processed_files} of {total_files} files ({failed_files} failed)")
        else:
            processing_status.error(f"❌ Failed to process any files")
        
        return processed_files > 0
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return False

# Function to save uploaded files to Snowflake stage without deduplication checks
def save_files_to_stage(upload_id, client_name, uploaded_files):
    # Check if parsed_document column exists
    has_parsed_document = check_parsed_document_column()
    
    # Prepare upload records
    upload_timestamp = datetime.now()
    file_records = []
    
    for uploaded_file in uploaded_files:
        # Generate a unique ID for each file upload instead of using MD5 hash
        file_id = str(uuid.uuid4())
        
        file_content = uploaded_file.getvalue()
        original_filename = uploaded_file.name
        file_extension = os.path.splitext(original_filename)[1]
        
        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Put file into Snowflake stage
        try:
            # Upload file and get the result
            result = session.file.put(
                temp_path,
                f"@pdf_contracts_stage/{client_name}/{upload_id}/{original_filename}",
                auto_compress=False,
                overwrite=True
            )
            
            # Extract the actual staged filename from the result
            if result and len(result) > 0:
                # Try to extract the staged filename
                staged_filename = None
                
                # If result is a list of tuples
                if isinstance(result[0], tuple) and len(result[0]) >= 2:
                    staged_filename = str(result[0][1])  # Second element might contain the target path
                # If result is a list of dictionaries 
                elif isinstance(result[0], dict) and 'target' in result[0]:
                    staged_filename = result[0]['target']
                
                if not staged_filename or '@pdf_contracts_stage' not in staged_filename:
                    # List files in the directory to find our file
                    list_result = session.sql(f"""
                    LIST @pdf_contracts_stage/{client_name}/{upload_id}/
                    """).collect()
                    
                    # Find the most recently added file (likely our upload)
                    if list_result and len(list_result) > 0:
                        # Get the last file in the list (most recent upload)
                        last_file = list_result[-1]
                        if 'name' in last_file:
                            staged_path = last_file['name']
                            staged_filename = staged_path.split('/')[-1]
                
                # If all else fails, just use the original filename with tmp prefix
                if not staged_filename:
                    staged_filename = f"tmp_{original_filename}"
                    
                # Store the full stage path
                full_stage_path = f"@pdf_contracts_stage/{client_name}/{upload_id}/{staged_filename}"
            else:
                # Fallback if we can't get the actual filename
                staged_filename = original_filename
                full_stage_path = f"@pdf_contracts_stage/{client_name}/{upload_id}/{staged_filename}"
                st.warning(f"Could not determine actual staged filename for {original_filename}")
        except Exception as e:
            st.error(f"Error uploading file to Snowflake: {str(e)}")
            staged_filename = original_filename
            full_stage_path = f"@pdf_contracts_stage/{client_name}/{upload_id}/{staged_filename}"
        
        # Add record to our tracking with unique file_id
        file_records.append({
            "file_id": file_id,  # Using UUID as file_id instead of MD5 hash
            "upload_id": upload_id,
            "original_filename": original_filename,
            "stored_filename": staged_filename,
            "file_path": full_stage_path,
            "upload_timestamp": upload_timestamp,
            "file_size": len(file_content)
        })
        
        # Clean up temp file
        os.unlink(temp_path)
    
    # Record the client upload
    try:
        session.sql(f"""
        INSERT INTO client_contract_uploads (upload_id, client_name, upload_timestamp, num_files)
        VALUES ('{upload_id}', '{client_name}', '{upload_timestamp}', {len(file_records)})
        """).collect()
    except Exception as e:
        st.error(f"Error recording upload: {str(e)}")
        return []
    
    # Record individual files
    for record in file_records:
        try:
            if has_parsed_document:
                # Include parsed_document column
                session.sql(f"""
                INSERT INTO contract_pdf_files (
                    file_id, upload_id, original_filename, stored_filename, file_path, 
                    upload_timestamp, file_size, parsed_document
                )
                VALUES (
                    '{record["file_id"]}', 
                    '{record["upload_id"]}', 
                    '{record["original_filename"]}', 
                    '{record["stored_filename"]}', 
                    '{record["file_path"]}', 
                    '{record["upload_timestamp"]}', 
                    {record["file_size"]},
                    NULL
                )
                """).collect()
            else:
                # Skip the parsed_document column
                session.sql(f"""
                INSERT INTO contract_pdf_files (
                    file_id, upload_id, original_filename, stored_filename, file_path, 
                    upload_timestamp, file_size
                )
                VALUES (
                    '{record["file_id"]}', 
                    '{record["upload_id"]}', 
                    '{record["original_filename"]}', 
                    '{record["stored_filename"]}', 
                    '{record["file_path"]}', 
                    '{record["upload_timestamp"]}', 
                    {record["file_size"]}
                )
                """).collect()
        except Exception as e:
            st.error(f"Error recording file: {str(e)}")
    
    return file_records

# Verify required Snowflake objects exist
def verify_snowflake_objects():
    try:
        # Check if tables exist
        tables_check = session.sql("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = CURRENT_SCHEMA() 
        AND table_name IN ('CLIENT_CONTRACT_UPLOADS', 'CONTRACT_PDF_FILES')
        """).collect()
        
        # Check if stage exists
        stage_check = session.sql("""
        SELECT stage_name 
        FROM information_schema.stages 
        WHERE stage_schema = CURRENT_SCHEMA() 
        AND stage_name = 'PDF_CONTRACTS_STAGE'
        """).collect()
        
        # Validate all required objects exist
        if len(tables_check) != 2 or len(stage_check) != 1:
            st.error("""
            Required Snowflake objects are missing. Please run the setup script first.
            
            Missing objects:
            - Tables: CLIENT_CONTRACT_UPLOADS, CONTRACT_PDF_FILES
            - Stage: PDF_CONTRACTS_STAGE
            """)
            st.stop()
            
    except Exception as e:
        st.error(f"Error verifying Snowflake objects: {str(e)}")
        st.stop()

# Get lease context for a specific upload
def get_lease_context(client_name, upload_id):
    """Get context from the lease documents for a specific upload"""
    context = ""
    
    if client_name and upload_id:
        try:
            # Get the 5 most recent chunks as context
            context_chunks = session.sql(f"""
            SELECT lc.CHUNK 
            FROM LEASE_CHUNKS lc
            JOIN CONTRACT_PDF_FILES cf ON lc.FILE_ID = cf.FILE_ID
            WHERE cf.upload_id = '{upload_id}'
            LIMIT 5
            """).collect()
            
            # Build context string
            if context_chunks and len(context_chunks) > 0:
                for i, chunk in enumerate(context_chunks):
                    context += f"Document chunk {i+1}: {chunk['CHUNK']}\n\n"
        except Exception as e:
            st.sidebar.error(f"Error retrieving lease context: {str(e)}")
    
    return context

# Display chat interface
def render_chat_interface(client_name=None, upload_id=None):
    # Initialize chat messages
    init_messages()
    
    # Initialize service metadata for Cortex Search
    init_service_metadata()
    
    # Initialize configuration options
    init_config_options()
    
    icons = {"assistant": "📄", "user": "👤"}
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.markdown(message["content"])
    
    # Handle new user input
    if question := st.chat_input("Ask about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message in chat message container
        with st.chat_message("user", avatar=icons["user"]):
            st.markdown(question.replace("$", "\$"))
        
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=icons["assistant"]):
            message_placeholder = st.empty()
            question_safe = question.replace("'", "")
            
            # Get additional context from lease documents if available
            additional_context = get_lease_context(client_name, upload_id)
            
            # Create prompt and generate response
            with st.spinner("Thinking..."):
                prompt, results = create_prompt(question_safe, additional_context)
                
                if "model_name" not in st.session_state:
                    st.session_state.model_name = "claude-3-5-sonnet"
                    
                generated_response = complete(st.session_state.model_name, prompt)
                
                # Display the response
                message_placeholder.markdown(generated_response)
                
                # Build references table if there are search results
                if results:
                    with st.expander("References"):
                        st.write("Search results that informed this answer:")
                        st.dataframe(results)
            
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": generated_response}
            )

# Function to check if client name already exists
def check_client_name_exists(client_name):
    """Check if a client name already exists in the database"""
    try:
        result = session.sql(f"""
        SELECT COUNT(*) as client_count
        FROM client_contract_uploads
        WHERE client_name = '{client_name}'
        """).collect()
        
        return result[0]['CLIENT_COUNT'] > 0
    except Exception as e:
        st.error(f"Error checking client name: {str(e)}")
        return False

# Setup minimal chat config options
def init_config_options():
    """
    Initialize the configuration options for chat with default values.
    Only show the Clear Conversation button in the sidebar.
    All other settings are initialized with default values but not displayed.
    """
    # Only show the clear conversation button in the sidebar
    st.sidebar.button("Clear conversation", on_click=clear_chat_history)
    
    # Initialize settings with default values (without displaying them)
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODELS[0]  # Default to first model in list
        
    if "num_retrieved_chunks" not in st.session_state:
        st.session_state.num_retrieved_chunks = 200
        
    if "num_chat_messages" not in st.session_state:
        st.session_state.num_chat_messages = 10
        
    if "use_chat_history" not in st.session_state:
        st.session_state.use_chat_history = True
        
    if "debug" not in st.session_state:
        st.session_state.debug = False
    
    # Initialize cortex search service if available
    if "service_metadata" in st.session_state and st.session_state.service_metadata and "selected_cortex_search_service" not in st.session_state:
        # Set the first available service as default
        st.session_state.selected_cortex_search_service = st.session_state.service_metadata[0]["name"]

# Main application
def main():
    # Verify required Snowflake objects exist
    verify_snowflake_objects()
    
    # Initialize session state variables if they don't exist
    if 'upload_complete' not in st.session_state:
        st.session_state.upload_complete = False
    
    # Initialize the current page in session state if not already set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload Contracts"
    
    # Keep track of previous page to detect page changes
    previous_page = st.session_state.get("previous_page", None)
    
    # Add sidebar navigation - clean and minimal
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Contracts", "Contract Chat"], 
                           index=0 if st.session_state.current_page == "Upload Contracts" else 1,
                           key="navigation")
    
    # Update current page based on navigation
    st.session_state.current_page = page
    
    # Add document selection dropdown in sidebar only if on Contract Chat page
    if page == "Contract Chat":
        st.sidebar.markdown("---")  # Add a separator
        
        # Fetch available uploads for the dropdown
        try:
            uploads = session.sql("""
            SELECT upload_id, client_name, upload_timestamp, num_files 
            FROM client_contract_uploads 
            ORDER BY upload_timestamp DESC
            """).collect()
            
            if uploads and len(uploads) > 0:
                # Create dropdown options - show only client name
                upload_options = [row['CLIENT_NAME'] for row in uploads]
                
                # Track previous selection
                previous_selection = st.session_state.get("selected_upload_idx", None)
                
                # Add a dropdown to select documents
                st.sidebar.markdown("### Document Selection")
                selected_idx = st.sidebar.selectbox(
                    "Choose documents to chat about:", 
                    range(len(upload_options)),
                    format_func=lambda x: upload_options[x],
                    key="selected_upload_idx"
                )
                
                # If selection changed, clear chat history
                if previous_selection != selected_idx and previous_selection is not None:
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                
                # Update client_name and upload_id based on selection
                if selected_idx is not None:
                    st.session_state.current_client_name = uploads[selected_idx]["CLIENT_NAME"]
                    st.session_state.current_upload_id = uploads[selected_idx]["UPLOAD_ID"]
            else:
                st.sidebar.warning("No uploads available")
        except Exception as e:
            st.sidebar.error(f"Error loading uploads: {str(e)}")
 
    # Check if we're changing pages to Contract Chat
    if previous_page != page and page == "Contract Chat":
        # Clear chat history when navigating to chat page
        if "messages" in st.session_state:
            st.session_state.messages = []
    
    # Update previous page for next cycle
    st.session_state.previous_page = page
    
    # Main page content
    if page == "Upload Contracts":
        st.title("Contract PDF Processor")
        st.markdown("""
        This application allows you to upload PDF contract files for a client.
        The system will automatically store the files and process them using Snowflake Cortex.
        """)
        
        # Check if we need to start a new upload
        if st.session_state.upload_complete:
            st.success("Your previous upload was completed successfully.")
            if st.button("Start a New Upload"):
                # Reset upload state
                st.session_state.upload_complete = False
                # Clear current upload info
                if 'current_upload_id' in st.session_state:
                    del st.session_state.current_upload_id
                if 'current_client_name' in st.session_state:
                    del st.session_state.current_client_name
                st.rerun()
            
            # Show option to chat about these docs
            st.info("You can chat about your uploaded documents in the Contract Chat section.")
            if st.button("Go to Contract Chat"):
                # Clear chat history before switching pages
                if "messages" in st.session_state:
                    st.session_state.messages = []
                # Switch to chat page by updating session state and triggering rerun
                st.session_state.current_page = "Contract Chat"
                st.rerun()
        else:
            # Standard upload flow
            st.header("Upload Contract Files")
            
            # Get client name
            client_name = st.text_input("Client Name", key="client_name")
            client_name = client_name.replace(" ", "_")
            
            # Check if client name exists
            client_name_exists = False
            client_name_error = st.empty()
            
            if client_name:
                client_name_exists = check_client_name_exists(client_name)
                if client_name_exists:
                    client_name_error.error("This client name already exists. Please use a different name.")
            
            uploaded_files = st.file_uploader("Upload PDF Contracts", 
                                            type=["pdf"], 
                                            accept_multiple_files=True,
                                            key="file_uploader")
            
            if uploaded_files and client_name and not client_name_exists:               
                file_info = []
                for uploaded_file in uploaded_files:
                    file_info.append({
                        "filename": uploaded_file.name,
                        "size": uploaded_file.size,
                        "type": uploaded_file.type
                    })
                
                if 1 <= len(uploaded_files) <= 20:
                    upload_id = str(uuid.uuid4())
                    
                    # Store upload ID for chat
                    st.session_state.current_upload_id = upload_id
                    st.session_state.current_client_name = client_name
                    
                    with st.spinner("Uploading and storing files..."):
                        file_records = save_files_to_stage(upload_id, client_name, uploaded_files)
                        
                    if file_records:
                        st.success(f"Successfully uploaded {len(file_records)} files for client {client_name}")
                        
                        # Process files immediately using robust processing approach
                        with st.spinner("Processing files with AI..."):
                            success = process_files_with_ai_parallel(file_records, client_name, upload_id)
                            if success:
                                st.success("🎉 AI processing complete!")
                                # Mark upload as complete
                                st.session_state.upload_complete = True
                                st.rerun()
                            else:
                                st.error("❌ There was an error during AI processing.")
                    else:
                        st.error("Failed to upload files. Check the error messages above.")
                        
                elif len(uploaded_files) > 20:
                    st.error("You can upload a maximum of 20 files at once")
            elif uploaded_files and client_name and client_name_exists:
                st.warning("Please choose a different client name before uploading files")
            elif uploaded_files and not client_name:
                st.warning("Please enter a client name before uploading files")
    
    elif page == "Contract Chat":
        st.title("Contract Chat")
        st.markdown("""
        Ask questions about your documents and get AI-powered answers using Snowflake Cortex Search.
        """)
        
        # Get client name and upload ID for context
        client_name = st.session_state.get("current_client_name", None)
        upload_id = st.session_state.get("current_upload_id", None)
        
        if client_name and upload_id:
            st.success(f"Chatting about documents for client: {client_name}")
            
            # Render the chat interface with context
            render_chat_interface(client_name, upload_id)
        else:
            st.warning("Please select a document set from the sidebar to start chatting.")

# Run the application
if __name__ == "__main__":
    main()
