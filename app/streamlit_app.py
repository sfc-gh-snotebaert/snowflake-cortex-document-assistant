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
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
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

# Setup chat config options
def init_config_options():
    """
    Initialize the configuration options in the Streamlit sidebar for chat.
    """
    if st.session_state.service_metadata:
        st.sidebar.selectbox(
            "Select cortex search service:",
            [s["name"] for s in st.session_state.service_metadata],
            key="selected_cortex_search_service",
        )

    # Use on_click to call the clear function instead of a session state variable
    st.sidebar.button("Clear conversation", on_click=clear_chat_history)
    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name", index=1)  # Default to llama3.1-70b
        st.number_input(
            "Select number of context chunks",
            value=15,
            key="num_retrieved_chunks",
            min_value=10,
            max_value=20,
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=10,
            key="num_chat_messages",
            min_value=1,
            max_value=10,
        )

    if st.session_state.debug:
        st.sidebar.expander("Session State").write(st.session_state)

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
        You are an expert legal professional like the fictional character Mike Ross from the show Suits.

        Review the following question and the provided legal evidence to answer the question in a legally accurate manner. 

        Please pay extra attention to the date of the legal details. Always prioritize newer information if there is a contradiction to past information.

        
        <context>
        {prompt_context}
        </context>
        
        <question>
        {user_question}
        </question>
        
        Please provide a helpful, concise, and accurate response based on the context provided.
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

# Function to process files with full document parsing, metadata extraction, and chunking in a robust way
def process_files_with_ai_parallel(file_records, client_name, upload_id):
    if not file_records:
        return False
    
    try:
        # Get all the file paths first
        list_result = session.sql(f"""
        LIST @pdf_contracts_stage/{client_name}/{upload_id}/
        """).collect()
        
        # Process files in smaller batches to avoid internal errors
        batch_size = 5  # Process up to 5 files at once
        total_files = len(file_records)
        processed_files = 0
        success = True
        
        # Create a single metadata table
        session.sql("""
        CREATE OR REPLACE TABLE LEASE_METADATA (
            FILE_ID VARCHAR,
            ORIGINAL_FILENAME VARCHAR,
            STORED_FILENAME VARCHAR,
            METADATA VARCHAR
        )
        """).collect()
        
        # Create chunks table if it doesn't exist
        session.sql("""
        CREATE TABLE IF NOT EXISTS LEASE_CHUNKS (
            FILE_ID VARCHAR,
            ORIGINAL_FILENAME VARCHAR,
            STORED_FILENAME VARCHAR,
            CHUNK VARCHAR
        )
        """).collect()
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = file_records[i:i+batch_size]
            st.write(f"Processing batch {i//batch_size + 1} of {(total_files + batch_size - 1) // batch_size} ({len(batch)} files)...")
            
            # Process each file in the batch individually
            for record in batch:
                try:
                    file_id = record["file_id"]
                    original_filename = record["original_filename"]
                    stored_filename = record["stored_filename"]
                    
                    # Find matching files
                    matching_files = []
                    for row in list_result:
                        if 'name' in row and original_filename in row['name']:
                            matching_files.append(row['name'])
                    
                    if matching_files:
                        relative_path = matching_files[0].split('PDF_CONTRACTS_STAGE/')[1] if 'PDF_CONTRACTS_STAGE/' in matching_files[0] else f"{client_name}/{upload_id}/{original_filename}/{stored_filename}"
                        
                        # Step 1: Parse document
                        st.write(f"Parsing document: {original_filename}")
                        session.sql(f"""
                        UPDATE contract_pdf_files
                        SET parsed_document = (
                            SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
                                '@pdf_contracts_stage',
                                '{relative_path}',
                                {{'mode': 'LAYOUT'}}
                            )
                        )
                        WHERE file_id = '{file_id}'
                        """).collect()
                        
                        # Step 2: Extract metadata
                        st.write(f"Extracting metadata from: {original_filename}")
                        session.sql(f"""
                        INSERT INTO LEASE_METADATA
                        SELECT 
                            FILE_ID,
                            ORIGINAL_FILENAME,
                            STORED_FILENAME,
                            SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', 
                                CONCAT($$Extract the following information from the lease in JSON format as follows. Just provide the JSON: 
                                        [METADATA {{
                                        'commencement_date' : 'value',
                                        'location': 'value', 
                                        'street': 'value', 
                                        'city': 'value',
                                        'province': 'value', 
                                        'landlord': 'value', 
                                        'tenant': 'value' }} ] $$, 
                                    PARSED_DOCUMENT ) ) AS METADATA
                        FROM 
                            CONTRACT_PDF_FILES
                        WHERE 
                            FILE_ID = '{file_id}'
                        """).collect()
                        
                        # Step 3: Create chunks
                        st.write(f"Creating chunks for: {original_filename}")
                        session.sql(f"""
                        INSERT INTO LEASE_CHUNKS
                        SELECT
                            cf.FILE_ID,
                            cf.ORIGINAL_FILENAME,
                            cf.STORED_FILENAME,
                            cf.PARSED_DOCUMENT,
                            c.VALUE || (SELECT METADATA FROM LEASE_METADATA WHERE FILE_ID = '{file_id}') AS chunk
                        FROM
                            CONTRACT_PDF_FILES cf,
                            LATERAL FLATTEN( 
                                SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER (
                                    cf.parsed_document,
                                    'markdown',
                                    1200,
                                    120 
                                ) 
                            ) c
                        WHERE 
                            cf.FILE_ID = '{file_id}'
                        """).collect()
                        
                        processed_files += 1
                        st.success(f"✅ Completed processing {original_filename}")
                    else:
                        st.error(f"Could not find file in stage: {original_filename}")
                except Exception as e:
                    st.error(f"Error processing {original_filename}: {str(e)}")
                    # Continue with next file
        
        if processed_files == total_files:
            st.success(f"✅ Successfully processed all {total_files} files")
        else:
            st.warning(f"⚠️ Processed {processed_files} of {total_files} files with some errors")
        
        return processed_files > 0
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return False

# Function to save uploaded files to Snowflake stage with MD5 hash-based deduplication
def save_files_to_stage(upload_id, client_name, uploaded_files):
    # Check if parsed_document column exists
    has_parsed_document = check_parsed_document_column()
    
    # Prepare upload records
    upload_timestamp = datetime.now()
    file_records = []
    
    for uploaded_file in uploaded_files:
        # Calculate MD5 hash and use it as the file_id
        file_content = uploaded_file.getvalue()
        file_id = calculate_md5(file_content)
        
        # Check if file already exists and its processing status
        file_exists, is_parsed, metadata_exists, chunks_exist, existing_files = check_file_exists(file_id)
        
        if file_exists:
            # Determine complete processing status
            fully_processed = is_parsed and metadata_exists and chunks_exist
            
            if fully_processed:
                # File exists and has been fully processed
                for existing_file in existing_files:
                    st.warning(f"File '{uploaded_file.name}' already exists as '{existing_file['ORIGINAL_FILENAME']}' (uploaded for client {existing_file['CLIENT_NAME']})")
                continue
            else:
                # File exists but wasn't fully processed
                status_msg = "File was previously uploaded but "
                if not is_parsed:
                    status_msg += "document parsing failed."
                elif not metadata_exists:
                    status_msg += "metadata extraction failed."
                elif not chunks_exist:
                    status_msg += "chunking failed."
                    
                # Offer to reprocess
                reprocess = st.checkbox(f"File '{uploaded_file.name}': {status_msg} Reprocess it?", value=True)
                if not reprocess:
                    continue
                # If user wants to reprocess, we'll continue with the upload
        
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
                # The result structure can vary based on API version
                # Debug the structure
                st.write("Upload result structure:", result)
                
                # Try to extract the staged filename
                staged_filename = None
                
                # If result is a list of tuples
                if isinstance(result[0], tuple) and len(result[0]) >= 2:
                    staged_filename = str(result[0][1])  # Second element might contain the target path
                # If result is a list of dictionaries 
                elif isinstance(result[0], dict) and 'target' in result[0]:
                    staged_filename = result[0]['target']
                # If we can't extract it properly, list the directory to find it
                
                if not staged_filename or '@pdf_contracts_stage' not in staged_filename:
                    # List files in the directory to find our file
                    list_result = session.sql(f"""
                    LIST @pdf_contracts_stage/{client_name}/{upload_id}/
                    """).collect()
                    
                    st.write("Files in stage:", list_result)
                    
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
                
                st.write(f"Original filename: {original_filename}")
                st.write(f"Staged as: {staged_filename}")
            else:
                # Fallback if we can't get the actual filename
                staged_filename = original_filename
                full_stage_path = f"@pdf_contracts_stage/{client_name}/{upload_id}/{staged_filename}"
                st.warning(f"Could not determine actual staged filename for {original_filename}")
        except Exception as e:
            st.error(f"Error uploading file to Snowflake: {str(e)}")
            staged_filename = original_filename
            full_stage_path = f"@pdf_contracts_stage/{client_name}/{upload_id}/{staged_filename}"
        
        # Add record to our tracking with MD5 hash as file_id
        file_records.append({
            "file_id": file_id,  # Using MD5 hash as file_id
            "upload_id": upload_id,
            "original_filename": original_filename,
            "stored_filename": staged_filename,  # Store the actual filename in Snowflake (with tmp prefix)
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
                    st.session_state.model_name = "llama3.1-70b"
                    
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

# Main application
def main():
    # Verify required Snowflake objects exist
    verify_snowflake_objects()
    
    # Add sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Contracts", "Contract Chat"])
    
    if page == "Upload Contracts":
        st.title("Contract PDF Processor")
        st.markdown("""
        This application allows you to upload PDF contract files for a client.
        The system will automatically store the files and process them using Snowflake Cortex.
        """)
        
        st.header("Upload Contract Files")
        
        # Initialize session state if needed
        if 'upload_complete' not in st.session_state:
            st.session_state.upload_complete = False
        
        # Get client name
        client_name = st.text_input("Client Name", key="client_name")
        client_name = client_name.replace(" ", "_")

        
        # Show file uploader only if we don't have a completed upload to process
        if not st.session_state.upload_complete:
            uploaded_files = st.file_uploader("Upload PDF Contracts", 
                                            type=["pdf"], 
                                            accept_multiple_files=True,
                                            key="file_uploader")
            
            if uploaded_files and client_name:
                st.write(f"Number of files selected: {len(uploaded_files)}")
                
                file_info = []
                for uploaded_file in uploaded_files:
                    file_info.append({
                        "filename": uploaded_file.name,
                        "size": uploaded_file.size,
                        "type": uploaded_file.type
                    })
                
                st.dataframe(file_info)
                
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
                                
                                # Show option to chat about these docs
                                st.info("You can now chat about these documents in the Contract Chat section.")
                                if st.button("Go to Contract Chat"):
                                    st.session_state.page = "Contract Chat"
                                    st.experimental_rerun()
                            else:
                                st.error("❌ There was an error during AI processing.")
                    else:
                        st.error("Failed to upload files. Check the error messages above.")
                        
                elif len(uploaded_files) > 20:
                    st.error("You can upload a maximum of 20 files at once")
            elif uploaded_files and not client_name:
                st.warning("Please enter a client name before uploading files")
        else:
            # Show a button to upload more files
            if st.button("Upload More Files"):
                st.session_state.upload_complete = False
                # Clear session state for new upload
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
    
    elif page == "Contract Chat":
        st.title("Contract Chat")
        st.markdown("""
        Ask questions about your documents and get AI-powered answers using Snowflake Cortex Search.
        """)
        
        # Get client name and upload ID for context
        client_name = st.session_state.get("current_client_name", None)
        upload_id = st.session_state.get("current_upload_id", None)
        
        if not client_name or not upload_id:
            # Allow user to select from available uploads
            try:
                uploads = session.sql("""
                SELECT upload_id, client_name, upload_timestamp, num_files 
                FROM client_contract_uploads 
                ORDER BY upload_timestamp DESC
                """).collect()
                
                if uploads and len(uploads) > 0:
                    st.write("Select which documents to chat about:")
                    upload_options = [f"{row['CLIENT_NAME']} - {row['UPLOAD_TIMESTAMP']} ({row['NUM_FILES']} files)" for row in uploads]
                    
                    selected_idx = st.selectbox(
                        "Available uploads:", 
                        range(len(upload_options)),
                        format_func=lambda x: upload_options[x]
                    )
                    
                    if selected_idx is not None:
                        client_name = uploads[selected_idx]["CLIENT_NAME"]
                        upload_id = uploads[selected_idx]["UPLOAD_ID"]
                        st.success(f"Chatting about documents for client: {client_name}")
                else:
                    st.warning("No uploads found. Please upload documents first.")
            except Exception as e:
                st.error(f"Error loading available uploads: {str(e)}")
        else:
            st.success(f"Chatting about documents for client: {client_name}")
        
        # Render the chat interface with context
        render_chat_interface(client_name, upload_id)

# Run the application
if __name__ == "__main__":
    main()