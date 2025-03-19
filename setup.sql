USE ROLE DEMO_ROLE;

-- Fetch most recent files from Github repository
ALTER GIT REPOSITORY GITHUB_REPO_AI_CONTRACT_INTELLIGENCE FETCH;

-- Create Streamlit app with method based on status of Behavior Change Bundle 2025_01
BEGIN
  -- Check if 2025 Bundle is enabled (multi-page streamlit apps)
  LET status_2025_01 STRING := (SELECT SYSTEM$BEHAVIOR_CHANGE_BUNDLE_STATUS('2025_01'));
  -- Log result
  SYSTEM$LOG_INFO('Bundle 2025_01 is ' || :status_2025_01);
  
  -- Check if the bundle is ENABLED and execute appropriate commands
  IF (status_2025_01 = 'ENABLED') THEN

    -- Create Streamlit with multifile editing
    CREATE STREAMLIT CORTEX_AI_CONTRACT_CHAT_APP
      FROM @DOC_EXTRACTION_DB.PUBLIC.GITHUB_REPO_AI_CONTRACT_INTELLIGENCE/branches/main/app/
      MAIN_FILE = 'streamlit_app.py'
      QUERY_WAREHOUSE = COMPUTE_WH
      TITLE = 'Cortex AI Contracts Chat App'
      COMMENT = 'Demo Streamlit frontend for Contract Intelligence';

    RETURN 'Bundle 2025_01 is ENABLED. Created Streamlit app with multi-file editing.';
  ELSE
    -- Check if the bundle is DISABLED and execute appropriate commands
      -- Create stage for the Streamlit App
      CREATE OR REPLACE STAGE STREAMLIT_APP
        DIRECTORY = (ENABLE = TRUE)
        ENCRYPTION = ( TYPE = 'SNOWFLAKE_FULL' );

      -- Copy Streamlit App into to stage
      COPY FILES
        INTO @STREAMLIT_APP
        FROM @DOC_EXTRACTION_DB.PUBLIC.GITHUB_REPO_AI_CONTRACT_INTELLIGENCE/branches/main/app/;
      ALTER STAGE STREAMLIT_APP REFRESH;

      -- Create Streamlit App
      CREATE OR REPLACE STREAMLIT CORTEX_AI_CONTRACT_CHAT_APP
          ROOT_LOCATION = '@DOC_EXTRACTION_DB.PUBLIC.STREAMLIT_APP'
          MAIN_FILE = '/streamlit_app.py'
          QUERY_WAREHOUSE = COMPUTE_WH
          TITLE = 'Cortex AI Contracts Chat App'
          COMMENT = 'Demo Streamlit frontend for Contract Intelligence';

      RETURN 'Bundle 2025_01 is DISABLED. Created Streamlit app with single file editing.';
  END IF;
END;
