# Databricks notebook source
# MAGIC %pip install langchain_community

# COMMAND ----------

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# COMMAND ----------

from langchain.chains import LLMChain, SequentialChain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Call Llama 70B for this Lab
import os
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chat_models import ChatDatabricks

# COMMAND ----------


llama_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens = 400)

# COMMAND ----------


#system_template=''' Below is the data schema you need to use for generating python code
#postingerrorDF= {
#  "properties":{
#    "PostingErrorDate":{"type": "date", "description":"Date when the Error Log was recorded"},
#    "PostingError":{"type": "string", "description":"This is the posting error log message""},
#    "LineConfirmationID_d": {"type": "string", "description": "This is the unique identifier for the line confirmation"},
#    "SalesOrder_d":{"type": "string", "description": "This is the sales order number associated with the sales line"},
#    "join_columns1": { "description": "Please use ALL THE COLUMNS to join with salesorderlinesDF ", "columns" :"["SalesOrder_d","LineConfirmationID_d]"},
#    "join_columns2": { "description": "Please use ALL THE COLUMNS to join with vouchersDF ", "columns" :"["SalesOrder_d", "LineConfirmationID_d"]"},
#    "join_columns2": { "description": "Please use ALL THE COLUMNS to join with cardauthorizationsDF ","columns" :"["SalesOrder_d", #"LineConfirmationID_d"]"}
#  }
#}'''
#
#system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
#
#

# COMMAND ----------

#human_template='''Generate a python code to extract all the rows where a part of the posting error should contain the following message (Refund failed: #AshleyPaymentGateway:Full refund is done against reference number)'''
#human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# COMMAND ----------

#chat_prompt = ChatPromptTemplate(messages=[system_message_prompt])#, human_message_prompt])
#chain1= LLMChain(llm=llama_model, prompt=chat_prompt) 
#result=chain1.run(user_input=human_template)
#print(result)

# COMMAND ----------

#---------------------Input Schemas-----------------

system_prompt= '''
As a AI Assistant throroughly understand the request as described by the user {user_input}.iSuggest the appropriate schema for the code genration, you should follow these steps:

1. Understand the JSON schema: Familiarize yourself with the different types of JSON schema objects and their properties. This will help you gain a better understanding of how to analyze and interpret these objects.

2. Analyze the issue: Identify the specific issue or problem that needs to be addressed. What kind of data or functionality is required for code generation ? Determine the specific attributes or properties that need to be present in the JSON schema.

3. Develop a parsing algorithm: Create an algorithm that can parse and analyze the JSON schema objects. This algorithm should consider the issue at hand and compare it with the available schema objects. It should evaluate the compatibility and relevance of each schema object based on predefined criteria.

4. Define criteria for schema selection: Determine the criteria for selecting the most suitable schema for the code generation. This may include factors such as data type compatibility, required attributes, constraints, and any specific business rules or guidelines.

5: AVOID REDUNDANT SCHEMA JOINS: If a SINGLE SCHEMA CONTAINS ALL NECESSARY DATA, EXCLUDE JOINING OTHER SCHEMAS to Avoid Confusion for code generation. Prioritize clarity and simplicity by recommending only the most relevant schema.

6: Output: COntains two parts: 
    a: Give the Schema name, and the column names with properties as the output. 
    b: Provide the summary of the user request in the output.

Below are the schemas for you to analyze and accurately comprehend based on the above rules.

postingerrorDF= {{
  "properties":{{
    "PostingErrorDate":{{"type": "date", "description":"Date when the Error Log was recorded"}},
    "PostingError":{{"type": "string", "description":"This is the posting error log message""}},
    "LineConfirmationID_d": {{"type": "string", "description": "This is the unique identifier for the line confirmation"}},
    "SalesOrder_d":{{"type": "string", "description": "This is the sales order number associated with the sales line"}},
    "join_columns1": {{ "description": "Please use ALL THE COLUMNS to join with salesorderlinesDF ", "columns" :"["SalesOrder_d","LineConfirmationID_d]}},
    "join_columns2": {{ "description": "Please use ALL THE COLUMNS to join with vouchersDF ", "columns" :"["SalesOrder_d", "LineConfirmationID_d"]"}},
    "join_columns2": {{ "description": "Please use ALL THE COLUMNS to join with cardauthorizationsDF ","columns" :"["SalesOrder_d", "LineConfirmationID_d"]"}}
  }}
}}

cardauthorizationsDF = {{
  "properties": {{
    "LineConfirmationID_b": {{"type": "string", "description": "This is the unique identifier for the line confirmation"}},
    "SalesOrder_b": {{"type": "string", "description": "This is the sales order number associated with the line confirmation"}},
    "Type": {{"type": "string", "description": "This is the type of the line confirmation", "distinct_values": " ['Post Authorization', 'Credit', 'Void', 'Finalization', 'Authorization' ]" }},
    "Status": {{"type": "string", "description": "This is the status of the line confirmation", "distinct_values":"['Approved', 'Settled', 'Declined']"}},
    "Date": {{"type": "timestamp", "description": "This is the date of the line confirmation or the date of the event occurrences."}},
    "CreditCardType": {{"type": "string", "description": "This is the type of credit card used", "distinct_values": "'customer', 'genesis', 'Discover', 'VISA', 'Caddi', None, 'Visa', 'Progressive', 'DISCOVER', 'Synchrony', 'Genesis', 'Gafco', 'AMEX', 'PayPal', 'Acima', 'MasterCard', 'Amex', 'synchrony']"}},
    "Amount": {{"type": "decimal(32,6)", "description": "This is the amount of the line confirmation in the currency of the sales order"}},
    "AuthorizationCode": {{"type": "string", "description": "This is the authorization code "}},
    "Description": {{"type": "string", "description": "This is the optional description of the line confirmation"}},
    "SeqNo": {{"type": "decimal(32,16)", "description": "This is the sequence of authorizations in the sales order"}},
    "join_columns1": {{ "description": "Please use ALL THE COLUMNS to join with salesorderlinesDF ", "columns" :"["SalesOrder_b", "LineConfirmationID_b"]"}},
    "join_columns2": {{ "description": "Please use ALL THE COLUMNS to join with postingerrorDF ", "columns" :"["SalesOrder_b", "LineConfirmationID_b"]"}}
  }}
}}

'''

#---------------Prompt Template------------
chat_template = ChatPromptTemplate.from_template(system_prompt)

chain1= LLMChain(llm=llama_model, prompt=chat_template, output_key="data_schema_user_input") 

# COMMAND ----------

#---------------------Code Generation-----------------

system_prompt= '''
You are an AI assistant that is well versed in writing pyspark queries. Generate the code based on the schema and the user request provided here: {data_schema_user_input}

When generating Python code, Here are some RULES YOU MUST FOLLOW to generate successful code:

a. Adhere to the guidelines provided in PEP 8, which covers a wide range of coding style conventions, including naming conventions,whitespace usage, line length, and more.
b. Use Descriptive and Meaningful Variable Names. Generate code that uses variable names that accurately describe the purpose or content of the variable. Avoid single-letter or Ambiguous Variable names.
c. EXCLUDE try-except blocks while generating the code.
d: DEFER to DataSchemaAgent for TABLE and COLUMN NAMES. Use provided names as the authoritative source to ensure accuracy and consistency.
e: ASSUME existing tables; DO NOT IMPORT any tables or files. Code should utilize tables mentioned by DataSchemaAgent as if they already exist.
f: AVOID ASSUMPTIONS ABOUT UNDEFINED COLUMN NAMES. Only rely on columns explicitly defined in the schemas to ensure accuracy and prevent errors.
g: AVOID REDUNDANT SCHEMA JOINS: If a SINGLE SCHEMA CONTAINS ALL NECESSARY DATA, EXCLUDE JOINING OTHER SCHEMAS to Avoid Confusion.
h: Please ALIAS THE DATASETS with different names via `Dataset.as` before joining them, and specify the column using qualified name, e.g. `df.as("a
").join(df.as("b"), $"a.id" > $"b.id") 
i: ELIMINATE WINDOW FUNCTIONS while developing the code. Always find a way to generate the code without Window functions.    
j: IMPORT ALL REQUIRED LIBRARIES for the code.
k: Do not filter your code for a single sales order. Your code must be applicable across all sales orders affected by the issue.
l: Generate code exclusively, placing any additional notes or information strictly within code comments.
m: Craft your generated code within a Python code block.
n: Finally, name your completed data frame as "resultDF"
'''



#---------------Prompt Template------------

chat_template = ChatPromptTemplate.from_template(system_prompt)

chain2= LLMChain(llm=llama_model, prompt=chat_template, output_key="python_code") 

# COMMAND ----------


human_prompt='''Extract Sales orders that have same Line Confirmation ID, where the previous Status was Approved but the latest Status is Declined and the Type is Void'''

seq_chain=SequentialChain(chains=[chain1,chain2]
                          ,input_variables=['human_prompt']
                          ,output_variables=['data_schema_user_input','python_code'])




# COMMAND ----------

#---------------------Input Schemas-----------------

# system_prompt2= '''
# You are a code reviewer and specialized in python coding. Test the python code:\n {python_code} to make sure that the code is correct. If it is correct, return the code and if it is not correct, change where needed and then retrun the code
# '''

# #---------------Prompt Template------------
# chat_template2 = ChatPromptTemplate.from_template(system_prompt2)
# chain2= LLMChain(llm=llama_model, prompt=chat_template2,output_key='review') 


# COMMAND ----------



# COMMAND ----------


#businesscontext_Agent= AssistantAgent(
#        name="BusinessContext_Agent"
#        ,description="To understand the context described by the user"
#        ,system_message="You are a AI Assitant that understands the context provided by the user. Provide a brief summary of the understood context"
#        ,llm_config={"config_list": config_list_params})

# COMMAND ----------

import mlflow

#grab the model URI that's generated from the run
model_uri = f"runs:/{last_run_id}/{model_name}"

#log the model to catalog.schema.model. The schema name referenced below is generated for you in the init script
catalog = dbutils.widgets.get("catalog_name")
schema = schema_name

#set our registry location to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=f"{catalog}.{schema}.{model_name}"
)
