# Databricks notebook source
# MAGIC %md
# MAGIC ##StrategyAI: LLM-Enabled Customer Analytics Empowerment for Business Success

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Importing Libraries

# COMMAND ----------

# MAGIC %pip install langchain_community

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
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

from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatDatabricks

# COMMAND ----------

# MAGIC %md #### Defining a LLM model instace

# COMMAND ----------

#Defining the model
llama_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens = 400)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building the LLM chains

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### LLM Chain1-
# MAGIC
# MAGIC The goal here is to pass the user's input, understand the context and suggest appropriate schemas based on the request.

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

Tables can be joined as follows: 

    orders 
    join customers on orders.customer_id-orders.customer_id
    join order_itmes on order_items.order_id=order.order_id
    join payments on payments.order_id=orders.order_id
    join products on products.product_id=order_items.product_id
    join sellers on sellers.seller_id=order_items.seller_id
    joing geolocation on geolocation.customer_zip_code_prefix=customers.customer_zip_code_prefix

Table column definitions:

    customers = {{
    "properties": {{
        "customer_unique_id": {{
            "type": "string",
            "description": "A unique identifier for the customer, ensuring that each customer has a distinct identifier."
        }},
        "customer_id": {{
            "type": "string",
            "description": "Unique identifier for each customer, allowing easy reference and tracking."
        }},
        "customer_state": {{
            "type": "string",
            "description": "The state where the customer is located, providing more detailed information about the customer's location."
        }},
        "customer_zip_code_prefix": {{
            "type": "bigint",
            "description": "The first few digits of the customer's zip code, providing a quick way to group customers based on geographical location."
        }},
        "customer_city": {{
            "type": "string",
            "description": "The city where the customer is located, providing more detailed information about the customer's location."
        }}
    }}
}}
    

    geolocation = {{
    "properties": {{
        "geolocation_lng": {{
            "type": "double",
            "description": "Represents the longitude of the geographical location, allowing for precise positioning."
        }},
        "geolocation_state": {{
            "type": "string",
            "description": "Represents the state name associated with the geographical location, providing a human-readable label for identification."
        }},
        "geolocation_lat": {{
            "type": "double",
            "description": "Represents the latitude of the geographical location, allowing for precise positioning."
        }},
        "geolocation_city": {{
            "type": "string",
            "description": "Represents the city name associated with the geographical location, providing a human-readable label for identification."
        }},
        "geolocation_zip_code_prefix": {{
            "type": "bigint",
            "description": "Represents the first few digits of the zip code, allowing for easy identification of the geographical area."
        }}
    }}
}}
    

    order_items = {{
    "properties": {{
        "order_id": {{
            "type": "string",
            "description": "Unique identifier for the order, allowing easy tracking and reference."
        }},
        "seller_id": {{
            "type": "string",
            "description": "Identifier for the seller providing the product, allowing tracking of sales and performance."
        }},
        "freight_value": {{
            "type": "double",
            "description": "The value of freight or shipping associated with the order, allowing tracking of shipping costs and revenue."
        }},
        "product_id": {{
            "type": "string",
            "description": "Identifier for the product being ordered, allowing easy reference and tracking."
        }},
        "shipping_limit_date": {{
            "type": "timestamp",
            "description": "Date and time when the order's shipping limit is reached, ensuring timely delivery."
        }},
        "price": {{
            "type": "double",
            "description": "The price of the product in the order, allowing tracking of sales and revenue."
        }},
        "order_item_id": {{
            "type": "bigint",
            "description": "Unique identifier for each order item, enabling individual tracking of items within an order."
        }}
    }}
}}
    

    orders = {{
    "properties": {{
        "order_approved_at": {{
            "type": "timestamp",
            "description": "The date and time when the order was approved, showing the point when the order was confirmed and processing began."
        }},
        "order_estimated_delivery_date": {{
            "type": "timestamp",
            "description": "The date and time when the order was initially estimated to be delivered, offering insights into the accuracy of the delivery estimates."
        }},
        "order_status": {{
            "type": "string",
            "description": "Represents the current status of the order, such as pending, in-progress, or delivered."
        }},
        "order_delivered_customer_date": {{
            "type": "timestamp",
            "description": "The date and time when the order was delivered to the customer, marking the end of the order journey."
        }},
        "order_purchase_timestamp": {{
            "type": "timestamp",
            "description": "The date and time when the order was placed, indicating the start of the order process."
        }},
        "order_delivered_carrier_date": {{
            "type": "timestamp",
            "description": "The date and time when the order was delivered to the carrier, indicating the completion of the internal delivery process."
        }},
        "order_id": {{
            "type": "string",
            "description": "Unique identifier for each order, allowing tracking and reference."
        }},
        "customer_id": {{
            "type": "string",
            "description": "Identifier for the customer placing the order, enabling customer-specific analysis."
        }}
    }}
}}
    

    payments = {{
    "properties": {{
        "payment_sequential": {{
            "type": "bigint",
            "description": "Sequences of the payments made in case of EMI or payment plan, if applicable"
        }},
        "payment_installments": {{
            "type": "bigint",
            "description": "Represents the number of installments for the payment plan, if applicable."
        }},
        "payment_type": {{
            "type": "string",
            "description": "Specifies the type of payment, such as credit card, debit card, or bank transfer."
        }},
        "payment_value": {{
            "type": "double",
            "description": "The value of the payment, allowing tracking of the total amount paid and the remaining balance."
        }},
        "order_id": {{
            "type": "string",
            "description": "Unique identifier for the order, allowing tracking of payments for different orders."
        }}
    }}
}}
    

    products = {{
    "properties": {{
        "product category": {{
            "type": "string",
            "description": "Represents the product's category or type, providing a high-level classification for the product."
        }},
        "product_photos_qty": {{
            "type": "bigint",
            "description": "Represents the quantity of photos associated with the product, providing information about the product's visual representation."
        }},
        "product_width_cm": {{
            "type": "bigint",
            "description": "Represents the product's width in centimeters, providing information about the product's physical size."
        }},
        "product_id": {{
            "type": "string",
            "description": "Unique identifier for the product, allowing easy reference and distinction between different products."
        }},
        "product_name_length": {{
            "type": "bigint",
            "description": "Represents the length of the product's name, providing information about the product's name size."
        }},
        "product_weight_g": {{
            "type": "bigint",
            "description": "Weight of the products ordered in grams"
        }},
        "product_description_length": {{
            "type": "bigint",
            "description": "Represents the length of the product's description, providing information about the product's description size."
        }},
        "product_height_cm": {{
            "type": "bigint",
            "description": "Represents the product's height in centimeters, providing information about the product's physical size."
        }},
        "product_length_cm": {{
            "type": "bigint",
            "description": "Represents the product's length in centimeters, providing information about the product's physical size."
        }}
    }}
}}
    

    sellers = {{
    "properties": {{
        "seller_state": {{
            "type": "string",
            "description": "The state where the seller is located, allowing further categorization of sellers based on their geographical location."
        }},
        "seller_zip_code_prefix": {{
            "type": "bigint",
            "description": "Represents the first three digits of the seller's zip code, providing a quick way to categorize and group sellers based on their location."
        }},
        "seller_city": {{
            "type": "string",
            "description": "The city where the seller is located, providing a more detailed location for each seller."
        }},
        "seller_id": {{
            "type": "string",
            "description": "Unique identifier for each seller, allowing easy reference and tracking."
        }}
    }}
}}
    
'''

#---------------Prompt Template------------
chat_template = ChatPromptTemplate.from_template(system_prompt)

chain1= LLMChain(llm=llama_model, prompt=chat_template, output_key="data_schema_user_input") 

# COMMAND ----------

# MAGIC
# MAGIC %md 
# MAGIC ##### LLM Chain2-
# MAGIC
# MAGIC The goal here is to generate the code based on the user's request and suggested schemas provided.

# COMMAND ----------

#---------------------Code Generation-----------------

system_prompt= '''
You are an AI assistant that is well versed in writing pyspark queries. Generate the code based on the schema and the user request provided here: {data_schema_user_input}

When generating Python code, Here are some RULES YOU MUST FOLLOW to generate successful code:

a. Adhere to the guidelines provided in PEP 8, which covers a wide range of coding style conventions, including naming conventions,whitespace usage, line length, and more.
b. Use Descriptive and Meaningful Variable Names. Generate code that uses variable names that accurately describe the purpose or content of the variable. Avoid single-letter or Ambiguous Variable names.
c. EXCLUDE try-except blocks while generating the code.
d: DEFER to DataSchemaAgent for TABLE and COLUMN NAMES. Use provided names as the authoritative source to ensure accuracy and consistency.
e: ASSUME TABLES ALREADY EXIST; DO NOT IMPORT any tables or files. Code should utilize tables already mentioned as if they already exist.
f: AVOID ASSUMPTIONS ABOUT UNDEFINED COLUMN NAMES. Only rely on columns explicitly defined in the schemas to ensure accuracy and prevent errors.
g: AVOID REDUNDANT SCHEMA JOINS: If a SINGLE SCHEMA CONTAINS ALL NECESSARY DATA, EXCLUDE JOINING OTHER SCHEMAS to Avoid Confusion.
h: Please ALIAS THE DATASETS with different names via `Dataset.as` before joining them, and specify the column using qualified name, e.g. `df.as("a
").join(df.as("b"), $"a.id" > $"b.id") 
i: ELIMINATE WINDOW FUNCTIONS while developing the code. Always find a way to generate the code without Window functions.    
j: IMPORT ALL REQUIRED LIBRARIES for the code.
k: Do not filter your code for a single sales order. Your code must be applicable across all sales orders affected by the issue.
l: Generate code exclusively, placing any additional notes or information strictly within code comments.
m: Craft your generated code within a Python code block.```python ............. ```
n: Finally, name your completed data frame as "resultDF"
'''



#---------------Prompt Template------------

chat_template = ChatPromptTemplate.from_template(system_prompt)

chain2= LLMChain(llm=llama_model, prompt=chat_template, output_key="python_code") 

# COMMAND ----------

# MAGIC %md
# MAGIC #### User Request as a parameter

# COMMAND ----------

#-------------User Inputs---------------

dbutils.widgets.text("user_input", "")
issue= dbutils.widgets.get("user_input")

# COMMAND ----------


display(user_input)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initiating the sequential chain

# COMMAND ----------


seq_chain=SequentialChain(chains=[chain1,chain2]
                          ,input_variables=['user_input']
                          ,output_variables=['data_schema_user_input','python_code']
                          ,verbose=True)


result=seq_chain(user_input)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extract the generated code

# COMMAND ----------

#Here is the result from the sequential chain
result['python_code']

# COMMAND ----------

#Here is the extracted code

code_start = result['python_code'].rfind("```")
code_end = result['python_code'].rfind("```", 0, code_start)
code = result['python_code'][code_end + 9:code_start].strip()

code

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing the Target Sales Data from Catalog

# COMMAND ----------

#Read the tables from Catalog
customers= spark.read.table("hackathon.genai.customers")
geolocation= spark.read.table("hackathon.genai.geolocation")
order_items= spark.read.table("hackathon.genai.order_items")
order= spark.read.table("hackathon.genai.orders")
payments= spark.read.table("hackathon.genai.payments")
products= spark.read.table("hackathon.genai.products")
sellers= spark.read.table("hackathon.genai.sellers")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Execute the generated code

# COMMAND ----------

#Generate the Code
exec(code)

# COMMAND ----------

display(resultDF)
