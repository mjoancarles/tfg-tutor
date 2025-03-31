from neo4j import GraphDatabase
import requests
import os
import re
import streamlit as st
from langchain.chat_models import init_chat_model
from interference_api_embeddings import InferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore
import logging
import time
from langchain_core.documents import Document
from langchain_core.prompts.base import format_document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from operator import add
from typing import Literal, Annotated, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from unidecode import unidecode
from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

# Initialize the LLM
def init_llm():
    return init_chat_model(
        model=os.getenv("LLM_MODEL"),
        model_provider="openai",
        api_key="empty",
        base_url=os.getenv("LLM_HOST"),
        temperature=os.getenv("LLM_TEMPERATURE"),
        max_tokens=os.getenv("LLM_MAX_TOKENS"),
    )

def init_embeddings():
    return InferenceAPIEmbeddings()
    
def neo4j_get_session():
    AUTH = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    neo4j_driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=AUTH)
    neo4j_session = neo4j_driver.session(database="neo4j")
    return neo4j_session

def get_qdrant_vector_store(embeddings):
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = os.getenv("QDRANT_PORT")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=qdrant_collection,
        url=f"http://{qdrant_host}:{qdrant_port}",
        content_payload_key = 'page_content',
        metadata_payload_key = 'metadata'
    )
    
def generate_session_id():
    # Generates if based on timestamp
    return str(int(time.time()))

# Utility function to check the status of a Qdrant collection using requests
def check_qdrant():
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = os.getenv("QDRANT_PORT")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an error for unsuccessful status codes
        col_info = response.json()
        print(col_info)
        return col_info["result"]["status"] == "green"
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

# Utility function to check FastAPI health
def check_fastapi_health(type = "llm"):
    if type == "llm":
        FASTAPI_HOST = os.getenv("LLM_HOST")
        FASTAPI_PORT = os.getenv("LLM_PORT")
    if type == "embeddings":
        FASTAPI_HOST = os.getenv("EMBEDDINGS_HOST")
        FASTAPI_PORT = os.getenv("EMBEDDINGS_PORT")
    url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/health"
    
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        raise ConnectionError(f"Failed to connect to FastAPI: {str(e)}")

def check_connections():
    try:
        if not check_qdrant():
            st.error("Qdrant is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return

    '''try:
        if not check_fastapi_health(type="llm"):
            st.error("LLM FastAPI is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return'''

    try:
        if not check_fastapi_health(type="embeddings"):
            st.error("Embeddings FastAPI is not healthy. Please check the server.")
            return
    except ConnectionError as e:
        st.error(str(e))
        return

def format_semantic_entities(semantic_entities: dict, candidates: int) -> str:
    """
    Returns a human-readable string of relevant keywords and authors
    from the semantic_entities dictionary, ignoring scores.
    Only the top `candidates` items from each category will be included.
    """
    if not semantic_entities:
        return ""

    # Slice the lists to include only the top `candidates`
    keywords = [item["candidate"] for item in semantic_entities.get("Keyword", [])][:candidates]
    people = [item["candidate"] for item in semantic_entities.get("Person", [])][:candidates]

    output_lines = []

    if keywords:
        output_lines.append("Relevant Keywords for the user query:")
        output_lines.append(", ".join(keywords))
        output_lines.append("")  # blank line

    if people:
        output_lines.append("Relevant Authors for the user query:")
        output_lines.append(", ".join(people))
        output_lines.append("")  # blank line

    return "\n".join(output_lines).strip()

def get_rag_graph(llm, vector_store):
    template = """
    General Info:
    You are an assistant answering questions about TFGs (Final Degree Projects by Catalan students) and CVC scientific publications. The CVC is a non-profit research center established in 1995 by the Generalitat de Catalunya and the UAB, focusing on computer vision research and collaborating on TFGs.

    Instructions:

    -When Responding:
    Only include retrieved documents if they are relevant to the query. If no relevant documents are found, answer directly.
    -Document Formatting:
    For Publications:
        Title: (publication title)
        - Authors: (comma-separated list)
        - Abstract: (summarized abstract)
        - Published in: (year)
        - Link: (if available)
    For TFGs:
        Title: (TFG title)
        - Authors: (comma-separated list)
        - Abstract: (summarized abstract)
        - Delivered in: (year)
        - Link: (if available)
    Exclude any field that is NULL/EMPTY. Show only the top 5 most relevant results by default (unless the user requests more). After listing, include a brief explanation and comparison of the documents, highlighting key details in bold.
    Direct Answers:
    If the query does not require listing documents, answer directly without showing retrieved documents. Always answer in english.
    ------------------------------
    Retrieved documents: {context}
        """
    prompt = PromptTemplate.from_template(template)

    document_prompt = PromptTemplate.from_template(
                "Title: {title}\n"
                "Type: {type}\n"
                "Abstract: {page_content}\n"
                "Authors: {authors}\n"
                "Year: {year}\n"
                "Keywords: {keywords}\n"
                "Link: {link}\n"
    )

    # https://python.langchain.com/docs/tutorials/rag/
    class State(TypedDict):
        messages: List[BaseMessage]
        context: List[Document]

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["messages"][-1].content)
        return {"context": retrieved_docs}

    def generate(state: State):
        messages = state["messages"]
        print(messages)
        #docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        docs_content = "\n\n".join(format_document(doc, document_prompt) for doc in state["context"])
        system_prompt = prompt.invoke({"context": docs_content}).to_string()
        #print(system_prompt)
        messages = [SystemMessage(system_prompt)] + messages
        print(messages)
        response = llm.invoke(messages)
        #print(type(response)) # <class 'langchain_core.messages.ai.AIMessage'>
        return {"messages": [response]}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("generate", END)
    graph = graph_builder.compile()
    return graph

def get_agentic_rag_graph(llm, neo4j_session, embeddings, enhanced_graph, examples_content):
    
    # ------------------------------------------------------------------------------------
    # STATE DEFINITIONS
    # ------------------------------------------------------------------------------------
    class InputState(TypedDict):
        messages: List[BaseMessage]

    class OverallState(TypedDict):
        messages: List[BaseMessage]
        next_action: str
        cypher_statement: str
        cypher_errors: List[str]
        number_of_corrections: int
        database_records: List[dict]
        steps: Annotated[List[str], add]
        semantic_entities: List[Dict[str, str]] # e.g. [{"candidate": "Adria Molina", "label": "Person"}, ...]

    class OutputState(TypedDict):
        answer: str
        steps: List[str]
        cypher_statement: str
        
    # ------------------------------------------------------------------------------------
    # GUARDRAILS NODE
    # ------------------------------------------------------------------------------------

    # Updated system prompt for academic guidance
    guardrails_system = """
    Decide if a question is about academic guidance for final university projects (TFG), professors/students, or publications—or if a database query might help. 
    Default to "academic" unless it is clearly unrelated; then output "end".
    """

    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",guardrails_system),
            ("human",("{question}")),
        ]
    )

    class GuardrailsOutput(BaseModel):
        decision: Literal["academic", "end"] = Field(
            description="Decision on whether the question is related to academic research guidance"
        )

    # https://python.langchain.com/docs/how_to/structured_output/#the-with_structured_output-method
    # https://docs.vllm.ai/en/latest/features/structured_outputs.html
    guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

    def guardrails(state: InputState) -> OverallState:
        """
        Decides if the question is related to academic research guidance or not.
        """
        logging.info("Guardrails node")
        guardrails_output = guardrails_chain.invoke({"question": state.get("messages")[-1].content})
        database_records = None
        if guardrails_output.decision == "end":
            database_records = "This question is not related to academic research guidance. Therefore, I cannot answer this question."
        
        return {
            "next_action": guardrails_output.decision,
            "database_records": database_records,
            "steps": ["guardrail"],
            "semantic_entities": [],
        }

    # ------------------------------------------------------------------------------------
    # SEMANTIC LAYER NODE
    # ------------------------------------------------------------------------------------
    def sanitize_query(query: str) -> str:
        """
        Escapes special characters in a Lucene query to prevent parsing errors.
        The list below includes common Lucene special characters.
        """
        # Define a regex pattern for common Lucene special characters
        pattern = r'([\+\-\!\(\)\{\}\[\]\^"~\*\?:\\\/%])'
        sanitized = re.sub(pattern, r'\\\1', query)
        # If the sanitized query is empty or only whitespace, default to an empty string
        if not sanitized.strip():
            sanitized = ""
        return sanitized

    def semantic_layer(state: OverallState) -> OverallState:
        """
        Extracts semantic entities from the user's question.
        """
        logging.info("Semantic Layer node")
        # Get and embed the user query
        user_query = unidecode(state.get("messages")[-1].content)
        query_vector = embeddings.embed_query(user_query)
        semantic_entities = {}
        
        # Query top 5 Keyword candidates based on keyword embeddings
        result_kw = neo4j_session.run("""
            CALL db.index.vector.queryNodes('keywordNameIndex', 6, $v)
            YIELD node, score
            RETURN node.keyword AS candidate, score
            ORDER BY score DESC
        """, v=query_vector).data()
        semantic_entities["Keyword"] = result_kw
        
        # Sanitize the query for the full-text search
        sanitized_query = sanitize_query(user_query)
        
        # For Person, using a full-text index with the sanitized query
        result_person = neo4j_session.run("""
            CALL db.index.fulltext.queryNodes("personNames", $q) YIELD node, score
            RETURN node.name AS candidate, score
            ORDER BY score DESC
            LIMIT 6
        """, q=sanitized_query).data()
        semantic_entities["Person"] = result_person
        
        logging.info(f"Semantic entities: {semantic_entities}")
        
        return {
            "steps": ["semantic_layer"],
            "semantic_entities": format_semantic_entities(semantic_entities, 6),
        }

    # ------------------------------------------------------------------------------------
    # EXAMPLES + TEXT2CYPHER NODE
    # ------------------------------------------------------------------------------------

    examples = examples_content["examples"]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, embeddings, Neo4jVector, k=5, input_keys=["question"]
    )

    text2cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    """Given an input question, convert it to a Cypher query. Avoid Cartesian products by ensuring that all MATCH statements are logically connected. Use OPTIONAL MATCH only when necessary, and avoid unnecessary duplicate entity retrieval. Ensure that queries retrieve only the required entities without creating excessive joins. Respond with a Cypher statement only, without any additional formatting or commentary."""
                ),
            ),
            (
                "human",
                (
                    """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run. Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only! Avoid Cartesian products by ensuring that nodes are connected logically. For example, if retrieving publications and their authors, ensure that they are linked via relationships. Use OPTIONAL MATCH sparingly and only where relationships might not exist. If named entities do not use special characters like à, è, ò, etc., use the regular characters a, e, o, etc., when writing the Cypher query. Here is the graph schema information:
    {schema}

    Below are a number of examples of questions and their corresponding Cypher queries:
    {fewshot_examples}

    User input (that MUST be solved):
    {question}

    If asked for who might supervise a TFG related to X, please use keywords and try finding the authors that did publications that conatins them, for instance.

    These are the semantic entities extracted from the user's question just by using embeddings. Use those author and keyword names to create the Cypher query whenever needed. Please only use the entities that are relevant to the user's question; if the question asks only for a person, do not include keywords for instance.
    {semantic_entities}

    Past chat history, only to be used if relevant to the user query; if not, please ignore:
    {history}

    Try including OPTIONAL MATCH in the query to retrieve additional information if available. When asked about publications only, do not include TFGs in the query and vice versa.

    Make sure to LIMIT the responses if asking questions that might return lots of results.

    Cypher query:"""
                ),
            ),
        ]
    )

    text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------------------------
    # GENERATE CYPHER
    # ------------------------------------------------------------------------------------

    def generate_cypher(state: OverallState) -> OverallState:
        """
        Generates a cypher statement based on the provided schema and user input
        """
        logging.info("Generate Cypher node")
        NL = "\n"
        fewshot_examples = (NL * 2).join(
            [
                f"Question: {el['question']}{NL}Cypher:{el['query']}"
                for el in example_selector.select_examples(
                    {"question": state.get("messages")[-1].content}
                )
            ]
        )
        history = "\n".join([(message.type + ": " + message.content) for message in state.get("messages")[1:]])
        semantic_context = state.get("semantic_entities")
        generated_cypher = text2cypher_chain.invoke(
            {
                "question": state.get("messages")[-1].content,
                "history": history,
                "semantic_entities": semantic_context,
                "fewshot_examples": fewshot_examples,
                "schema": enhanced_graph.schema,
            }
        )
        logging.info(f"Generated Cypher: {generated_cypher}")
        return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

    # ------------------------------------------------------------------------------------
    # QUERY VALIDATION
    # ------------------------------------------------------------------------------------

    # Cypher query corrector is experimental
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in enhanced_graph.structured_schema.get("relationships")
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    def validate_cypher(state: OverallState) -> OverallState:
        """
        Simplified validation: Use EXPLAIN to catch syntax errors and fix relationship directions.
        If an error is detected, set next_action to "correct_cypher"; otherwise, proceed to execution.
        """
        errors = []
        
        # Check for syntax errors using EXPLAIN to catch CypherSyntaxError exceptions
        try:
            enhanced_graph.query(f"EXPLAIN {state.get('cypher_statement')}")
        except CypherSyntaxError as e:
            logging.error(f"Syntax error: {e}")
            errors.append(e.message)
        
        # Correct relationship directions if necessary
        corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
        if not corrected_cypher:
            errors.append("The generated Cypher statement doesn't fit the graph schema")
        if corrected_cypher != state.get("cypher_statement"):
            print("Relationship direction was corrected")
            
        corrections = state.get("number_of_corrections", 0) + 1
        
        # Decide next action based on errors
        if corrections > 3:
            next_action = "execute_cypher"
        else:
            next_action = "correct_cypher" if errors else "execute_cypher"
        
        return {
            "next_action": next_action,
            "number_of_corrections": corrections,
            "cypher_statement": corrected_cypher,
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
        }

    # ------------------------------------------------------------------------------------
    # CORRECT CYPHER QUERY
    # ------------------------------------------------------------------------------------
        
    correct_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Cypher expert reviewing a statement written by a junior developer. "
                    "Your task is to correct the Cypher statement based on the provided errors. "
                    "Do not include any preamble, explanations, or apologies. "
                    "Respond with a Cypher statement only, without any backticks or additional formatting."
                ),
            ),
            (
                "human",
                (
                    """Check for invalid syntax or semantics and return a corrected Cypher statement.

    Schema:
    {schema}

    Note: The corrected Cypher query MUST be different from the original query and must fix the mentioned error. 
    Do not include any explanations or commentary.
    Do not respond to any requests other than constructing a Cypher statement.

    The question is:
    {question}

    Past chat history (only to be considered if relevant):
    {history}

    The Cypher statement is:
    {cypher}

    The errors are:
    {errors}

    Corrected Cypher statement: """
                ),
            ),
        ]
    )

    correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()


    def correct_cypher(state: OverallState) -> OverallState:
        """
        Correct the Cypher statement based on the provided errors.
        """
        history = "\n".join([(message.type + ": " + message.content) for message in state.get("messages")[1:]])
        question = state.get("messages")[-1].content
        corrected_cypher = correct_cypher_chain.invoke(
            {
                "question": question,
                "history": history,
                "errors": state.get("cypher_errors"),
                "cypher": state.get("cypher_statement"),
                "schema": enhanced_graph.schema,
            }
        )
        
        logging.info(f"Corrected Cypher: {corrected_cypher}")
        
        return {
            "next_action": "validate_cypher",
            "cypher_statement": corrected_cypher,
            "steps": ["correct_cypher"],
        }

    # ------------------------------------------------------------------------------------
    # EXECUTE CYPHER
    # ------------------------------------------------------------------------------------

    no_results = "I couldn't find any relevant information in the database"

    def execute_cypher(state: OverallState) -> OverallState:
        """
        Executes the given Cypher statement.
        """
        logging.info("Execute Cypher node")
        records = enhanced_graph.query(state.get("cypher_statement"))
        return {
            "database_records": records if records else no_results,
            "next_action": "end",
            "steps": ["execute_cypher"],
        }

    # ------------------------------------------------------------------------------------
    # GENERATE FINAL ANSWER
    # ------------------------------------------------------------------------------------

    generate_final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are an assistant specialized in TFGs (Final Degree Projects by Catalan students) 
    and CVC scientific publications. The CVC is a non-profit research center established in 1995 
    by the Generalitat de Catalunya and the UAB, focusing on computer vision research 
    and collaborating on TFGs.

    Your role is to help students seeking academic guidance on TFGs and publications.

    - Only show relevant documents if found; otherwise, answer directly.
    - By default, return only the top 5 results (unless the user requests more or we find less than 5).
    - Summarize briefly, highlighting key details in **bold**.
    - Always answer in English.
                """,
            ),
            (
                "human",
                """
    Generate a final response for the following information

    Provide an answer to the user question:
    {question}

    Neo4j query results found:
    {results}

    First initially extracted semantic entities (only to be used in case of no results, and if relevant to the user query, if not please ignore: for instance if asking for a person, do not include retrieved keywords or publications):
    {semantic_entities}

    Past history:
    {history}
                """,
            ),
        ]
    )

    generate_final_chain = generate_final_prompt | llm | StrOutputParser()

    def generate_final_answer(state: OverallState) -> OutputState:
        """
        Generates the final answer, returning relevant documents if they exist,
        or a direct response if none are found or needed.
        """
        logging.info("Generate Final Answer node")
        history = "\n".join([(message.type + ": " + message.content) for message in state.get("messages")[1:]])
        semantic_context = state.get("semantic_entities")
        logging.info("GENERATE FINAL ANSWER")
        logging.info(f"Semantic entities: {semantic_context}")
        logging.info(f"Database records: {state.get('database_records')}")
        logging.info(f"History: {history}")
        logging.info(f"Question: {state.get('messages')[-1].content}")
        
        final_answer = generate_final_chain.invoke(
            {
                "question": state.get("messages")[-1].content,
                "results": str(state.get("database_records"))[:10000],
                "semantic_entities": semantic_context,
                "history": history,
            }
        )
        return {"answer": final_answer,
                "steps": ["generate_final_answer"],
                "cypher_statement": state.get("cypher_statement")}

    # ------------------------------------------------------------------------------------
    # STATEGRAPH
    # ------------------------------------------------------------------------------------

    # From https://python.langchain.com/docs/tutorials/graph/

    def guardrails_condition(
        state: OverallState,
    ) -> Literal["semantic_layer", "generate_final_answer"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "academic":
            return "semantic_layer"

    def validate_cypher_condition(
        state: OverallState,
    ) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "correct_cypher":
            return "correct_cypher"
        elif state.get("next_action") == "execute_cypher":
            return "execute_cypher"

    langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
    langgraph.add_node(guardrails)
    langgraph.add_node(semantic_layer)
    langgraph.add_node(generate_cypher)
    langgraph.add_node(validate_cypher)
    langgraph.add_node(correct_cypher)
    langgraph.add_node(execute_cypher)
    langgraph.add_node(generate_final_answer)

    langgraph.add_edge(START, "guardrails")
    langgraph.add_conditional_edges("guardrails",guardrails_condition,)
    langgraph.add_edge("generate_cypher", "validate_cypher")
    langgraph.add_conditional_edges("validate_cypher",validate_cypher_condition,)
    langgraph.add_edge("semantic_layer", "generate_cypher")
    langgraph.add_edge("execute_cypher", "generate_final_answer")
    langgraph.add_edge("correct_cypher", "validate_cypher")
    langgraph.add_edge("generate_final_answer", END)

    graph = langgraph.compile()
    return graph