import streamlit as st
from langchain_neo4j import Neo4jGraph, Neo4jVector
import os
import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from operator import add
from typing_extensions import TypedDict
from typing import Literal, Annotated, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import StrOutputParser
from unidecode import unidecode
from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from utils import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------------
# STREAMLIT / LANGCHAIN / NEO4J SETUP 
# ------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG",
    page_icon=":desktop_computer:",
    layout="centered",
    initial_sidebar_state="collapsed", # To be changed if having filters!
)

st.title("Agentic RAG - TFG Tutor Chatbot")

# --- Enforce Login ---
_ = '''if not st.session_state.get("logged_in", False):
    st.switch_page("Welcome.py")
    st.stop()'''

llm = init_llm()
embeddings = init_embeddings()
neo4j_session = neo4j_get_session()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

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
As an intelligent assistant, your primary objective is to decide whether a given question is related to academic research guidance for students preparing their Treball de Fi de Grau (TFG).
If the question is relevant to this topic, output "academic". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to:
- Searching for relevant TFGs, academic publications, or keywords.
- Finding potential tutors or researchers based on their previous work.
- Finding information about possible researchers and their publications.
- Finding about someone's information, because they are a potential tutor.
- Identifying academic connections between students, investigators, publications, and keywords.
- Requesting insights into specific research topics or fields of study.
In case of doubt, redirect to the academic guidance path always.
Provide only the specified output: "academic" or "end".
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
    guardrails_output = guardrails_chain.invoke({"question": state.get("messages")[0].content})
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
def semantic_layer(state: OverallState) -> OverallState:
    """
    Extracts semantic entities from the user's question.
    """
    logging.info("Semantic Layer node")
    user_query = unidecode(state.get("messages")[0].content)
    query_vector = embeddings.embed_query(user_query)
    semantic_entities = {}
     
    _ = '''# Query top 5 TFG candidates based on abstract embeddings
    result_tfg = neo4j_session.run("""
        CALL db.index.vector.queryNodes('tfgAbstractIndex', 5, $v)
        YIELD node, score
        RETURN node.title AS candidate, score
        ORDER BY score DESC
    """, v=query_vector).data()
    semantic_entities["TFG"] = result_tfg
    
    # Query top 5 Publication candidates based on abstract embeddings
    result_pub = neo4j_session.run("""
        CALL db.index.vector.queryNodes('pubAbstractIndex', 5, $v)
        YIELD node, score
        RETURN node.title AS candidate, score
        ORDER BY score DESC
    """, v=query_vector).data()
    semantic_entities["Publication"] = result_pub'''
    
    # Query top 5 Keyword candidates based on keyword embeddings
    result_kw = neo4j_session.run("""
        CALL db.index.vector.queryNodes('keywordNameIndex', 6, $v)
        YIELD node, score
        RETURN node.keyword AS candidate, score
        ORDER BY score DESC
    """, v=query_vector).data()
    semantic_entities["Keyword"] = result_kw
    
    # For Person, using a full-text index
    result_person = neo4j_session.run("""
        CALL db.index.fulltext.queryNodes("personNames", $q) YIELD node, score
        RETURN node.name AS candidate, score
        ORDER BY score DESC
        LIMIT 6
    """, q=user_query).data()
    semantic_entities["Person"] = result_person
    
    logging.info(f"Semantic entities: {semantic_entities}")
    
    return {
        "steps": ["semantic_layer"],
        "semantic_entities": semantic_entities,
    }

# ------------------------------------------------------------------------------------
# EXAMPLES + TEXT2CYPHER NODE
# ------------------------------------------------------------------------------------

# Get the absolute path of the current script
script_dir = os.path.dirname(__file__)  # This is /app/pages

# Construct the path to query_examples.json (which is at /app)
json_path = os.path.join(script_dir, "..", "query_examples.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

examples = data["examples"]

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
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run. Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only! Avoid Cartesian products by ensuring that nodes are connected logically. For example, if retrieving publications and their authors, ensure that they are linked via relationships. Use OPTIONAL MATCH sparingly and only where relationships might not exist. If named entities do not use special characters like à, è, ò, etc., use the regular characters a, e, o, etc., when writing the Cypher query. Here is the graph schema information: {schema}

Below are a number of examples of questions and their corresponding Cypher queries: {fewshot_examples}

User input (that MUST be solved): {question}

If asked for who might supervise a TFG related to X, please use keywords and try finding the authors that did publications that conatins them, for instance.

These are the semantic entities extracted from the user's question just by using embeddings. Use those author and keyword names to create the Cypher query whenever needed. Please only use the entities that are relevant to the user's question; if the question asks only for a person, do not include keywords. {semantic_entities}

Past chat history, only to be used if relevant to the user query; if not, please ignore: {history}

Try including OPTIONAL MATCH in the query to retrieve additional information if available. When asked about publications only, do not include TFGs in the query and vice versa.

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
                {"question": state.get("messages")[0].content}
            )
        ]
    )
    history = "\n".join([(message.type + ": " + message.content) for message in state.get("messages")[1:]])
    semantic_context = format_semantic_entities(state.get("semantic_entities"))
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("messages")[0].content,
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
    question = state.get("messages")[0].content
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
- By default, return only the top 5 results (unless the user requests more).
- Summarize briefly, highlighting key details in **bold**.
- Always answer in English.
- If no relevant results are found, still provide a helpful academic response based on your knowledge.
            """,
        ),
        (
            "human",
            """
Given the user's question, retrieved results from the neo4j database, and the chat history,
please generate a final response for the user.

Question:
{question}

Neo4j query results:
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
    semantic_context = format_semantic_entities(state.get("semantic_entities"))
    final_answer = generate_final_chain.invoke(
        {
            "question": state.get("messages")[0].content,
            "results": state.get("database_records"),
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

# ------------------------------------------------------------------------------------
# MAIN STREAMLIT APP
# ------------------------------------------------------------------------------------

def main():
    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build pipeline_messages list from the chat history
    pipeline_messages = []
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            pipeline_messages.append(HumanMessage(chat["message"]))
        elif chat["role"] == "assistant":
            pipeline_messages.append(AIMessage(chat["message"]))

    # Display previous chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    # Chatbot interface using st.chat_input and st.chat_message
    user_input = st.chat_input("Ask your question about TFGs or publications")
    if user_input:
        # Append user's message to chat history and pipeline messages
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        pipeline_messages.append(HumanMessage(user_input))

        # Process the user's question through the retrieval/generation pipeline
        config = {"configurable": {"thread_id": "1"}}

        # Variable to accumulate the complete assistant response
        complete_response = ""

        def generate_streaming_message():
            nonlocal complete_response
            # Pass the full conversation history into the pipeline
            for message, metadata in graph.stream(
                {"messages": pipeline_messages},
                stream_mode="messages",
                config=config
            ):
                # Ensure only user-facing content is included
                if metadata["langgraph_node"] == "generate_final_answer":
                    complete_response += message.content
                    yield message.content

        with st.chat_message("assistant"):
            st.write_stream(generate_streaming_message())
        
        # Log the complete response
        logging.info(f"Complete response: {complete_response}")

        # Append the complete assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": complete_response})

if __name__ == "__main__":
    # Log start of the app
    logging.info("Starting Agentic RAG chatbot --------------------------------------------------")
    main()
