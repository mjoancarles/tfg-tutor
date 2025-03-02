import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph, Neo4jVector
import requests
from langchain.embeddings.base import Embeddings
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from operator import add
from typing_extensions import TypedDict
from typing import Literal, Annotated, List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------------
# STREAMLIT / LANGCHAIN SETUP
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

class InferenceAPIEmbeddings(Embeddings):
    
    def __init__(self):
        EMBEDDINGS_HOST = os.getenv("EMBEDDINGS_HOST", "localhost")
        EMBEDDINGS_PORT = os.getenv("EMBEDDINGS_PORT", "8099")
        self.api_url = f"http://{EMBEDDINGS_HOST}:{EMBEDDINGS_PORT}/generate_embeddings/"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"sentences": texts}  # Updated to match FastAPI's expected payload structure
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            return response.json()["embeddings"]  # Return the entire list of embeddings
        else:
            raise ValueError(f"Failed to get embeddings: {response.status_code} - {response.text}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]  # Embed a single query and return its embedding

# Initialize the LLM
def init_llm():
    return init_chat_model(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_provider="openai",
        api_key="empty",
        base_url="http://host.docker.internal:8111/v1",
        temperature=0.2,
        max_tokens=700,
    )
    
llm = init_llm()
embeddings = InferenceAPIEmbeddings()

os.environ["NEO4J_URI"] = "bolt://host.docker.internal:7300"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

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
- Identifying academic connections between students, investigators, publications, and keywords.
- Requesting insights into specific research topics or fields of study.
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
# EXAMPLES + TEXT2CYPHER NODE
# ------------------------------------------------------------------------------------

examples = [
    {
        "question": "How many TFG projects are there in the Engineering faculty?",
        "query": """
            MATCH (t:TFG)
            WHERE toLower(t.faculty) CONTAINS toLower('Engineering')
            RETURN count(t) AS totalTFGs
        """
    },
    {
        "question": "List all TFG projects related to the keyword 'green space'.",
        "query": """
            MATCH (t:TFG)-[:CONTAINS_KEYWORD]->(k:Keyword)
            WHERE toLower(k.keyword) CONTAINS toLower('green space')
            OPTIONAL MATCH (t)<-[:WRITES]-(s:Person:Student)
            OPTIONAL MATCH (i:Person:Investigator)-[:SUPERVISES]->(t)
            RETURN t.title AS title, t.year AS year, t.abstract AS abstract,
                   t.bachelor AS bachelor, t.faculty AS faculty, t.link AS link,
                   collect(DISTINCT s.name) AS students,
                   collect(DISTINCT i.name) AS investigators
        """
    },
    {
        "question": "Which investigator has supervised the most TFG projects?",
        "query": """
            MATCH (i:Person:Investigator)-[:SUPERVISES]->(t:TFG)
            WITH i, count(t) AS tfgCount, collect(t.title) AS projects
            RETURN i.name AS investigator, tfgCount, projects
            ORDER BY tfgCount DESC
            LIMIT 1
        """
    },
    {
        "question": "Find all publications by investigator 'cristina domingo-marimon'.",
        "query": """
            MATCH (i:Person:Investigator)-[:WRITES]->(p:Publication)
            WHERE toLower(i.name) CONTAINS toLower('cristina domingo-marimon')
            OPTIONAL MATCH (a:Person:Investigator)-[:WRITES]->(p)
            RETURN p.title AS title, p.type AS type, p.year AS year, 
                   p.abstract AS abstract, p.link AS link, p.doi AS doi, p.publication AS publication,
                   collect(DISTINCT a.name) AS authors
        """
    },
    {
        "question": "List all students who completed a TFG under investigator 'cristina domingo-marimon'.",
        "query": """
            MATCH (i:Person:Investigator)-[:SUPERVISES]->(t:TFG)<-[:WRITES]-(s:Person:Student)
            WHERE toLower(i.name) CONTAINS toLower('cristina domingo-marimon')
            RETURN s.name AS student, t.title AS title, t.year AS year,
                   t.abstract AS abstract, t.link AS link
        """
    },
    {
        "question": "Which TFG projects were completed in 2023?",
        "query": """
            MATCH (t:TFG)
            WHERE t.year = '2023'
            OPTIONAL MATCH (t)<-[:WRITES]-(s:Person:Student)
            OPTIONAL MATCH (i:Person:Investigator)-[:SUPERVISES]->(t)
            RETURN t.title AS title, t.year AS year, t.abstract AS abstract,
                   t.bachelor AS bachelor, t.faculty AS faculty, t.link AS link,
                   collect(DISTINCT s.name) AS students,
                   collect(DISTINCT i.name) AS investigators
        """
    },
    {
        "question": "Find all TFG projects under the bachelor 'Smart and Sustainable Cities Management'.",
        "query": """
            MATCH (t:TFG)
            WHERE toLower(t.bachelor) CONTAINS toLower('Smart and Sustainable Cities Management')
            OPTIONAL MATCH (t)<-[:WRITES]-(s:Person:Student)
            OPTIONAL MATCH (i:Person:Investigator)-[:SUPERVISES]->(t)
            RETURN t.title AS title, t.year AS year, t.abstract AS abstract, t.link AS link,
                   collect(DISTINCT s.name) AS students,
                   collect(DISTINCT i.name) AS investigators
        """
    },
    {
        "question": "Identify the most common keywords in TFG projects from the Science faculty.",
        "query": """
            MATCH (t:TFG)-[:CONTAINS_KEYWORD]->(k:Keyword)
            WHERE toLower(t.faculty) CONTAINS toLower('Science')
            WITH k, count(*) AS frequency, collect(t.title) AS projects
            RETURN k.keyword AS keyword, frequency, projects
            ORDER BY frequency DESC
        """
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, Neo4jVector, k=5, input_keys=["question"]
)

text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
If named entities do not use spetial characters like à, è, ò, etc. use the regular characters a, e, o, etc when writing the Cypher query.
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input (that needs to be solved): {question}

Past history (that might help to formulate the query): {history}

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
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("messages")[0].content,
            "history": history,
            "fewshot_examples": fewshot_examples,
            "schema": enhanced_graph.schema,
        }
    )
    return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

# ------------------------------------------------------------------------------------
# EXECUTE CYPHER
# ------------------------------------------------------------------------------------

no_results = "I couldn't find any relevant information in the database"

def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """

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

This tool is designed for students seeking academic guidance on TFGs and publications. Overall you are a chatbot agent be kind and helpful.

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
Use the following results retrieved from a database to provide a succinct, definitive answer 
to the user's question. Respond as if you are answering the question directly.

Question: {question}
Results: {results}
Past history: {history}
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
    history = "\n".join([(message.type + ": " + message.content) for message in state.get("messages")[1:]])
    final_answer = generate_final_chain.invoke(
        {
            "question": state.get("messages")[0].content,
            "results": state.get("database_records"),
            "history": history,
        }
    )
    return {"answer": final_answer,
            "steps": ["generate_final_answer"],
            "cypher_statement": state.get("cypher_statement")}

# ------------------------------------------------------------------------------------
# STATEGRAPH
# ------------------------------------------------------------------------------------

def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "academic":
        return "generate_cypher"

langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges("guardrails",guardrails_condition,)
langgraph.add_edge("generate_cypher", "execute_cypher")
langgraph.add_edge("execute_cypher", "generate_final_answer")
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

        # Append the complete assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": complete_response})

if __name__ == "__main__":
    main()
