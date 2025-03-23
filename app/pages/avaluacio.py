import streamlit as st
import os
import json
import random
from datetime import datetime

def load_data():
    # Obté la ruta absoluta del directori actual i construeix la ruta cap a "avaluacio.json"
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "..", "avaluacio.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_submission(evaluations):
    output_dir = "user_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "user_answers.json")

    # Carrega les submissions anteriors si existeixen
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            submissions = json.load(f)
    else:
        submissions = []

    # Crea una nova entrada amb un timestamp i la informació de les avaluacions
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": evaluations
    }
    submissions.append(new_entry)

    # Desa la llista actualitzada
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=4, ensure_ascii=False)

def main():
    st.set_page_config(
        page_title="Avaluació de Respostes",
        page_icon=":desktop_computer:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    st.title("Avaluació de Respostes")
    data = load_data()
    
    # Inicialitza l'ordre aleatori de les respostes per a cada pregunta, només la primera vegada
    if "answers_order" not in st.session_state:
        st.session_state["answers_order"] = {}
    
    evaluations = {}
    
    for question in data["questions"]:
        st.header(question["text"])
        evaluations[question["id"]] = {}
        
        # Mantenim l'ordre aleatori per cada pregunta a la sessió
        if question["id"] not in st.session_state["answers_order"]:
            answers = question["answers"].copy()
            random.shuffle(answers)
            st.session_state["answers_order"][question["id"]] = answers
        else:
            answers = st.session_state["answers_order"][question["id"]]
        
        # Mostra les respostes en dues columnes
        col1, col2 = st.columns(2)
        columns = [col1, col2]
        
        for idx, col in enumerate(columns):
            if idx < len(answers):
                with col:
                    st.write(answers[idx]["text"])
                    # Utilitza el nom original del model com a clau en el slider
                    rating = st.slider("Avaluació (1-5)", 1, 5, key=f"{question['id']}_{answers[idx]['model']}")
                    evaluations[question["id"]][answers[idx]["model"]] = rating
        
        st.markdown("---")
    
    if st.button("Enviar Avaluacions"):
        save_submission(evaluations)
        st.success("Les avaluacions s'han guardat correctament!")

if __name__ == "__main__":
    main()
