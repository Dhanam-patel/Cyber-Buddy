from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import time
import streamlit as st

st.title("Cyber Buddy")
# user_model = st.selectbox("Select Model", ["HuggingFace", "Google Generative AI"])
# if user_model == "HuggingFace":
#     hf_api_key = st.text_input("Please enter your HuggingFace API key", type="password")
#     llm = HuggingFaceEndpoint(
#         repo_id="baidu/ERNIE-4.5-300B-A47B-Base-PT",
#         task="text-generation",
#         huggingfacehub_api_token=hf_api_key
#     )
#     model = ChatHuggingFace(llm=llm, temperature=2)
#     model_name = "baidu/ERNIE-4.5-300B-A47B-Base-PT"
# else:
# gem_api_key = st.text_input("Please enter your Google API key", type="password")
model = ChatGoogleGenerativeAI(
model="gemini-2.5-flash-lite",
temperature=1,
google_api_key = 'AIzaSyCIxXS7whyWcf8VWOgQF59xqJBUs9KpZy0'
)
model_name = "gemini-2.0-flash-lite"

# User_gender = st.selectbox('Gender', ['Male', 'Female'])
# User_age = st.text_input('Age: ')
# System_Message = str(load_pro/mpt("Template/System_Message.json").invoke({}))

if 'history' not in st.session_state:
    st.session_state.history = [
        SystemMessage("""
You are a personalized cybersecurity advisor AI, specialized in offering tailored, practical guidance for cybersecurity best 
practices, threat analysis, vulnerability assessment, incident response planning, and secure development. You incorporate the latest cybersecurity trends and standards to help users protect their digital environments.””
         """)

    ,
    ]


user = st.chat_input("You: ")

# Prompt_Template = load_prompt("Template/Prompt.json")
# Prompt = Prompt_Template.invoke({
#     'User_gender': User_gender,
#     'User_age': User_age,
#     'chat_history_text': chat_history_text,
# })
if user:
    st.session_state.history.append(HumanMessage(content=user))
    start = time.time()
    chat_history_text = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in st.session_state.history])
    response = model.invoke(f"""
    I want you to act as my personal cybersecurity advisor. I will share details about my systems, applications, or security concerns, and you will provide customized advice, including risk assessments, mitigation strategies, real-world attack examples, and compliance recommendations. Your suggestions should consider the latest cybersecurity threats, tools, and frameworks relevant to my context. Help me build strong defenses, detect vulnerabilities, respond to incidents, and improve security posture effectively.

Chat History
{chat_history_text}
""")

    
    end = time.time()
    st.session_state.history.append(AIMessage(content=response.content))
    st.write(f"Time taken: {end - start} seconds")
    st.write(f"Model name: {model_name}") 


for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        st.markdown(
            f"""
            <div style='text-align: right; padding: 5px; margin: 5px; border-radius: 10px; width: fit-content; max-width: 70%; float: right; clear: both;'>
                <b>You:</b> {message.content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif isinstance(message, AIMessage):
        st.markdown(
            f"""
            <div style='text-align: left; padding: 5px; margin: 5px; border-radius: 10px; width: fit-content; max-width: 70%; float: left; clear: both;'>
                <b>AI:</b>{message.content}
            </div>
            """,
            unsafe_allow_html=True,
        )




