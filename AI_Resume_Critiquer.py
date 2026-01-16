#import all the necessary packages
import streamlit as st 
import PyPDF2
import io
import os 
from openai import OpenAI 
from dotenv import load_dotenv 

# load the environment variable
load_dotenv()

# title of the page
st.set_page_config(page_title="AI Resume Criticar", page_icon="📜", layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT):", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you are targeting (optional)")

analyze = st.button("Analyze the Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("latin-1")

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if len(file_content) > 20000:
            st.warning("Large resume detected. Only the first part will be analyzed.")
            file_content = file_content[:20000]

        if not file_content.strip(): 
            st.error("File does not have any content")
            st.stop()

        Client = OpenAI(api_key=OPENAI_API_KEY)

        # -----------------------------
        # STEP 1: CREATE SUMMARY
        # -----------------------------
        summary_prompt = f"""Summarize the following resume into key sections. 
Focus on:
1. Profile
2. Skills
3. Experience
4. Education

Resume content:
{file_content}

Provide the summary in a clear, structured format.
"""

        summary_response = Client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        resume_summary = summary_response.choices[0].message.content

        # -----------------------------
        # STEP 2: GENERAL RESUME ANALYSIS
        # -----------------------------
        analysis_prompt = f"""Please analyze this resume and provide constructive feedback.
Focus on:
1. Content clarity and impact
2. Skills presentation
3. Experience descriptions

Resume Summary:
{resume_summary}

Provide your analysis in a clear, structured format.
"""

        analysis_response = Client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        # -----------------------------
        # STEP 3: JOB-SPECIFIC IMPROVEMENTS
        # -----------------------------
        improvement_prompt = f"""Based ONLY on the following resume summary, suggest specific improvements tailored for {job_role if job_role else "general job applications"}.

Focus on:
1. Missing skills or keywords
2. How to strengthen experience bullet points
3. ATS optimization
4. Industry-specific alignment

Resume Summary:
{resume_summary}

Provide clear, actionable recommendations.
"""

        improvement_response = Client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert career coach and ATS optimization specialist."},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.markdown("### 📄 Resume Summary")
        st.markdown(resume_summary)

        st.markdown("### 🔍 General Resume Analysis")
        st.markdown(analysis_response.choices[0].message.content)

        st.markdown("### 🎯 Job-Specific Improvements")
        st.markdown(improvement_response.choices[0].message.content)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
