import streamlit as st
import requests
import os
import openai
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv  # Load Ragie API key from .env

# Load environment variables (Ragie API Key)
load_dotenv()
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")

class RAGPipeline:
    def __init__(self, sambanova_api_key: str):
        """
        Initialize the RAG pipeline with Ragie API key from .env and user-provided SambaNova API key.
        """
        self.ragie_api_key = RAGIE_API_KEY
        self.sambanova_api_key = sambanova_api_key

        if not self.ragie_api_key:
            raise Exception("Missing Ragie API key! Please check your .env file.")

        # Initialize SambaNova API client with user-provided key
        self.sambanova_client = openai.OpenAI(
            api_key=self.sambanova_api_key,
            base_url="https://api.sambanova.ai/v1",
        )

        # API endpoints
        self.RAGIE_UPLOAD_URL = "https://api.ragie.ai/documents/url"
        self.RAGIE_RETRIEVAL_URL = "https://api.ragie.ai/retrievals"

    def upload_document(self, url: str, name: Optional[str] = None, mode: str = "fast") -> Dict:
        """
        Upload a document to Ragie from a URL.
        """
        if not name:
            name = urlparse(url).path.split('/')[-1] or "document"

        payload = {
            "mode": mode,
            "name": name,
            "url": url
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }

        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)

        if not response.ok:
            raise Exception(f"Document upload failed: {response.status_code} {response.reason}")

        return response.json()

    def retrieve_chunks(self, query: str, scope: str = "tutorial") -> List[str]:
        """
        Retrieve relevant chunks from Ragie for a given query and measure time taken.
        """
        start_time = time.time()  # Start time measurement
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ragie_api_key}"
        }

        payload = {
            "query": query,
            "filters": {
                "scope": scope
            }
        }

        response = requests.post(
            self.RAGIE_RETRIEVAL_URL,
            headers=headers,
            json=payload
        )

        if not response.ok:
            raise Exception(f"Retrieval failed: {response.status_code} {response.reason}")

        data = response.json()
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        st.info(f"Time taken to retrieve document information: {elapsed_time:.2f} seconds")
        print(f"Time taken to retrieve document information: {elapsed_time:.2f} seconds")  # Print in console

        return [chunk["text"] for chunk in data["scored_chunks"]]

    def create_system_prompt(self, chunk_texts: List[str]) -> str:
        """
        Create the system prompt with the retrieved chunks.
        """
        return f"""These are very important to follow: You are "Ragie AI", a professional but friendly AI chatbot working as an assistant to the user. Your current task is to help the user based on all of the information available to you shown below. Answer informally, directly, and concisely without a heading or greeting but include everything relevant. Use richtext Markdown when appropriate including bold, italic, paragraphs, and lists when helpful. If using LaTeX, use double $$ as delimiter instead of single $. Use $$...$$ instead of parentheses. Organize information into multiple sections or points when appropriate. Don't include raw item IDs or other raw fields from the source. Don't use XML or other markup unless requested by the user. Here is all of the information available to answer the user: === {chunk_texts} === If the user asked for a search and there are no results, make sure to let the user know that you couldn't find anything, and what they might be able to do to find the information they need. END SYSTEM INSTRUCTIONS"""

    def generate_response(self, system_prompt: str, query: str) -> str:
        """
        Generate response using SambaNova AI (Llama-3.2-11B-Vision-Instruct).
        """
        response = self.sambanova_client.chat.completions.create(
            model="Llama-3.2-11B-Vision-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            top_p=0.1
        )

        return response.choices[0].message.content

    def process_query(self, query: str, scope: str = "tutorial") -> str:
        """
        Process a query through the complete RAG pipeline.
        """
        chunks = self.retrieve_chunks(query, scope)

        if not chunks:
            return "No relevant information found for your query."

        system_prompt = self.create_system_prompt(chunks)
        return self.generate_response(system_prompt, query)

def main():
    st.set_page_config(page_title="RAG Over Online Documents with SambaNova", layout="wide")
    
    # Title with SambaNova in Orange
    st.markdown("""
        <h1 style="text-align: center;">
            <span style="color: orange;">SambaNova</span> RAG Over Online Documents 📚
        </h1>
    """, unsafe_allow_html=True)

    # API Key Input for SambaNova
    sambanova_key = st.text_input("Enter SambaNova API Key", type="password")

    if st.button("Submit API Key"):
        if sambanova_key:
            st.session_state["sambanova_key"] = sambanova_key
            st.success("SambaNova API key submitted successfully!")
        else:
            st.error("Please enter your SambaNova API key.")

    if "sambanova_key" in st.session_state:
        try:
            pipeline = RAGPipeline(sambanova_api_key=st.session_state["sambanova_key"])
            st.success("API keys loaded successfully!")
        except Exception as e:
            st.error(str(e))
            return

        # Document Upload Section
        st.markdown("### 📄 Document Upload")
        doc_url = st.text_input("Enter document URL")
        doc_name = st.text_input("Document name (optional)")

        col1, col2 = st.columns([1, 3])
        with col1:
            upload_mode = st.selectbox("Upload mode", ["fast", "accurate"])

        if st.button("Upload Document"):
            if doc_url:
                try:
                    with st.spinner("Uploading document..."):
                        pipeline.upload_document(
                            url=doc_url,
                            name=doc_name if doc_name else None,
                            mode=upload_mode
                        )
                        time.sleep(5)  # Wait for indexing
                        st.success("Document uploaded and indexed successfully!")
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")
            else:
                st.error("Please provide a document URL.")

        # Query Section
        st.markdown("### 🔍 Query Document")
        query = st.text_input("Enter your query")

        if st.button("Generate Response"):
            if query:
                try:
                    with st.spinner("Generating response..."):
                        response = pipeline.process_query(query)
                        st.markdown("### Response:")
                        st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.error("Please enter a query.")

if __name__ == "__main__":
    main()
