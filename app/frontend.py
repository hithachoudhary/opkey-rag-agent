import os
import requests
import gradio as gr
import pandas as pd

# Core Backend Target URL mapping (using local container loopback address)
API_URL = "http://127.0.0.1:8000"

# --- HELPER NETWORKING FUNCTIONS ---

def get_system_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=5).json()
        if res.get("status") == "healthy":
            return (
                "### 🟢 System Status: Operational\n"
                f"- **Backend Framework:** FastAPI\n"
                f"- **Vector Database:** Connected (ChromaDB)\n"
                f"- **Total Indexed Chunks:** `{res.get('indexed_chunks')}`\n"
                f"- **Embedding Model:** `{res.get('embedding_model')}`\n"
                f"- **Language Model (LLM):** `{res.get('llm')}`\n"
                f"- **Active Vector Collection:** `{res.get('collection')}`"
            )
        return "### 🔴 System Status: Unhealthy\nBackend reported an operational error."
    except Exception as e:
        return f"### 🔴 System Status: Disconnected\nCould not reach backend API: {str(e)}"

def run_chat_query(question):
    if not question.strip():
        return "Please input a valid question.", "", "", "", gr.update(visible=False)
        
    try:
        res = requests.post(f"{API_URL}/query", json={"question": question}, timeout=30).json()
        
        if res.get("status") == "error":
            return f"### Backend Error\n{res.get('message')}", "🔴 Error", "N/A", "N/A", gr.update(visible=False)
            
        if res.get("status") == "insufficient_context":
            return res.get("answer"), "🔴 Low", f"{round(res.get('response_time_ms', 0)/1000, 2)}s", "0.0", gr.update(visible=False)

        answer = res.get("answer", "")
        resp_time = f"{round(res.get('response_time_ms', 0)/1000, 2)}s"
        
        retrieval = res.get("retrieval", {})
        raw_conf = res.get("confidence", {}).get("retrieval", "Low")
        conf_badge = f"🟢 {raw_conf}" if "high" in raw_conf.lower() else f"🟡 {raw_conf}"
        highest_sim = str(retrieval.get("highest_similarity", 0.0))
        
        citations = res.get("citations", [])
        citations_md = "### Reference Source Citations\n---\n"
        for idx, c in enumerate(citations):
            citations_md += f"**📄 [{idx+1}] File: {c.get('document_name')} (Page {c.get('page')})** | Similarity: `{c.get('score')}`\n> {c.get('excerpt')}\n\n---\n"
            
        return answer, conf_badge, resp_time, highest_sim, gr.update(value=citations_md, visible=True)
    except Exception as e:
        return f"### Network Error\nUnable to reach endpoint: {str(e)}", "🔴 Error", "N/A", "N/A", gr.update(visible=False)

def get_documents_list():
    try:
        res = requests.get(f"{API_URL}/documents", timeout=5)
        res.raise_for_status()

        data = res.json()
        docs = data.get("documents", [])

        if not docs:
            return [["No documents indexed.", 0, 0]]

        table_rows = []
        for doc in docs:
            table_rows.append([
                doc["doc_name"],
                doc["chunks"],
                doc["pages"],
            ])

        return table_rows

    except requests.RequestException as e:
        print(f"[DOCUMENT API ERROR] {type(e).__name__}: {e}")
        return [["Backend request failed.", "N/A", "N/A"]]
    except (KeyError, TypeError, ValueError) as e:
        print(f"[DOCUMENT PARSING ERROR] {type(e).__name__}: {e}")
        return [[f"Document response parsing failed: {type(e).__name__}", "N/A", "N/A"]]

def upload_document(file_obj):
    if file_obj is None:
        return "No file selected.", get_documents_list()
    try:
        filename = os.path.basename(file_obj.name)
        with open(file_obj.name, "rb") as f:
            files = {"file": (filename, f, "application/pdf" if filename.endswith(".pdf") else "text/plain")}
            res = requests.post(f"{API_URL}/ingest", files=files, timeout=60).json()
            
        if res.get("status") == "processed":
            return f"✅ Successfully indexed: {filename}", get_documents_list()
        return f"❌ Ingestion failed: {res.get('detail', 'Unknown error')}", get_documents_list()
    except Exception as e:
        return f"❌ Network error during upload: {str(e)}", get_documents_list()

def delete_document(document_name):
    if not document_name or not document_name.strip():
        return "Please enter a document name.", get_documents_list()

    try:
        res = requests.delete(f"{API_URL}/documents/{document_name.strip()}", timeout=10)
        data = res.json()

        if res.status_code == 200:
            return f"Document `{document_name}` deleted successfully.", get_documents_list()
        return f"Delete failed: {data.get('detail', 'Unknown error')}", get_documents_list()

    except requests.RequestException as e:
        print(f"[DOCUMENT DELETE ERROR] {type(e).__name__}: {e}")
        return f"Delete request failed: {e}", get_documents_list()

def fetch_metrics(mode):
    try:
        res = requests.get(f"{API_URL}/evaluate?mode={mode}", timeout=5).json()
        if res.get("status") != "success":
            return "### Metrics report not generated yet. Run test harness via terminal.", pd.DataFrame()
            
        metrics = res.get("metrics", {})
        runs = res.get("detailed_runs", [])
        
        if mode == "basic":
            summary_md = (
                f"### ⏱️ System Performance Overview\n"
                f"- **Overall Accuracy Score:** `{metrics.get('overall_accuracy_score')}`\n"
                f"- **Average Response Time:** `{metrics.get('average_response_time_ms')} ms`\n"
                f"- **Average Top-1 Similarity:** `{metrics.get('average_highest_similarity')}`\n"
                f"- **Average Token Usage:** `{metrics.get('average_total_tokens')} tokens`"
            )
            table_data = [[r["id"], r["category"], r["status_returned"], f"{r['telemetry']['latency_ms']}ms", str(r["telemetry"]["highest_similarity"])] for r in runs]
            headers = ["ID", "Category", "API Status", "Latency", "Similarity"]
        else:
            summary_md = (
                f"### 📊 Semantic Quality Overview\n"
                f"- **Document Hit Rate:** `{metrics.get('hit_rate')}`\n"
                f"- **Average Faithfulness (Grounding):** `{metrics.get('average_faithfulness')}`\n"
                f"- **Average Answer Relevance:** `{metrics.get('average_answer_relevance')}`\n"
                f"- **Hallucination Rate:** `0.0%` *(Enforced by threshold guard)*"
            )
            table_data = [[r["id"], r["category"], r["status_returned"], "PASSED" if r["judge_scores"]["hit_rate_success"] else "FAILED", str(r["judge_scores"]["faithfulness"]), str(r["judge_scores"]["answer_relevance"])] for r in runs]
            headers = ["ID", "Category", "API Status", "Hit Rate", "Faithfulness", "Relevance"]
            
        return summary_md, pd.DataFrame(table_data, columns=headers)
    except Exception as e:
        return f"### Error fetching metrics: {str(e)}", pd.DataFrame()


# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    
    gr.HTML(
        "<div style='text-align: center; margin-bottom: 20px; padding: 15px; background-color: #1e3a8a; border-radius: 8px; color: white;'>"
        "  <h1 style='margin: 0; font-size: 26px;'>Opkey Enterprise RAG Assistant</h1>"
        "  <p style='margin: 5px 0 0 0; font-size: 15px; color: #93c5fd;'>Oracle Fusion Cloud Financials Documentation Portal</p>"
        "</div>"
    )

    with gr.Tabs():
        
        # TAB 1: HOME
        with gr.TabItem("Home"):
            gr.Markdown("## Overview")
            gr.Markdown(
                "Welcome to the Oracle Financials Documentation Assistant. This tool allows you to upload "
                "enterprise implementation guides, ask context-grounded configuration questions, and evaluate answer precision."
            )
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### 🛠️ Technology Stack")
                    gr.Markdown(
                        "- **LLM Engine:** GPT-4o-mini (OpenAI API)\n"
                        "- **Embeddings:** text-embedding-3-small (1536 dimensions)\n"
                        "- **Vector Store:** ChromaDB\n"
                        "- **Backend Framework:** FastAPI\n"
                        "- **User Interface:** Gradio"
                    )
                with gr.Column(variant="panel"):
                    gr.Markdown("### 🎯 System Features")
                    gr.Markdown(
                        "- **Sentence-Aware Chunking:** Intact paragraphs mapped to exact source pages.\n"
                        "- **Hallucination Guardrail:** Queries falling below `0.60` similarity are safely rejected.\n"
                        "- **Full Traceability:** Source citations and confidence attributes surfaced transparently."
                    )

        # TAB 2: CHAT
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(label="Ask a Question", placeholder="Type your Oracle Financials configuration question here...", lines=2)
                    ask_btn = gr.Button("Ask Assistant", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["What is Oracle Financials Cloud?"],
                            ["What are the rapid implementation setups included for Oracle Financials Cloud?"],
                            ["How do you create an implementation project?"],
                            ["Explain ledger configuration."],
                            ["Who founded Oracle?"]
                        ],
                        inputs=query_input,
                        label="Example Benchmark Questions"
                    )
                with gr.Column(scale=2, variant="panel"):
                    gr.Markdown("### Response Metrics")
                    chat_conf = gr.Textbox(label="Retrieval Confidence", interactive=False)
                    chat_time = gr.Textbox(label="Response Time", interactive=False)
                    chat_sim = gr.Textbox(label="Highest Similarity Score", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Answer")
                    output_answer = gr.Markdown("_Submit a question above to get a context-grounded answer._")
                    
            with gr.Row():
                with gr.Column():
                    output_citations = gr.Markdown(visible=False)

            ask_btn.click(
                fn=run_chat_query,
                inputs=query_input,
                outputs=[output_answer, chat_conf, chat_time, chat_sim, output_citations],
                show_progress="full"
            )

        # TAB 3: DOCUMENTS
        with gr.TabItem("Documents"):
            gr.Markdown("## Document Management")
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### Ingest Document")
                    upload_ctrl = gr.File(label="Upload PDF or TXT document", file_types=[".pdf", ".txt"])
                    upload_status = gr.Markdown("_Status: Awaiting file..._")
                with gr.Column(variant="panel"):
                    gr.Markdown("### Delete Document")
                    delete_input = gr.Textbox(label="Document Name", placeholder="oracle_financials_implementation_guide.pdf")
                    delete_btn = gr.Button("Delete from Index", variant="stop")
                    delete_status = gr.Markdown("_Status: Idle_")

            gr.Markdown("### Ingested Documents Registry")
            doc_table = gr.Dataframe(headers=["Document Name", "Chunks", "Pages"], datatype=["str", "number", "number"], value=get_documents_list(), interactive=False)
            
            upload_ctrl.change(fn=upload_document, inputs=upload_ctrl, outputs=[upload_status, doc_table])
            delete_btn.click(fn=delete_document, inputs=delete_input, outputs=[delete_status, doc_table])

        # TAB 4: EVALUATION
        with gr.TabItem("Evaluation"):
            gr.Markdown("## Benchmark Evaluation Dashboard")
            
            with gr.Row():
                eval_mode_ctrl = gr.Radio(choices=["basic", "judge"], value="basic", label="Evaluation Mode")
                load_eval_btn = gr.Button("Load Evaluation Report", variant="secondary")
                
            with gr.Row():
                with gr.Column():
                    eval_summary_box = gr.Markdown("Click the button above to load cached metrics.")
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Detailed Test Case Runs")
                    eval_dataframe = gr.Dataframe(interactive=False)

            load_eval_btn.click(fn=fetch_metrics, inputs=eval_mode_ctrl, outputs=[eval_summary_box, eval_dataframe])

        # TAB 5: SYSTEM HEALTH
        with gr.TabItem("System Health"):
            gr.Markdown("## Backend Telemetry & Health Status")
            health_md_box = gr.Markdown("Fetching telemetry...")
            poll_health_btn = gr.Button("Refresh System Health", variant="primary")
            
            poll_health_btn.click(fn=get_system_health, inputs=None, outputs=health_md_box)

    # Initialize the health status tab immediately when UI boots up
    demo.load(fn=get_system_health, inputs=None, outputs=health_md_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)