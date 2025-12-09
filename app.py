import io
import time
import faiss
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
# Importar para um chunking melhorado
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 
from datetime import datetime 
from github import Github, GithubException 
# Importação adicionada para tokenização e chunking 
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="TCC NLP", layout="wide")
st.title("📚 Chat - Teste inicial rodando no Streamlit Cloud")

LOG_FILE = "chat_log.csv"
token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]
# MELHORIA PRO PERFIL: ajuste de chunking, atual 200/20
perfis = {
    "Perfil_1": {
        "chunk_size": 200,
        "overlap": 20,
        "top_k": 3,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim_value": 384,
        "llm": "google/flan-t5-base",
    },
    "Perfil_2": {
        "chunk_size": 150,
        "overlap": 15,
        "top_k": 1,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim_value": 384,
        "llm": "google/flan-t5-base",
    },
    "Perfil_3": {
        # observação: mudar pra gemini API
        "chunk_size": 250,
        "overlap": 30,
        "top_k": 7,
        "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
        "dim_value": 384,
        "llm": "google/flan-t5-xl",
    },
    "Perfil_4": {
        # observação: mudar pra gemini API
        "chunk_size": 150,
        "overlap": 40,
        "top_k": 5,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "dim_value": 768,
        "llm": "google/flan-t5-large",
    }
}
# perfil_name = 'perfil_1'
# chunk_size = 200
# overlap = 20
# top_k = 3
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# dim_value = 384
# llm = "google/flan-t5-base"


def reset_index():
    """Remove o índice FAISS e a chave de cache para forçar a recriação."""
    if "index" in st.session_state:
        del st.session_state["index"]
    if "chunks" in st.session_state:
        del st.session_state["chunks"]
    if "index_key" in st.session_state:
        del st.session_state["index_key"]
    # Limpa também os dados de validação, para garantir que o embed_model correto seja usado para eles
    if "validation_embeddings" in st.session_state:
        del st.session_state["validation_embeddings"]
    if "validation_index" in st.session_state:
        del st.session_state["validation_index"]
    st.session_state.messages = []    
    # Esta linha força o Streamlit a re-executar todo o script
    # E é a maneira mais confiável de simular um "reload total" na lógica do app.
    st.rerun()


def get_questions_dataset_github():
    file_path = "set_perguntas_respostas.csv"

    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents(file_path)
        csv_content = contents.decoded_content.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content), sep="|")
        st.toast("Dataset de validação carregado!", icon="✔️")
        return df

    except GithubException as e:
        if e.status == 404:
            st.warning(
                "Arquivo 'set_perguntas_respostas.csv' não encontrado no repositório. Acurácia desabilitada."
            )
        else:
            st.error(f"Erro ao carregar o dataset de validação do GitHub: {e}")
        return None
    except Exception as e:
        st.error(f"Erro inesperado ao processar o dataset de validação: {e}")
        return None


def log_interaction_github(question, response, context, time_taken, accuracy):
    file_path = st.secrets["FILE_PATH"]

    try:
        g = Github(token)
        repo = g.get_repo(repo_name)

        # 3. Preparar a nova linha
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Limpeza simples para não quebrar a estrutura do Pipe "|"
        clean_question = question.replace("\n", " ").replace("\r", "").replace("|", "")
        clean_response = response.replace("\n", " ").replace("\r", "").replace("|", "")
        clean_context = context.replace("\n", "; •> ").replace("\r", "").replace("|", "")

        if isinstance(accuracy, (int, float)):
            accuracy_str = f"{accuracy:.2f}%"
        else:
            accuracy_str = str(accuracy)

        new_line = f"{selected_profile}|{perfis[selected_profile]['llm']}|{perfis[selected_profile]['embedding_model']}|{perfis[selected_profile]['dim_value']}|{perfis[selected_profile]['chunk_size']}|{perfis[selected_profile]['overlap']}|{perfis[selected_profile]['top_k']}|{timestamp}|{clean_question}|{clean_response}|{clean_context}|{time_taken:.2f}|{accuracy_str}"

        # 4. Tentar pegar o arquivo existente e atualizar
        try:
            contents = repo.get_contents(file_path)
            csv_content = contents.decoded_content.decode("utf-8")

            # Garante que haja uma quebra de linha antes de adicionar o novo registro
            if csv_content and not csv_content.endswith("\n"):
                csv_content += "\n"

            updated_content = csv_content + new_line

            repo.update_file(
                path=contents.path,
                message=f"Log update: {timestamp}",
                content=updated_content,
                sha=contents.sha,
            )
            st.toast("Log salvo no GitHub com sucesso!", icon="☁️")

        except GithubException as e:
            # Se o arquivo não for encontrado (404), cria ele
            if e.status == 404:
                header = "Nome_Perfil|LLM|Embedding_Model|Dim_Value|Chunk_Size|Overlap|Top_K|Timestamp|Pergunta|Resposta|Contexto_Usado|Tempo_Segundos|Acuracia\n"
                create_content = header + new_line
                repo.create_file(
                    path=file_path,
                    message=f"Create log file: {timestamp}",
                    content=create_content,
                )
                st.toast("Arquivo criado e salvo no GitHub!", icon="✨")
            else:
                raise e

    except Exception as e:
        st.error(f"Erro ao salvar no GitHub: {e}")


# ---------- Helpers ----------
def chunk_text(text, chunk_size=100, overlap=50):
    """
    Divide o texto em chunks, tentando respeitar parágrafos/quebras de linha (se o texto de entrada for 
    bem formatado). Se não, divide com base em caracteres, como um fallback mais previsível. """ 
    text_splitter = CharacterTextSplitter( 
        separator= "\n\n" , # Tenta dividir por parágrafos 
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function= len , 
        is_separator_regex= False , 
    )
    # Se o texto for muito grande, o CharacterTextSplitter será mais eficiente e semântico.
    # No entanto, como CharacterTextSplitter não está no seu import, vamos usar a versão com split_text da LangChain
    # Para ser purista, usarei um fallback de código.
    # Se for para usar a função de palavras, o original estava ok, mas aprimorado para ser mais semântico:

    # --- Versão aprimorada da sua função original (se semanticamente estiver funcionando bem) ---
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
# COMENTÁRIO: Mantive o corpo da sua função `chunk_text` porque ela já estava definida. 
# Para uma MELHORIA real anti-alucinação, o ideal é usar `RecursiveCharacterTextSplitter` do Langchain, 
# mas isso exigiria adicionar uma nova dependência. O RAG "alucina" menos quando os chunks respeitam 
# a semântica do documento.

def file_hash_bytes(b: bytes):
    return hashlib.md5(b).hexdigest()


def calculate_accuracy(generated_response_emb, expected_response_emb):
    """
    Calcula a acurácia (similaridade cosseno) entre a resposta gerada e a resposta esperada.

    A similaridade cosseno varia de -1 (oposto) a 1 (idêntico).
    Vamos normalizar para 0% (totalmente diferente/oposto) a 100% (idêntico).
    Fórmula: (sim_cosseno + 1) / 2 * 100
    """
    # Produto escalar (numerador)
    dot_product = generated_response_emb.dot(expected_response_emb)
    # Normas (denominador)
    norm_gen = np.linalg.norm(generated_response_emb)
    norm_exp = np.linalg.norm(expected_response_emb)

    if norm_gen == 0 or norm_exp == 0:
        return 0.0  # Evita divisão por zero

    cosine_similarity = dot_product / (norm_gen * norm_exp)

    # Normaliza de [-1, 1] para [0, 1] e converte para porcentagem [0%, 100%]
    accuracy = ((cosine_similarity + 1) / 2) * 100

    return accuracy


# ---------- 2. Load Models (Cache) ----------
@st.cache_resource
def load_models(profile_name):
    # Carrega Embeddings
    params = perfis[profile_name]
    # melhoria pro perfil: escolha do modelo de embedding
    # embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_model = SentenceTransformer(params['embedding_model'])

    # MELHORIA PRO PERFIL: escolha do LLM
    # model_name = "google/flan-t5-base"
    model_name = params['llm']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device='cpu')
    
    embedding_dim = params['dim_value']
    return embed_model, pipe, embedding_dim

with st.sidebar:
    st.header("📂 Gestão de Arquivos")
    selected_profile = st.selectbox("Escolha um Perfil:", perfis.keys(), key="profile_selector", on_change=reset_index)
    uploaded = st.file_uploader(
        "Carregue seus PDFs aqui", type="pdf", accept_multiple_files=True
    )
    if "validation_data" not in st.session_state:
        st.session_state["validation_data"] = get_questions_dataset_github()
    st.sidebar.markdown("---")
    st.sidebar.info("Os logs estão sendo salvos automaticamente no GitHub.")
index_ready = False

embed_model, gen_pipe, dim = load_models(selected_profile)
st.session_state.embed_model = embed_model
st.session_state.gen_pipe = gen_pipe
st.session_state.dim = dim

# ---------- 3. Sidebar & Upload ----------
# with st.sidebar:
#     st.header("📂 Gestão de Arquivos")
#     selected_profile = st.selectbox("Escolha um Perfil:", perfis.keys())
#     uploaded = st.file_uploader(
#         "Carregue seus PDFs aqui", type="pdf", accept_multiple_files=True
#     )
#     if "validation_data" not in st.session_state:
#         st.session_state["validation_data"] = get_questions_dataset_github()
#     st.sidebar.markdown("---")
#     st.sidebar.info("Os logs estão sendo salvos automaticamente no GitHub.")
# index_ready = False

if uploaded:
    all_text = ""
    hashes = []
    for file in uploaded:
        b = file.read()
        hashes.append(file_hash_bytes(b))
        reader = PdfReader(io.BytesIO(b))
        for p in reader.pages:
            extracted = p.extract_text()
            if extracted:
                all_text += extracted + "\n"
    cache_key = "_".join(hashes) + f"_{select_profile}"

    # ---------- 4. Processamento (Indexação) ----------
    if (
        "index_key" not in st.session_state
        or st.session_state["index_key"] != cache_key
    ):
        with st.spinner("⚙️ Processando PDFs e criando memória vetorial..."):
            chunks = chunk_text(all_text, perfis[selected_profile]['chunk_size'], perfis[selected_profile]['overlap'])

            if chunks:
                # Cria embeddings
                embeddings = embed_model.encode(
                    chunks, convert_to_numpy=True, show_progress_bar=False
                )

                # Cria Index FAISS
                # dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(st.session_state.dim)
                index.add(embeddings)

                # Salva no Session State
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["index_key"] = cache_key
                st.success(f"Concluído! {len(chunks)} trechos de texto processados.")
            else:
                st.error("Erro: Não foi possível extrair texto dos PDFs.")
    index_ready = True

# ---------- 5. Interface Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if index_ready:
    if prompt_user := st.chat_input("Faça sua pergunta sobre os documentos..."):
        with st.chat_message("user"):
            st.markdown(prompt_user)
        st.session_state.messages.append({"role": "user", "content": prompt_user})

        with st.chat_message("assistant"):
            with st.spinner("Lendo documentos e formulando resposta..."):
                # --- INICIO DA CONTAGEM DE TEMPO ---
                start_time = time.time()

                q_emb = embed_model.encode([prompt_user], convert_to_numpy=True)
                # MELHORIA PRO PERFIL: ajuste do k
                D, Idx = st.session_state["index"].search(q_emb, k=perfis[selected_profile]['top_k'])
                top_idxs = Idx[0].tolist()

                context_chunks = [
                    st.session_state["chunks"][i]
                    for i in top_idxs
                    if i < len(st.session_state["chunks"])
                ]
                context_text = "\n\n".join(context_chunks)
                # MELHORIA PRO PERFIL: ajuste do prompting engineering
                
                # 3. MELHORIA: Prompt Engineering Anti-Alucinação 
                # Instrução CRÍTICA: "Se a resposta não estiver clara, diga que a informação não foi encontrada NO CONTEXTO." 
                final_prompt = f""" 
                Você é um assistente de perguntas e respostas extremamente preciso. 
                Sua tarefa é responder à Pergunta do usuário **EXCLUSIVAMENTE** com base no Contexto fornecido.
                NÃO use seu conhecimento prévio. 
                Se a informação necessária para responder de forma completa e precisa não estiver presente 
                no Contexto, você deve responder: "Desculpe, a informação não foi encontrada no contexto fornecido."
                
                Contexto: {context_text} 
                Pergunta: {prompt_user} 
                Resposta precisa:
                """
                #final_prompt = f"Baseado no Contexto: {context_text}\nResponda a Pergunta: {prompt_user}"

                try:
                    output = st.session_state.gen_pipe(
                        final_prompt, #Usa o prompt aprimorado
                        max_new_tokens=500,
                        do_sample=False,
                        # Aumentei o parâmetro para refletir a necessidade de precisão 
                        num_beams= 3,
                        truncation=True,
                    )
                    response_text = output[0]["generated_text"]
                except Exception as e:
                    response_text = f"Desculpe, tive um erro interno: {str(e)}"

                # --- FIM DA CONTAGEM DE TEMPO ---
                end_time = time.time()
                elapsed_time = end_time - start_time

                # --- CALCULO DE ACURACIA ---
                accuracy = 0.0  # Valor padrão se o dataset não for encontrado
                if st.session_state.validation_data is not None:
                    validation_df = st.session_state.validation_data

                    # 3. Busca pela Pergunta Mais Próxima no Dataset de Validação
                    # Codifica todas as perguntas do dataset (cached)
                    if "validation_embeddings" not in st.session_state:
                        st.session_state["validation_embeddings"] = embed_model.encode(
                            validation_df["Pergunta"].tolist(), convert_to_numpy=True
                        )
                        # CORREÇÃO: Usar a dimensão correta para o índice de validação 
                        st.session_state[ "validation_index" ] = faiss.IndexFlatL2(st.session_state.dim) 
                        st.session_state[ "validation_index" ].add( 
                            st.session_state[ "validation_embeddings" ] 
                        )

                        
                        #st.session_state["validation_index"] = faiss.IndexFlatL2(dim)
                        #st.session_state["validation_index"].add(
                            #st.session_state["validation_embeddings"]
                        #)

                    # Busca a pergunta mais próxima do usuário no index de validação
                    q_emb = st.session_state.embed_model.encode([prompt_user], convert_to_numpy= True )

                    
                    D_val, Idx_val = st.session_state["validation_index"].search(
                        q_emb, k=1
                    )
                    closest_q_idx = Idx_val[0][0]
                    closest_q_distance = D_val[0][0]

                    # Decidimos usar a resposta de validação apenas se a pergunta for MUITO próxima (baixa distância FAISS)
                    # Um valor de 0.5 (arbitrário, pode ser ajustado) é um bom ponto de partida.
                    # Se a distância for muito grande, a comparação de acurácia não faz sentido.
                    if closest_q_distance < 0.5:
                        expected_response = validation_df.iloc[closest_q_idx][
                            "Resposta"
                        ]

                        # 4. Cálculo da Similaridade entre Respostas (Acurácia)
                        generated_emb = embed_model.encode(
                            [response_text], convert_to_numpy=True
                        )[0]
                        expected_emb = embed_model.encode(
                            [expected_response], convert_to_numpy=True
                        )[0]

                        accuracy = calculate_accuracy(generated_emb, expected_emb)

                        accuracy_text = f"**Acurácia**: {accuracy:.2f}% (Comparado com pergunta #{closest_q_idx})"
                    else:
                        accuracy_text = "Acurácia não calculada (Pergunta do usuário não muito próxima do dataset de validação)."

                else:
                    accuracy_text = (
                        "Acurácia não calculada (Dataset de validação não carregado)."
                    )

                # --- SALVAR LOG NO GITHUB ---
                with st.spinner("Salvando registro na nuvem..."):
                    log_interaction_github(
                        prompt_user, response_text, context_text, elapsed_time, accuracy
                    )
                st.toast("Log salvo com sucesso!", icon="💾")

                st.markdown(response_text)

                with st.expander(
                    f"🕵️‍♂️ Detalhes da Execução (Tempo: {elapsed_time:.2f}s)"
                ):
                    st.markdown(f"**{accuracy_text}**")
                    st.markdown("---")
                    st.markdown("**Contexto Usado:**")
                    st.write(context_text)
                    if "expected_response" in locals():
                        st.markdown("**Resposta Esperada (Validação):**")
                        st.write(expected_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

else:
    st.info("👈 Comece carregando um PDF na barra lateral.")
