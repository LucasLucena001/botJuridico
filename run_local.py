#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Jurídico - Execução Local Otimizada
Projeto Final - Pós-Graduação iCEV - NLP

Script principal para execução local com configurações otimizadas.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Configurações para melhor performance
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

def check_requirements():
    """Verifica dependências essenciais."""
    required = ['torch', 'transformers', 'sentence_transformers', 'faiss', 'gradio', 'sklearn']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'faiss':
                import faiss
            elif pkg == 'sklearn':
                import sklearn
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Instale: pip install {' '.join(missing)}")
        return False
    return True

def setup_gpu():
    """Configura GPU se disponível."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {device} ({memory:.1f}GB)")
            torch.cuda.empty_cache()
            return 'cuda'
        else:
            print("💻 Usando CPU")
            return 'cpu'
    except:
        print("💻 PyTorch não encontrado, usando CPU")
        return 'cpu'

class LocalChatbot:
    """Chatbot otimizado para execução local."""
    
    def __init__(self, data_dir="dados"):
        print("🚀 Inicializando Chatbot Local...")
        
        self.device = setup_gpu()
        self.data_dir = Path(data_dir)
        self.documents = {}
        self.chunks = []
        self.embeddings_model = None
        self.vectors = None
        self.index = None
        
        self._load_system()
    
    def _load_system(self):
        """Carrega sistema completo."""
        try:
            # 1. Carregar documentos
            self._load_documents()
            
            # 2. Processar chunks
            self._create_chunks()
            
            # 3. Carregar modelo de embeddings
            self._load_embeddings_model()
            
            # 4. Gerar embeddings
            self._generate_embeddings()
            
            # 5. Criar índice de busca
            self._create_search_index()
            
            print("✅ Sistema carregado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
            raise
    
    def _load_documents(self):
        """Carrega documentos jurídicos."""
        print("📚 Carregando documentos...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Pasta '{self.data_dir}' não encontrada!")
        
        txt_files = list(self.data_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError("Nenhum arquivo .txt encontrado!")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content) > 100:
                    self.documents[txt_file.stem] = self._clean_text(content)
                    print(f"✅ {txt_file.stem}")
                    
            except Exception as e:
                print(f"⚠️ Erro em {txt_file.name}: {e}")
        
        print(f"📚 {len(self.documents)} documentos carregados")
    
    def _clean_text(self, text):
        """Limpeza básica de texto."""
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        return '\n'.join(lines)
    
    def _create_chunks(self):
        """Cria chunks dos documentos."""
        print("🔪 Criando chunks...")
        
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len
            )
            
            for doc_name, content in self.documents.items():
                chunks = splitter.split_text(content)
                
                for i, chunk_text in enumerate(chunks):
                    self.chunks.append({
                        'text': chunk_text,
                        'doc_name': doc_name,
                        'chunk_id': i
                    })
            
            print(f"🧩 {len(self.chunks)} chunks criados")
            
        except ImportError:
            # Fallback simples se langchain não estiver disponível
            print("⚠️ LangChain não disponível, usando chunking simples")
            
            for doc_name, content in self.documents.items():
                words = content.split()
                chunk_size = 200
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    self.chunks.append({
                        'text': chunk_text,
                        'doc_name': doc_name,
                        'chunk_id': i // chunk_size
                    })
            
            print(f"🧩 {len(self.chunks)} chunks criados (método simples)")
    
    def _load_embeddings_model(self):
        """Carrega modelo de embeddings."""
        print("🤖 Carregando modelo de embeddings...")
        
        from sentence_transformers import SentenceTransformer
        
        # Modelo leve e eficiente
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        self.embeddings_model = SentenceTransformer(
            model_name,
            device=self.device
        )
        
        print("✅ Modelo carregado")
    
    def _generate_embeddings(self):
        """Gera embeddings dos chunks."""
        print("🧠 Gerando embeddings...")
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Batch size baseado no dispositivo
        batch_size = 32 if self.device == 'cuda' else 16
        
        self.vectors = self.embeddings_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Embeddings gerados: {self.vectors.shape}")
    
    def _create_search_index(self):
        """Cria índice de busca FAISS."""
        print("📊 Criando índice de busca...")
        
        try:
            import faiss
            import numpy as np
            
            # Garantir tipo float32
            if self.vectors.dtype != np.float32:
                self.vectors = self.vectors.astype(np.float32)
            
            # Criar índice
            dimension = self.vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.index.add(self.vectors)
            
            print(f"✅ Índice criado: {len(self.chunks)} vetores")
            
        except ImportError:
            print("⚠️ FAISS não disponível, usando busca linear")
            self.index = None
    
    def search(self, query, k=5):
        """Busca documentos similares."""
        # Gerar embedding da query
        query_vector = self.embeddings_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if self.index is not None:
            # Busca com FAISS
            import faiss
            import numpy as np
            
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)
            
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    results.append(chunk)
            
            return results
        
        else:
            # Busca linear
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarities[idx])
                results.append(chunk)
            
            return results
    
    def classify_query(self, query):
        """Classifica tipo de query."""
        query_lower = query.lower()
        
        # Keywords jurídicas
        legal_keywords = [
            'artigo', 'lei', 'código', 'direito', 'constituição',
            'clt', 'cdc', 'prazo', 'crime', 'pena', 'processo',
            'habeas', 'mandado', 'ação', 'recurso'
        ]
        
        # Keywords conversacionais
        chat_keywords = [
            'olá', 'oi', 'bom dia', 'boa tarde', 'boa noite',
            'obrigado', 'obrigada', 'tchau', 'até logo',
            'como você', 'quem é', 'funciona'
        ]
        
        if any(kw in query_lower for kw in legal_keywords):
            return 'juridica'
        elif any(kw in query_lower for kw in chat_keywords):
            return 'conversa'
        else:
            return 'fora_escopo'
    
    def generate_answer(self, query, results):
        """Gera resposta baseada nos resultados."""
        if not results:
            return "Não encontrei informações relevantes na base de conhecimento."
        
        # Pegar melhor resultado
        best_result = results[0]
        doc_name = best_result['doc_name'].replace('_', ' ')
        text = best_result['text']
        score = best_result['similarity_score']
        
        # Resposta baseada em template
        answer = f"Com base na legislação brasileira:\n\n"
        answer += f"**Fonte:** {doc_name}\n"
        answer += f"**Relevância:** {score:.3f}\n\n"
        
        # Limitar tamanho do texto
        if len(text) > 500:
            text = text[:500] + "..."
        
        answer += text
        
        if len(results) > 1:
            answer += f"\n\n*Encontrei também informações em {len(results)-1} outro(s) documento(s).*"
        
        answer += "\n\n⚖️ **Aviso:** Esta resposta é informativa. Para casos específicos, consulte um advogado."
        
        return answer
    
    def query(self, question):
        """Processa uma consulta completa."""
        start_time = time.time()
        
        # Classificar query
        classification = self.classify_query(question)
        
        if classification == 'juridica':
            # Buscar contexto
            results = self.search(question, k=3)
            answer = self.generate_answer(question, results)
            sources = [r['doc_name'] for r in results]
            
        elif classification == 'conversa':
            # Resposta conversacional
            if any(word in question.lower() for word in ['olá', 'oi', 'bom dia']):
                answer = "Olá! Sou seu assistente jurídico. Como posso ajudar com questões legais?"
            elif any(word in question.lower() for word in ['obrigado', 'valeu']):
                answer = "De nada! Estou aqui para ajudar com dúvidas jurídicas."
            elif any(word in question.lower() for word in ['como funciona', 'quem é']):
                answer = "Sou um chatbot especializado em legislação brasileira. Posso consultar leis, códigos e normas para responder suas dúvidas."
            else:
                answer = "Olá! Como posso ajudar com questões jurídicas hoje?"
            
            results = []
            sources = []
            
        else:  # fora_escopo
            answer = "Desculpe, minha especialidade é legislação brasileira. Posso ajudar com consultas sobre leis, códigos e normas jurídicas."
            results = []
            sources = []
        
        processing_time = time.time() - start_time
        
        return {
            'question': question,
            'answer': answer,
            'classification': classification,
            'sources': sources,
            'processing_time': processing_time,
            'results_count': len(results) if results else 0
        }

def create_gradio_interface(chatbot):
    """Cria interface Gradio."""
    try:
        import gradio as gr
        
        def chat_function(message, history):
            if not message.strip():
                return history, ""
            
            # Processar consulta
            response = chatbot.query(message)
            
            # Formatar resposta
            answer = response['answer']
            footer = f"\n\n---\n📊 Tempo: {response['processing_time']:.2f}s"
            footer += f" | 🏷️ Tipo: {response['classification']}"
            
            if response['sources']:
                footer += f" | 📚 Fontes: {response['results_count']}"
            
            full_response = answer + footer
            
            # Atualizar histórico
            history.append([message, full_response])
            
            return history, ""
        
        # Interface
        with gr.Blocks(
            title="Chatbot Jurídico Local",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
            # ⚖️ Chatbot Jurídico Inteligente
            **Sistema RAG para Consultas Jurídicas Brasileiras**
            """)
            
            chatbot_ui = gr.Chatbot(
                height=500,
                show_label=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Digite sua pergunta jurídica...",
                    label="Mensagem",
                    scale=4
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
            
            # Exemplos
            gr.Examples(
                examples=[
                    "O que diz o artigo 5º da Constituição Federal?",
                    "Qual o prazo para reclamar de vícios em produtos duráveis?",
                    "Como funciona a jornada de trabalho na CLT?",
                    "O que é considerado união estável?",
                    "Olá, como você funciona?"
                ],
                inputs=msg,
                label="💡 Exemplos de perguntas:"
            )
            
            # Eventos
            msg.submit(chat_function, [msg, chatbot_ui], [chatbot_ui, msg])
            send_btn.click(chat_function, [msg, chatbot_ui], [chatbot_ui, msg])
        
        return interface
        
    except ImportError:
        print("❌ Gradio não disponível")
        return None

def run_cli_interface(chatbot):
    """Interface de linha de comando."""
    print("\n" + "="*60)
    print("💬 INTERFACE DE LINHA DE COMANDO")
    print("="*60)
    print("Digite 'sair' para encerrar")
    print("="*60)
    
    while True:
        try:
            question = input("\n👤 Você: ").strip()
            
            if question.lower() in ['sair', 'quit', 'exit']:
                print("👋 Até logo!")
                break
            
            if not question:
                continue
            
            print("🤔 Processando...")
            response = chatbot.query(question)
            
            print(f"\n🤖 Assistente:")
            print(response['answer'])
            
            print(f"\n📊 Info: {response['classification']} | {response['processing_time']:.2f}s")
            if response['sources']:
                print(f"📚 Fontes: {', '.join(response['sources'][:3])}")
            
        except KeyboardInterrupt:
            print("\n👋 Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")

def main():
    """Função principal."""
    print("🚀 CHATBOT JURÍDICO - EXECUÇÃO LOCAL")
    print("="*50)
    
    # Verificar dependências
    if not check_requirements():
        return
    
    # Verificar dados
    if not Path("dados").exists():
        print("❌ Pasta 'dados' não encontrada!")
        print("💡 Crie a pasta 'dados' e adicione os arquivos .txt da legislação")
        return
    
    try:
        # Inicializar chatbot
        chatbot = LocalChatbot()
        
        # Escolher interface
        print("\n🎯 Escolha a interface:")
        print("1. Interface Web (Gradio) - Recomendado")
        print("2. Linha de Comando (CLI)")
        
        choice = input("\nEscolha (1 ou 2): ").strip()
        
        if choice == "1":
            print("\n🌐 Iniciando interface web...")
            interface = create_gradio_interface(chatbot)
            
            if interface:
                interface.launch(
                    server_name="0.0.0.0",
                    server_port=7860,
                    share=False,
                    show_error=True
                )
            else:
                print("❌ Gradio não disponível, usando CLI...")
                run_cli_interface(chatbot)
        
        else:
            run_cli_interface(chatbot)
    
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()