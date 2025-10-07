# -*- coding: utf-8 -*-
"""
Chatbot Jurídico Inteligente com Classificador RAG
Projeto Final - Pós-Graduação iCEV - NLP

Autor: [Seu Nome]
Data: Outubro 2025
Professor: Dimmy Magalhães

Sistema completo de chatbot jurídico que:
1. Classifica automaticamente o tipo de pergunta
2. Usa RAG para consultas jurídicas específicas
3. Mantém histórico de conversas
4. Oferece feedback e métricas de confiabilidade
"""

import os
import re
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Suprimir warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SISTEMA DE CARREGAMENTO E PROCESSAMENTO DE DOCUMENTOS
# ============================================================================

class DocumentProcessor:
    """Classe responsável por carregar e processar documentos jurídicos."""
    
    def __init__(self, data_dir: str = "dados"):
        self.data_dir = Path(data_dir)
        self.documents = {}
        self.metadata = {}
        
    def load_text_documents(self) -> Dict[str, str]:
        """Carrega todos os documentos .txt da pasta dados."""
        print("📚 Carregando documentos jurídicos...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Diretório {self.data_dir} não encontrado!")
        
        txt_files = list(self.data_dir.glob("*.txt"))
        
        if not txt_files:
            raise FileNotFoundError("Nenhum arquivo .txt encontrado na pasta dados!")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extrair metadados do nome do arquivo
                filename = txt_file.stem
                self.documents[filename] = content
                self.metadata[filename] = self._extract_metadata(filename, content)
                
                print(f"✅ {filename}: {len(content):,} caracteres")
                
            except Exception as e:
                print(f"❌ Erro ao carregar {txt_file.name}: {e}")
        
        print(f"\n📊 Total: {len(self.documents)} documentos carregados")
        print(f"📝 Total de caracteres: {sum(len(doc) for doc in self.documents.values()):,}")
        
        return self.documents
    
    def _extract_metadata(self, filename: str, content: str) -> Dict:
        """Extrai metadados do documento."""
        # Categorizar por tipo de lei
        categories = {
            'Código': ['Código_Civil', 'Código_de_Defesa', 'Código_de_Processo', 'Código_de_Trânsito', 'Código_Eleitoral', 'Código_Tributário'],
            'Consolidação': ['Consolidação_das_Leis'],
            'Constituição': ['Constituição_Federal'],
            'Estatuto': ['Estatuto_da_Cidade', 'Estatuto_da_Criança', 'Estatuto_da_Pessoa'],
            'Lei_Específica': ['Lei_de_', 'Lei_do_', 'Lei_dos_', 'Lei_Geral', 'Lei_Maria', 'Lei_Orgânica', 'Lei_Brasileira'],
            'Marco': ['Marco_Civil'],
            'Novo': ['Novo_Código']
        }
        
        category = 'Outros'
        for cat, keywords in categories.items():
            if any(keyword in filename for keyword in keywords):
                category = cat
                break
        
        # Extrair ano se possível
        year_match = re.search(r'(\d{4})', filename)
        year = int(year_match.group(1)) if year_match else None
        
        return {
            'filename': filename,
            'category': category,
            'year': year,
            'length': len(content),
            'word_count': len(content.split())
        }
    
    def clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto."""
        # Remover quebras de linha excessivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remover espaços excessivos
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remover caracteres especiais mantendo acentos
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\n\r°§]', ' ', text)
        
        # Remover linhas muito curtas (provavelmente ruído)
        lines = text.split('\n')
        clean_lines = [line.strip() for line in lines if len(line.strip()) > 5]
        
        return '\n'.join(clean_lines).strip()

# ============================================================================
# 2. SISTEMA DE CHUNKING INTELIGENTE
# ============================================================================

class IntelligentChunker:
    """Sistema de chunking otimizado para textos jurídicos."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_documents(self, documents: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Divide documentos em chunks com metadados."""
        print("🔪 Aplicando chunking inteligente...")
        
        chunked_docs = {}
        total_chunks = 0
        
        for doc_name, content in documents.items():
            chunks = self._chunk_legal_text(content, doc_name)
            chunked_docs[doc_name] = chunks
            total_chunks += len(chunks)
            
            print(f"📄 {doc_name}: {len(chunks)} chunks")
        
        print(f"✅ Total de chunks: {total_chunks}")
        return chunked_docs
    
    def _chunk_legal_text(self, text: str, doc_name: str) -> List[Dict]:
        """Chunking específico para textos jurídicos."""
        chunks = []
        
        # Tentar dividir por artigos primeiro
        if 'Art.' in text or 'Artigo' in text:
            article_chunks = self._chunk_by_articles(text, doc_name)
            if article_chunks:
                return article_chunks
        
        # Fallback para chunking por tamanho
        return self._chunk_by_size(text, doc_name)
    
    def _chunk_by_articles(self, text: str, doc_name: str) -> List[Dict]:
        """Divide texto por artigos."""
        chunks = []
        
        # Padrões para identificar artigos
        article_patterns = [
            r'Art\.\s*\d+',
            r'Artigo\s*\d+',
            r'Art\s*\d+',
        ]
        
        for pattern in article_patterns:
            articles = re.split(f'({pattern})', text)
            if len(articles) > 3:  # Se encontrou divisões válidas
                current_chunk = ""
                current_article = ""
                
                for i, part in enumerate(articles):
                    if re.match(pattern, part):
                        # Salvar chunk anterior se existir
                        if current_chunk.strip():
                            chunks.append(self._create_chunk(
                                current_chunk.strip(), 
                                doc_name, 
                                len(chunks), 
                                current_article
                            ))
                        
                        current_article = part
                        current_chunk = part
                    else:
                        current_chunk += part
                        
                        # Se chunk ficou muito grande, dividir
                        if len(current_chunk) > self.chunk_size * 1.5:
                            chunks.append(self._create_chunk(
                                current_chunk.strip(), 
                                doc_name, 
                                len(chunks), 
                                current_article
                            ))
                            current_chunk = ""
                
                # Adicionar último chunk
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), 
                        doc_name, 
                        len(chunks), 
                        current_article
                    ))
                
                return chunks
        
        return []
    
    def _chunk_by_size(self, text: str, doc_name: str) -> List[Dict]:
        """Divide texto por tamanho com sobreposição."""
        chunks = []
        words = text.split()
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunks.append(self._create_chunk(
                chunk_text, 
                doc_name, 
                chunk_id, 
                f"Seção {chunk_id + 1}"
            ))
            
            start = end - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def _create_chunk(self, text: str, doc_name: str, chunk_id: int, section: str) -> Dict:
        """Cria um chunk com metadados."""
        return {
            'text': text,
            'doc_name': doc_name,
            'chunk_id': chunk_id,
            'section': section,
            'length': len(text),
            'word_count': len(text.split())
        }

# ============================================================================
# 3. GERENCIAMENTO DE MEMÓRIA GPU
# ============================================================================

class GPUMemoryManager:
    """Gerencia memória GPU para evitar travamentos."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Verifica se GPU está disponível."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_gpu_info(self) -> Dict:
        """Retorna informações da GPU."""
        if not self.gpu_available:
            return {"available": False}
        
        try:
            import torch
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            return {
                "available": True,
                "name": props.name,
                "total_memory": props.total_memory,
                "allocated": torch.cuda.memory_allocated(device),
                "cached": torch.cuda.memory_reserved(device),
                "free": props.total_memory - torch.cuda.memory_reserved(device)
            }
        except Exception as e:
            print(f"⚠️ Erro ao obter info da GPU: {e}")
            return {"available": False}
    
    def clear_cache(self):
        """Limpa cache da GPU."""
        if self.gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
                print("🧹 Cache da GPU limpo")
            except Exception as e:
                print(f"⚠️ Erro ao limpar cache: {e}")
    
    def monitor_memory(self) -> str:
        """Monitora uso de memória GPU."""
        info = self.get_gpu_info()
        if not info["available"]:
            return "GPU não disponível"
        
        total_gb = info["total_memory"] / 1e9
        allocated_gb = info["allocated"] / 1e9
        cached_gb = info["cached"] / 1e9
        free_gb = info["free"] / 1e9
        
        usage_percent = (info["allocated"] / info["total_memory"]) * 100
        
        return f"GPU: {allocated_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%) | Cache: {cached_gb:.1f}GB | Livre: {free_gb:.1f}GB"
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Calcula batch size otimizado baseado na memória disponível."""
        info = self.get_gpu_info()
        if not info["available"]:
            return 16  # Batch size conservador para CPU
        
        free_memory = info["free"]
        
        if free_memory > 6e9:  # > 6GB livre
            return base_batch_size * 2
        elif free_memory > 3e9:  # > 3GB livre
            return base_batch_size
        elif free_memory > 1e9:  # > 1GB livre
            return base_batch_size // 2
        else:  # < 1GB livre
            return base_batch_size // 4

# ============================================================================
# 4. SISTEMA DE EMBEDDINGS E BUSCA VETORIAL
# ============================================================================

class VectorStore:
    """Sistema de armazenamento e busca vetorial otimizado para GPU."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.embeddings_model = None
        self.vectors = None
        self.chunks = []
        self.index = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self):
        """Detecta e configura o dispositivo (GPU/CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
                print(f"💾 Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                device = 'cpu'
                print("💻 Usando CPU (GPU não disponível)")
            return device
        except ImportError:
            print("💻 PyTorch não encontrado, usando CPU")
            return 'cpu'
    
    def _load_model(self):
        """Carrega o modelo de embeddings otimizado para GPU."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🤖 Carregando modelo de embeddings: {self.model_name}")
            print(f"📍 Dispositivo: {self.device}")
            
            # Carregar modelo com configurações otimizadas
            self.embeddings_model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                cache_folder='./cache_models'  # Cache local para evitar redownloads
            )
            
            # Configurar para modo de inferência (economiza memória)
            self.embeddings_model.eval()
            
            if self.device == 'cuda':
                # Limpar cache da GPU
                import torch
                torch.cuda.empty_cache()
                print(f"🔧 Modelo carregado na GPU")
            
            print("✅ Modelo de embeddings carregado!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            print("💡 Tentando fallback para CPU...")
            try:
                self.device = 'cpu'
                self.embeddings_model = SentenceTransformer(self.model_name, device='cpu')
                print("✅ Modelo carregado na CPU como fallback")
            except Exception as e2:
                print(f"❌ Erro crítico: {e2}")
                raise
    
    def add_documents(self, chunked_docs: Dict[str, List[Dict]]):
        """Adiciona documentos ao vector store com otimização de memória."""
        print("🧠 Gerando embeddings...")
        
        # Preparar chunks
        all_chunks = []
        for doc_chunks in chunked_docs.values():
            all_chunks.extend(doc_chunks)
        
        self.chunks = all_chunks
        
        # Gerar embeddings em batches para economizar memória
        texts = [chunk['text'] for chunk in all_chunks]
        
        # Ajustar batch_size baseado no dispositivo
        if self.device == 'cuda':
            try:
                import torch
                # Calcular batch_size baseado na memória disponível
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 8e9:  # > 8GB
                    batch_size = 64
                elif gpu_memory > 4e9:  # > 4GB
                    batch_size = 32
                else:  # <= 4GB
                    batch_size = 16
            except:
                batch_size = 32
        else:
            batch_size = 16  # CPU mais conservador
        
        print(f"📊 Processando {len(texts)} textos em batches de {batch_size}")
        
        # Gerar embeddings com configurações otimizadas
        self.vectors = self.embeddings_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para cosine similarity
        )
        
        # Limpar cache se usando GPU
        if self.device == 'cuda':
            import torch
            torch.cuda.empty_cache()
        
        # Criar índice FAISS
        self._create_faiss_index()
        
        print(f"✅ {len(all_chunks)} chunks indexados!")
        print(f"📏 Dimensão dos embeddings: {self.vectors.shape[1]}")
        print(f"💾 Tamanho do índice: {self.vectors.nbytes / 1e6:.1f}MB")
    
    def _create_faiss_index(self):
        """Cria índice FAISS otimizado para busca rápida."""
        try:
            import faiss
            import numpy as np
            
            # Converter para float32 se necessário
            if self.vectors.dtype != np.float32:
                self.vectors = self.vectors.astype('float32')
            
            # Normalizar vetores para cosine similarity (já normalizado no encode)
            faiss.normalize_L2(self.vectors)
            
            # Criar índice otimizado baseado no tamanho
            dimension = self.vectors.shape[1]
            n_vectors = self.vectors.shape[0]
            
            if n_vectors > 10000:
                # Para grandes datasets, usar índice com clustering
                nlist = min(100, n_vectors // 100)  # Número de clusters
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.index.train(self.vectors)
                print(f"📊 Índice FAISS IVF criado (dimensão: {dimension}, clusters: {nlist})")
            else:
                # Para datasets menores, usar índice flat
                self.index = faiss.IndexFlatIP(dimension)
                print(f"📊 Índice FAISS Flat criado (dimensão: {dimension})")
            
            # Adicionar vetores ao índice
            self.index.add(self.vectors)
            
            # Configurar parâmetros de busca se for IVF
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(10, self.index.ntotal // 100)
            
            print(f"✅ {n_vectors} vetores indexados no FAISS")
            
        except ImportError:
            print("❌ FAISS não instalado. Usando busca linear.")
            self.index = None
        except Exception as e:
            print(f"⚠️ Erro ao criar índice FAISS: {e}")
            print("💡 Usando busca linear como fallback")
            self.index = None
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Busca documentos similares com otimização de memória."""
        try:
            # Gerar embedding da query com configurações otimizadas
            query_vector = self.embeddings_model.encode(
                [query], 
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1
            )
            
            # Limpar cache GPU se necessário
            if self.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
            
            if self.index is not None:
                # Busca com FAISS
                import faiss
                import numpy as np
                
                # Garantir tipo correto
                if query_vector.dtype != np.float32:
                    query_vector = query_vector.astype('float32')
                
                # Normalizar query (já normalizado no encode)
                faiss.normalize_L2(query_vector)
                
                # Buscar com tratamento de erro
                try:
                    scores, indices = self.index.search(query_vector, k)
                except Exception as e:
                    print(f"⚠️ Erro na busca FAISS: {e}")
                    return self._fallback_search(query_vector, k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.chunks):
                        chunk = self.chunks[idx].copy()
                        chunk['similarity_score'] = float(score)
                        results.append(chunk)
                
                return results
            else:
                # Busca linear (fallback)
                return self._fallback_search(query_vector, k)
                
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
    
    def _fallback_search(self, query_vector, k: int) -> List[Dict]:
        """Busca linear como fallback."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # Pegar top-k
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(similarities[idx])
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            print(f"❌ Erro na busca linear: {e}")
            return []

# ============================================================================
# 4. SISTEMA DE CLASSIFICAÇÃO INTELIGENTE
# ============================================================================

class QueryClassifier:
    """Classificador de intenções de consulta."""
    
    def __init__(self):
        self.categories = {
            'rag_required': {
                'keywords': [
                    'artigo', 'lei', 'código', 'direito', 'jurisprudência', 'norma',
                    'constituição', 'clt', 'cdc', 'prazo', 'multa', 'crime', 'pena',
                    'processo', 'recurso', 'habeas', 'mandado', 'ação', 'sentença'
                ],
                'patterns': [
                    r'o que (diz|fala|estabelece)',
                    r'qual (o|a) (prazo|multa|pena)',
                    r'como funciona',
                    r'quais são os direitos',
                    r'posso (fazer|pedir)',
                    r'é crime',
                    r'é legal'
                ]
            },
            'calculation_required': {
                'keywords': [
                    'calcular', 'cálculo', 'valor', 'quanto', 'porcentagem', '%',
                    'salário', 'indenização', 'multa', 'juros', 'correção'
                ],
                'patterns': [
                    r'quanto (vale|custa|é)',
                    r'como calcular',
                    r'qual o valor',
                    r'\d+%',
                    r'R\$\s*\d+'
                ]
            },
            'general_conversation': {
                'keywords': [
                    'olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'tchau',
                    'obrigado', 'obrigada', 'valeu', 'legal', 'bacana',
                    'quem é você', 'seu nome', 'como você funciona'
                ],
                'patterns': [
                    r'^(oi|olá|hey)',
                    r'(obrigad|valeu|legal)',
                    r'(quem é|o que é) você',
                    r'como você (funciona|trabalha)'
                ]
            }
        }
    
    def classify(self, query: str) -> str:
        """Classifica a query do usuário."""
        query_lower = query.lower()
        
        # Pontuação para cada categoria
        scores = {category: 0 for category in self.categories.keys()}
        
        for category, rules in self.categories.items():
            # Verificar keywords
            for keyword in rules['keywords']:
                if keyword in query_lower:
                    scores[category] += 1
            
            # Verificar patterns
            for pattern in rules['patterns']:
                if re.search(pattern, query_lower):
                    scores[category] += 2
        
        # Determinar categoria com maior pontuação
        if max(scores.values()) == 0:
            return 'out_of_scope'
        
        return max(scores, key=scores.get)
    
    def get_classification_confidence(self, query: str) -> Dict[str, float]:
        """Retorna confiança da classificação."""
        query_lower = query.lower()
        scores = {category: 0 for category in self.categories.keys()}
        
        for category, rules in self.categories.items():
            for keyword in rules['keywords']:
                if keyword in query_lower:
                    scores[category] += 1
            for pattern in rules['patterns']:
                if re.search(pattern, query_lower):
                    scores[category] += 2
        
        total_score = sum(scores.values())
        if total_score == 0:
            return {'out_of_scope': 1.0}
        
        return {cat: score/total_score for cat, score in scores.items()}

# ============================================================================
# 5. SISTEMA RAG (RETRIEVAL-AUGMENTED GENERATION)
# ============================================================================

class RAGSystem:
    """Sistema completo de RAG para consultas jurídicas."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.conversation_history = []
    
    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict], float]:
        """Recupera contexto relevante para a query."""
        # Buscar documentos similares
        results = self.vector_store.search(query, k=k)
        
        if not results:
            return "", [], 0.0
        
        # Calcular confiança média
        avg_confidence = np.mean([r['similarity_score'] for r in results])
        
        # Formatar contexto
        context_parts = []
        for i, result in enumerate(results, 1):
            doc_name = result['doc_name']
            section = result.get('section', 'Seção desconhecida')
            text = result['text']
            score = result['similarity_score']
            
            context_parts.append(
                f"[Fonte {i}: {doc_name} - {section} | Relevância: {score:.3f}]\n{text}"
            )
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        return context, results, avg_confidence
    
    def generate_response(self, query: str, context: str, confidence: float) -> Dict:
        """Gera resposta baseada no contexto recuperado."""
        # Sistema de resposta baseado em templates (simulando LLM)
        response_data = {
            'query': query,
            'context_used': len(context) > 0,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'sources': []
        }
        
        if confidence > 0.7:
            response_data['answer'] = self._generate_high_confidence_response(query, context)
            response_data['confidence_level'] = 'Alta'
        elif confidence > 0.4:
            response_data['answer'] = self._generate_medium_confidence_response(query, context)
            response_data['confidence_level'] = 'Média'
        else:
            response_data['answer'] = self._generate_low_confidence_response(query)
            response_data['confidence_level'] = 'Baixa'
        
        # Extrair fontes do contexto
        sources = re.findall(r'\[Fonte \d+: ([^|]+)', context)
        response_data['sources'] = list(set(sources))
        
        return response_data
    
    def _generate_high_confidence_response(self, query: str, context: str) -> str:
        """Gera resposta com alta confiança."""
        return f"""Com base na legislação brasileira encontrada, posso responder sua pergunta sobre: "{query}"

{self._extract_relevant_info(context)}

Esta resposta foi baseada em documentos oficiais com alta relevância para sua consulta."""
    
    def _generate_medium_confidence_response(self, query: str, context: str) -> str:
        """Gera resposta com média confiança."""
        return f"""Encontrei informações relacionadas à sua pergunta: "{query}"

{self._extract_relevant_info(context)}

⚠️ Recomendo verificar as fontes citadas, pois a correspondência com sua pergunta específica pode não ser completa."""
    
    def _generate_low_confidence_response(self, query: str) -> str:
        """Gera resposta com baixa confiança."""
        return f"""Sua pergunta "{query}" não encontrou correspondência direta na base de conhecimento disponível.

Isso pode acontecer porque:
- A informação específica não está nos documentos carregados
- A pergunta pode precisar ser reformulada
- O assunto pode estar fora do escopo da legislação disponível

Tente reformular sua pergunta ou seja mais específico sobre qual lei ou código você gostaria de consultar."""
    
    def _extract_relevant_info(self, context: str) -> str:
        """Extrai informações mais relevantes do contexto."""
        # Simplificação: pega os primeiros 500 caracteres de cada fonte
        sources = context.split('[Fonte')
        relevant_parts = []
        
        for source in sources[1:3]:  # Pegar apenas as 2 primeiras fontes
            if ']' in source:
                content = source.split(']', 1)[1].strip()
                if len(content) > 500:
                    content = content[:500] + "..."
                relevant_parts.append(content)
        
        return "\n\n".join(relevant_parts)

# ============================================================================
# 6. SISTEMA DE CONVERSAÇÃO E HISTÓRICO
# ============================================================================

class ConversationManager:
    """Gerencia histórico e contexto das conversas."""
    
    def __init__(self):
        self.conversations = {}
        self.current_session = None
        self.feedback_data = []
    
    def start_session(self, session_id: str = None) -> str:
        """Inicia uma nova sessão de conversa."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = session_id
        self.conversations[session_id] = {
            'messages': [],
            'start_time': datetime.now(),
            'metadata': {}
        }
        
        return session_id
    
    def add_message(self, message_type: str, content: str, metadata: Dict = None):
        """Adiciona mensagem ao histórico."""
        if self.current_session is None:
            self.start_session()
        
        message = {
            'type': message_type,  # 'user' ou 'assistant'
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.conversations[self.current_session]['messages'].append(message)
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Retorna contexto da conversa atual."""
        if self.current_session is None:
            return ""
        
        messages = self.conversations[self.current_session]['messages']
        recent_messages = messages[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            role = "Usuário" if msg['type'] == 'user' else "Assistente"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def add_feedback(self, message_id: int, rating: int, comment: str = ""):
        """Adiciona feedback do usuário."""
        feedback = {
            'session_id': self.current_session,
            'message_id': message_id,
            'rating': rating,  # 1-5
            'comment': comment,
            'timestamp': datetime.now()
        }
        
        self.feedback_data.append(feedback)
    
    def get_session_stats(self) -> Dict:
        """Retorna estatísticas da sessão atual."""
        if self.current_session is None:
            return {}
        
        session = self.conversations[self.current_session]
        messages = session['messages']
        
        user_messages = [m for m in messages if m['type'] == 'user']
        assistant_messages = [m for m in messages if m['type'] == 'assistant']
        
        return {
            'session_id': self.current_session,
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'duration': (datetime.now() - session['start_time']).total_seconds(),
            'avg_response_confidence': np.mean([
                m['metadata'].get('confidence', 0) 
                for m in assistant_messages 
                if 'confidence' in m.get('metadata', {})
            ]) if assistant_messages else 0
        }

# ============================================================================
# 7. SISTEMA PRINCIPAL DO CHATBOT
# ============================================================================

class ChatbotJuridicoInteligente:
    """Sistema principal do chatbot jurídico inteligente."""
    
    def __init__(self, data_dir: str = "dados"):
        print("🚀 Inicializando Chatbot Jurídico Inteligente...")
        
        # Gerenciador de memória GPU
        self.gpu_manager = GPUMemoryManager()
        gpu_info = self.gpu_manager.get_gpu_info()
        
        if gpu_info["available"]:
            print(f"🎮 GPU detectada: {gpu_info['name']}")
            print(f"💾 Memória total: {gpu_info['total_memory']/1e9:.1f}GB")
        else:
            print("💻 Executando em CPU")
        
        # Componentes principais
        self.doc_processor = DocumentProcessor(data_dir)
        self.chunker = IntelligentChunker()
        self.vector_store = VectorStore()
        self.classifier = QueryClassifier()
        self.rag_system = None
        self.conversation_manager = ConversationManager()
        
        # Métricas e logs
        self.performance_metrics = {
            'total_queries': 0,
            'rag_queries': 0,
            'general_queries': 0,
            'out_of_scope_queries': 0,
            'avg_response_time': 0,
            'classification_accuracy': []
        }
        
        # Inicializar sistema
        self._initialize_system()
    
    def _initialize_system(self):
        """Inicializa todos os componentes do sistema com monitoramento de memória."""
        try:
            print(f"\n📊 Memória inicial: {self.gpu_manager.monitor_memory()}")
            
            # 1. Carregar documentos
            print("\n📚 Carregando documentos...")
            documents = self.doc_processor.load_text_documents()
            
            # 2. Limpar textos
            print("\n🧹 Limpando textos...")
            cleaned_docs = {
                name: self.doc_processor.clean_text(content) 
                for name, content in documents.items()
            }
            
            # Limpar cache após processamento de texto
            self.gpu_manager.clear_cache()
            
            # 3. Fazer chunking
            print("\n🔪 Fazendo chunking...")
            chunked_docs = self.chunker.chunk_documents(cleaned_docs)
            
            print(f"\n📊 Memória após chunking: {self.gpu_manager.monitor_memory()}")
            
            # 4. Criar vector store (processo mais pesado)
            print("\n🧠 Criando vector store...")
            self.vector_store.add_documents(chunked_docs)
            
            print(f"\n📊 Memória após embeddings: {self.gpu_manager.monitor_memory()}")
            
            # 5. Inicializar RAG
            print("\n🔍 Inicializando sistema RAG...")
            self.rag_system = RAGSystem(self.vector_store)
            
            # Limpeza final
            self.gpu_manager.clear_cache()
            
            print("\n✅ Sistema inicializado com sucesso!")
            print(f"📚 Base de conhecimento: {len(documents)} documentos")
            print(f"🧩 Total de chunks: {sum(len(chunks) for chunks in chunked_docs.values())}")
            print(f"📊 Memória final: {self.gpu_manager.monitor_memory()}")
            
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            print("🧹 Limpando memória...")
            self.gpu_manager.clear_cache()
            raise
    
    def process_query(self, query: str, session_id: str = None) -> Dict:
        """Processa uma consulta do usuário."""
        start_time = time.time()
        
        # Gerenciar sessão
        if session_id:
            self.conversation_manager.current_session = session_id
        elif self.conversation_manager.current_session is None:
            self.conversation_manager.start_session()
        
        # Adicionar mensagem do usuário
        self.conversation_manager.add_message('user', query)
        
        # 1. Classificar query
        classification = self.classifier.classify(query)
        classification_confidence = self.classifier.get_classification_confidence(query)
        
        # 2. Processar baseado na classificação
        response_data = self._handle_classified_query(query, classification)
        
        # 3. Adicionar metadados
        response_data.update({
            'classification': classification,
            'classification_confidence': classification_confidence,
            'processing_time': time.time() - start_time,
            'session_id': self.conversation_manager.current_session
        })
        
        # 4. Adicionar resposta ao histórico
        self.conversation_manager.add_message(
            'assistant', 
            response_data['answer'], 
            response_data
        )
        
        # 5. Atualizar métricas
        self._update_metrics(classification, response_data['processing_time'])
        
        return response_data
    
    def _handle_classified_query(self, query: str, classification: str) -> Dict:
        """Processa query baseado na classificação."""
        
        if classification == 'rag_required':
            return self._handle_rag_query(query)
        
        elif classification == 'calculation_required':
            return self._handle_calculation_query(query)
        
        elif classification == 'general_conversation':
            return self._handle_general_conversation(query)
        
        else:  # out_of_scope
            return self._handle_out_of_scope_query(query)
    
    def _handle_rag_query(self, query: str) -> Dict:
        """Processa consulta que requer RAG."""
        # Recuperar contexto
        context, results, confidence = self.rag_system.retrieve_context(query, k=5)
        
        # Gerar resposta
        response_data = self.rag_system.generate_response(query, context, confidence)
        
        # Adicionar disclaimer
        response_data['answer'] += self._get_legal_disclaimer()
        
        return response_data
    
    def _handle_calculation_query(self, query: str) -> Dict:
        """Processa consulta que requer cálculo."""
        return {
            'answer': f"""Sua pergunta "{query}" parece requerer um cálculo específico.

⚠️ IMPORTANTE: Cálculos jurídicos dependem de diversos fatores específicos de cada caso, como:
- Valores atualizados de salários e índices
- Datas específicas para aplicação de correções
- Circunstâncias particulares do caso

🔍 RECOMENDAÇÃO: 
1. Consulte um profissional do direito para cálculos precisos
2. Verifique os valores atualizados na legislação vigente
3. Considere as particularidades do seu caso específico

Posso ajudar com informações sobre as regras gerais de cálculo se você reformular sua pergunta.""",
            'context_used': False,
            'confidence': 0.8,
            'sources': []
        }
    
    def _handle_general_conversation(self, query: str) -> Dict:
        """Processa conversa geral."""
        responses = {
            'greeting': "Olá! Sou seu assistente jurídico inteligente. Posso ajudar com consultas sobre a legislação brasileira. Como posso ajudá-lo hoje?",
            'thanks': "De nada! Fico feliz em ajudar. Se tiver mais dúvidas sobre legislação brasileira, estarei aqui!",
            'about': "Sou um chatbot especializado em legislação brasileira. Tenho acesso a diversos códigos e leis do Brasil e posso ajudar a esclarecer dúvidas jurídicas básicas.",
            'default': "Entendi! Estou aqui para ajudar com questões jurídicas. Você pode me perguntar sobre leis, códigos, direitos ou procedimentos legais no Brasil."
        }
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['oi', 'olá', 'bom dia', 'boa tarde']):
            answer = responses['greeting']
        elif any(word in query_lower for word in ['obrigad', 'valeu', 'legal']):
            answer = responses['thanks']
        elif any(word in query_lower for word in ['quem é', 'o que é', 'como funciona']):
            answer = responses['about']
        else:
            answer = responses['default']
        
        return {
            'answer': answer,
            'context_used': False,
            'confidence': 0.9,
            'sources': []
        }
    
    def _handle_out_of_scope_query(self, query: str) -> Dict:
        """Processa consulta fora do escopo."""
        return {
            'answer': f"""Desculpe, mas sua pergunta "{query}" parece estar fora do meu escopo de conhecimento jurídico.

🎯 EU POSSO AJUDAR COM:
- Consultas sobre leis e códigos brasileiros
- Esclarecimentos sobre direitos e deveres
- Informações sobre procedimentos legais
- Prazos e normas jurídicas

❌ NÃO POSSO AJUDAR COM:
- Assuntos não relacionados ao direito
- Conselhos médicos, financeiros ou técnicos
- Informações sobre outros países
- Temas fora da legislação brasileira

💡 DICA: Reformule sua pergunta focando em aspectos jurídicos ou legais do Brasil.""",
            'context_used': False,
            'confidence': 0.95,
            'sources': []
        }
    
    def _get_legal_disclaimer(self) -> str:
        """Retorna disclaimer legal padrão."""
        return """

⚖️ AVISO LEGAL IMPORTANTE:
Esta resposta foi gerada por inteligência artificial e tem caráter meramente informativo. 
NÃO substitui a consulta a um advogado ou profissional do direito qualificado.
Para casos específicos, sempre procure orientação jurídica profissional.
As informações podem estar desatualizadas ou incompletas."""
    
    def _update_metrics(self, classification: str, processing_time: float):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_queries'] += 1
        
        if classification == 'rag_required':
            self.performance_metrics['rag_queries'] += 1
        elif classification == 'general_conversation':
            self.performance_metrics['general_queries'] += 1
        elif classification == 'out_of_scope':
            self.performance_metrics['out_of_scope_queries'] += 1
        
        # Atualizar tempo médio de resposta
        current_avg = self.performance_metrics['avg_response_time']
        total_queries = self.performance_metrics['total_queries']
        
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    def get_system_stats(self) -> Dict:
        """Retorna estatísticas do sistema."""
        return {
            'performance_metrics': self.performance_metrics,
            'conversation_stats': self.conversation_manager.get_session_stats(),
            'system_info': {
                'documents_loaded': len(self.doc_processor.documents),
                'total_chunks': len(self.vector_store.chunks),
                'embedding_model': self.vector_store.model_name,
                'vector_dimension': self.vector_store.vectors.shape[1] if self.vector_store.vectors is not None else 0
            }
        }
    
    def add_feedback(self, rating: int, comment: str = ""):
        """Adiciona feedback do usuário."""
        if self.conversation_manager.current_session:
            messages = self.conversation_manager.conversations[self.conversation_manager.current_session]['messages']
            last_message_id = len(messages) - 1
            self.conversation_manager.add_feedback(last_message_id, rating, comment)
    
    def chat_interface(self):
        """Interface de chat simples para testes."""
        print("\n" + "="*60)
        print("🤖 CHATBOT JURÍDICO INTELIGENTE")
        print("="*60)
        print("Digite 'sair' para encerrar")
        print("Digite 'stats' para ver estatísticas")
        print("Digite 'feedback [1-5]' para avaliar a última resposta")
        print("="*60)
        
        session_id = self.conversation_manager.start_session()
        print(f"📱 Sessão iniciada: {session_id}")
        
        while True:
            try:
                user_input = input("\n👤 Você: ").strip()
                
                if user_input.lower() == 'sair':
                    print("👋 Até logo!")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = self.get_system_stats()
                    print("\n📊 ESTATÍSTICAS DO SISTEMA:")
                    print(f"Total de consultas: {stats['performance_metrics']['total_queries']}")
                    print(f"Consultas RAG: {stats['performance_metrics']['rag_queries']}")
                    print(f"Tempo médio de resposta: {stats['performance_metrics']['avg_response_time']:.2f}s")
                    continue
                
                elif user_input.lower().startswith('feedback'):
                    try:
                        rating = int(user_input.split()[1])
                        if 1 <= rating <= 5:
                            self.add_feedback(rating)
                            print(f"✅ Feedback registrado: {rating}/5")
                        else:
                            print("❌ Rating deve ser entre 1 e 5")
                    except:
                        print("❌ Formato: feedback [1-5]")
                    continue
                
                elif not user_input:
                    continue
                
                # Processar consulta
                print("\n🤔 Processando...")
                response = self.process_query(user_input, session_id)
                
                # Mostrar resposta
                print(f"\n🤖 Assistente:")
                print(f"{response['answer']}")
                
                # Mostrar metadados
                print(f"\n📊 Classificação: {response['classification']}")
                print(f"⏱️ Tempo de processamento: {response['processing_time']:.2f}s")
                if response['sources']:
                    print(f"📚 Fontes: {', '.join(response['sources'][:3])}")
                
            except KeyboardInterrupt:
                print("\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    # Exemplo de uso
    try:
        chatbot = ChatbotJuridicoInteligente()
        chatbot.chat_interface()
    except Exception as e:
        print(f"❌ Erro ao inicializar chatbot: {e}")