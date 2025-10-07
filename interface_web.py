# -*- coding: utf-8 -*-
"""
Interface Web para Chatbot Jurídico Inteligente
Projeto Final - Pós-Graduação iCEV - NLP

Interface moderna e intuitiva usando Gradio com:
1. Chat conversacional com histórico
2. Sistema de feedback
3. Métricas em tempo real
4. Explicações sobre classificação
5. Indicadores de confiança
"""

import gradio as gr
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict
import pandas as pd

# ============================================================================
# 1. CLASSE DA INTERFACE WEB
# ============================================================================

class ChatbotWebInterface:
    """Interface web completa para o chatbot jurídico."""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.session_stats = {
            'total_messages': 0,
            'rag_queries': 0,
            'general_queries': 0,
            'out_of_scope_queries': 0,
            'avg_confidence': 0,
            'feedback_ratings': []
        }
        
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str, str, str]:
        """
        Processa mensagem do usuário e retorna resposta formatada.
        
        Returns:
            - response: Resposta do chatbot
            - updated_history: Histórico atualizado
            - classification_info: Informações sobre classificação
            - confidence_info: Informações sobre confiança
            - sources_info: Informações sobre fontes
        """
        if not message.strip():
            return "", history, "", "", ""
        
        # Processar consulta
        start_time = time.time()
        response_data = self.chatbot.process_query(message)
        processing_time = time.time() - start_time
        
        # Atualizar estatísticas
        self._update_session_stats(response_data)
        
        # Formatar resposta
        formatted_response = self._format_response(response_data, processing_time)
        
        # Atualizar histórico
        history.append((message, formatted_response))
        
        # Preparar informações adicionais
        classification_info = self._format_classification_info(response_data)
        confidence_info = self._format_confidence_info(response_data)
        sources_info = self._format_sources_info(response_data)
        
        return "", history, classification_info, confidence_info, sources_info
    
    def _format_response(self, response_data: Dict, processing_time: float) -> str:
        """Formata a resposta do chatbot para exibição."""
        response = response_data['answer']
        
        # Adicionar informações de processamento
        footer = f"\n\n---\n"
        footer += f"⏱️ Processado em {processing_time:.2f}s | "
        footer += f"🎯 Classificação: {response_data['classification']} | "
        footer += f"📊 Confiança: {response_data.get('confidence', 0):.2f}"
        
        if response_data.get('sources'):
            footer += f" | 📚 {len(response_data['sources'])} fonte(s)"
        
        return response + footer
    
    def _format_classification_info(self, response_data: Dict) -> str:
        """Formata informações sobre classificação."""
        classification = response_data['classification']
        confidence_scores = response_data.get('classification_confidence', {})
        
        info = f"🎯 **Classificação:** {classification}\n\n"
        
        # Explicar o que significa cada classificação
        explanations = {
            'rag_required': "Esta pergunta requer busca na base de conhecimento jurídico.",
            'calculation_required': "Esta pergunta parece requerer cálculos específicos.",
            'general_conversation': "Esta é uma conversa geral, não requer consulta jurídica.",
            'out_of_scope': "Esta pergunta está fora do escopo jurídico do sistema."
        }
        
        info += f"**Explicação:** {explanations.get(classification, 'Classificação não reconhecida')}\n\n"
        
        # Mostrar scores de confiança para cada categoria
        if confidence_scores:
            info += "**Scores por categoria:**\n"
            for category, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                percentage = score * 100
                info += f"- {category}: {percentage:.1f}%\n"
        
        return info
    
    def _format_confidence_info(self, response_data: Dict) -> str:
        """Formata informações sobre confiança."""
        confidence = response_data.get('confidence', 0)
        confidence_level = response_data.get('confidence_level', 'Desconhecida')
        
        # Emoji baseado na confiança
        if confidence > 0.8:
            emoji = "🟢"
            color = "green"
        elif confidence > 0.6:
            emoji = "🟡"
            color = "orange"
        else:
            emoji = "🔴"
            color = "red"
        
        info = f"{emoji} **Nível de Confiança:** {confidence_level}\n\n"
        info += f"**Score numérico:** {confidence:.3f}\n\n"
        
        # Explicar o que significa o nível de confiança
        if confidence > 0.8:
            info += "**Interpretação:** Alta confiança - a resposta provavelmente está correta e bem fundamentada."
        elif confidence > 0.6:
            info += "**Interpretação:** Confiança moderada - a resposta pode estar correta, mas recomenda-se verificação."
        elif confidence > 0.3:
            info += "**Interpretação:** Baixa confiança - a resposta pode não estar completa ou precisa."
        else:
            info += "**Interpretação:** Confiança muito baixa - recomenda-se reformular a pergunta."
        
        return info
    
    def _format_sources_info(self, response_data: Dict) -> str:
        """Formata informações sobre fontes utilizadas."""
        sources = response_data.get('sources', [])
        context_used = response_data.get('context_used', False)
        
        if not context_used or not sources:
            return "📚 **Fontes:** Nenhuma fonte específica foi consultada para esta resposta."
        
        info = f"📚 **Fontes consultadas:** {len(sources)} documento(s)\n\n"
        
        for i, source in enumerate(sources[:5], 1):  # Mostrar até 5 fontes
            # Limpar nome da fonte
            clean_source = source.replace('_', ' ').replace('.txt', '')
            info += f"{i}. {clean_source}\n"
        
        if len(sources) > 5:
            info += f"\n... e mais {len(sources) - 5} fonte(s)."
        
        info += "\n\n**Nota:** As fontes listadas foram utilizadas para formular a resposta baseada na legislação brasileira."
        
        return info
    
    def _update_session_stats(self, response_data: Dict):
        """Atualiza estatísticas da sessão."""
        self.session_stats['total_messages'] += 1
        
        classification = response_data['classification']
        if classification == 'rag_required':
            self.session_stats['rag_queries'] += 1
        elif classification == 'general_conversation':
            self.session_stats['general_queries'] += 1
        elif classification == 'out_of_scope':
            self.session_stats['out_of_scope_queries'] += 1
        
        # Atualizar confiança média
        confidence = response_data.get('confidence', 0)
        current_avg = self.session_stats['avg_confidence']
        total_msgs = self.session_stats['total_messages']
        
        self.session_stats['avg_confidence'] = (
            (current_avg * (total_msgs - 1) + confidence) / total_msgs
        )
    
    def add_feedback(self, rating: int, comment: str = "") -> str:
        """Adiciona feedback do usuário."""
        if not (1 <= rating <= 5):
            return "❌ Por favor, selecione uma avaliação entre 1 e 5 estrelas."
        
        # Adicionar ao chatbot
        self.chatbot.add_feedback(rating, comment)
        
        # Atualizar estatísticas locais
        self.session_stats['feedback_ratings'].append(rating)
        
        return f"✅ Obrigado pelo seu feedback! Avaliação: {rating}/5 estrelas"
    
    def get_session_statistics(self) -> str:
        """Retorna estatísticas da sessão atual."""
        stats = self.session_stats
        
        info = "📊 **Estatísticas da Sessão**\n\n"
        info += f"**Total de mensagens:** {stats['total_messages']}\n"
        info += f"**Consultas jurídicas (RAG):** {stats['rag_queries']}\n"
        info += f"**Conversas gerais:** {stats['general_queries']}\n"
        info += f"**Fora de escopo:** {stats['out_of_scope_queries']}\n"
        info += f"**Confiança média:** {stats['avg_confidence']:.3f}\n"
        
        if stats['feedback_ratings']:
            avg_rating = sum(stats['feedback_ratings']) / len(stats['feedback_ratings'])
            info += f"**Avaliação média:** {avg_rating:.1f}/5 ({len(stats['feedback_ratings'])} avaliações)\n"
        
        return info
    
    def clear_conversation(self) -> Tuple[List, str, str, str]:
        """Limpa a conversa e reinicia a sessão."""
        # Reiniciar sessão no chatbot
        self.chatbot.conversation_manager.start_session()
        
        # Resetar estatísticas
        self.session_stats = {
            'total_messages': 0,
            'rag_queries': 0,
            'general_queries': 0,
            'out_of_scope_queries': 0,
            'avg_confidence': 0,
            'feedback_ratings': []
        }
        
        return [], "", "", ""
    
    def get_example_questions(self) -> List[str]:
        """Retorna lista de perguntas exemplo."""
        return [
            "O que diz o artigo 5º da Constituição Federal?",
            "Qual o prazo para reclamar de vícios em produtos duráveis?",
            "Como funciona a jornada de trabalho na CLT?",
            "O que é considerado união estável?",
            "Quais são os direitos do consumidor?",
            "Como calcular rescisão trabalhista?",
            "O que estabelece a LGPD sobre proteção de dados?",
            "Olá, como você funciona?",
            "Obrigado pela ajuda!"
        ]

# ============================================================================
# 2. CRIAÇÃO DA INTERFACE GRADIO
# ============================================================================

def create_gradio_interface(chatbot):
    """Cria interface Gradio completa."""
    
    # Inicializar interface
    web_interface = ChatbotWebInterface(chatbot)
    
    # CSS customizado
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    .info-panel {
        background-color: #fafafa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    """
    
    # Criar interface
    with gr.Blocks(
        title="Chatbot Jurídico Inteligente",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Cabeçalho
        gr.Markdown("""
        # ⚖️ Chatbot Jurídico Inteligente
        
        **Sistema RAG com Classificação Automática de Intenções**
        
        Faça perguntas sobre a legislação brasileira e receba respostas fundamentadas em documentos oficiais.
        
        ---
        """)
        
        with gr.Row():
            # Coluna principal - Chat
            with gr.Column(scale=2):
                chatbot_interface = gr.Chatbot(
                    label="Conversa",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Digite sua pergunta jurídica aqui...",
                        label="Sua mensagem",
                        lines=2,
                        max_lines=5,
                        show_label=False,
                        container=False
                    )
                    send_btn = gr.Button("Enviar", variant="primary", size="sm")
                
                # Botões de ação
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Limpar Conversa", size="sm")
                    stats_btn = gr.Button("📊 Ver Estatísticas", size="sm")
                
                # Exemplos de perguntas
                gr.Examples(
                    examples=web_interface.get_example_questions(),
                    inputs=msg_input,
                    label="💡 Perguntas de exemplo:"
                )
            
            # Coluna lateral - Informações
            with gr.Column(scale=1):
                # Painel de classificação
                classification_panel = gr.Markdown(
                    "🎯 **Classificação da última pergunta aparecerá aqui**",
                    label="Classificação"
                )
                
                # Painel de confiança
                confidence_panel = gr.Markdown(
                    "📊 **Informações de confiança aparecerão aqui**",
                    label="Confiança"
                )
                
                # Painel de fontes
                sources_panel = gr.Markdown(
                    "📚 **Fontes consultadas aparecerão aqui**",
                    label="Fontes"
                )
                
                # Sistema de feedback
                gr.Markdown("### 💬 Feedback")
                with gr.Row():
                    rating_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=5,
                        label="Avaliação (1-5 estrelas)"
                    )
                
                feedback_comment = gr.Textbox(
                    placeholder="Comentário opcional...",
                    label="Comentário",
                    lines=2,
                    max_lines=3
                )
                
                feedback_btn = gr.Button("Enviar Feedback", variant="secondary", size="sm")
                feedback_status = gr.Markdown("")
                
                # Estatísticas da sessão
                session_stats = gr.Markdown(
                    "📊 **Estatísticas da sessão aparecerão aqui**",
                    label="Estatísticas"
                )
        
        # Rodapé com informações
        gr.Markdown("""
        ---
        
        ### ℹ️ Como usar:
        
        1. **Digite sua pergunta** sobre legislação brasileira
        2. **Aguarde a resposta** com classificação automática
        3. **Verifique as fontes** consultadas no painel lateral
        4. **Avalie a resposta** para ajudar a melhorar o sistema
        
        ### ⚠️ Importante:
        - Este sistema tem fins educacionais e informativos
        - Não substitui consulta jurídica profissional
        - Sempre verifique informações com um advogado qualificado
        
        ### 🔧 Funcionalidades:
        - ✅ Classificação automática de intenções
        - ✅ Sistema RAG com busca semântica
        - ✅ Indicadores de confiança
        - ✅ Rastreamento de fontes
        - ✅ Histórico de conversas
        - ✅ Sistema de feedback
        """)
        
        # Eventos da interface
        
        # Enviar mensagem
        def handle_message(message, history):
            return web_interface.process_message(message, history)
        
        # Configurar eventos
        msg_input.submit(
            handle_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface, classification_panel, confidence_panel, sources_panel]
        )
        
        send_btn.click(
            handle_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface, classification_panel, confidence_panel, sources_panel]
        )
        
        # Limpar conversa
        clear_btn.click(
            web_interface.clear_conversation,
            outputs=[chatbot_interface, classification_panel, confidence_panel, sources_panel]
        )
        
        # Ver estatísticas
        stats_btn.click(
            web_interface.get_session_statistics,
            outputs=session_stats
        )
        
        # Enviar feedback
        feedback_btn.click(
            web_interface.add_feedback,
            inputs=[rating_slider, feedback_comment],
            outputs=feedback_status
        )
    
    return interface

# ============================================================================
# 3. FUNÇÃO PRINCIPAL PARA EXECUTAR A INTERFACE
# ============================================================================

def launch_web_interface(chatbot, share=False, server_port=7860):
    """Lança a interface web do chatbot."""
    
    print("🌐 Criando interface web...")
    interface = create_gradio_interface(chatbot)
    
    print("🚀 Iniciando servidor web...")
    print(f"📱 Interface disponível em: http://localhost:{server_port}")
    
    if share:
        print("🌍 Link público será gerado...")
    
    # Lançar interface
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",  # Permite acesso externo
        show_error=True,
        quiet=False,
        favicon_path=None,
        ssl_verify=False
    )

# ============================================================================
# 4. EXEMPLO DE USO
# ============================================================================

def main():
    """Função principal para executar a interface web."""
    try:
        # Importar e inicializar chatbot
        from chatbot_juridico_inteligente import ChatbotJuridicoInteligente
        
        print("🤖 Inicializando Chatbot Jurídico Inteligente...")
        chatbot = ChatbotJuridicoInteligente()
        
        print("✅ Chatbot inicializado com sucesso!")
        
        # Lançar interface web
        launch_web_interface(
            chatbot=chatbot,
            share=False,  # Mude para True se quiser link público
            server_port=7860
        )
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        print("💡 Verifique se todos os arquivos estão no local correto")

if __name__ == "__main__":
    main()