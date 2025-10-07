# -*- coding: utf-8 -*-
"""
Sistema de Avaliação e Métricas para Chatbot Jurídico
Projeto Final - Pós-Graduação iCEV - NLP

Este módulo implementa:
1. Testes automatizados do classificador
2. Avaliação do sistema RAG
3. Métricas de performance
4. Análise de casos extremos
5. Relatórios detalhados
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. DATASETS DE TESTE
# ============================================================================

class TestDatasets:
    """Datasets padronizados para avaliação do sistema."""
    
    @staticmethod
    def get_classification_dataset() -> Dict[str, List[str]]:
        """Dataset para teste do classificador de intenções."""
        return {
            'rag_required': [
                "O que diz o artigo 5º da Constituição Federal?",
                "Qual o prazo para reclamar de vícios em produtos duráveis segundo o CDC?",
                "Como funciona o habeas corpus no Brasil?",
                "Quais são os direitos do trabalhador na CLT?",
                "O que estabelece o Código Civil sobre contratos?",
                "É crime dirigir sem habilitação?",
                "Qual a pena para furto no Código Penal?",
                "Como funciona a usucapião de imóveis?",
                "O que é considerado união estável?",
                "Quais são os direitos do consumidor?",
                "Como funciona o processo de divórcio?",
                "O que diz a lei sobre assédio moral no trabalho?",
                "Qual o prazo prescricional para crimes contra a honra?",
                "Como funciona a licença maternidade?",
                "O que estabelece a LGPD sobre proteção de dados?",
                "Quais são os crimes ambientais previstos em lei?",
                "Como funciona o mandado de segurança?",
                "O que diz a lei sobre violência doméstica?",
                "Qual o procedimento para registro de marca?",
                "Como funciona a aposentadoria por invalidez?",
                "O que estabelece o ECA sobre direitos da criança?",
                "Qual a diferença entre furto e roubo?",
                "Como funciona a ação de despejo?",
                "O que diz a lei sobre direito autoral?",
                "Qual o prazo para contestar uma multa de trânsito?"
            ],
            'calculation_required': [
                "Como calcular o valor da rescisão trabalhista?",
                "Quanto vale 1/3 de férias sobre um salário de R$ 3000?",
                "Qual o valor da multa de 40% do FGTS?",
                "Como calcular juros de mora em contratos?",
                "Quanto é a multa por dirigir sem CNH?",
                "Como calcular indenização por danos morais?",
                "Qual o valor do adicional noturno?",
                "Como calcular horas extras?",
                "Quanto vale o auxílio-doença?",
                "Como calcular a pensão alimentícia?",
                "Qual o valor da taxa judiciária?",
                "Como calcular correção monetária?",
                "Quanto custa registrar uma empresa?",
                "Qual o valor do salário família?",
                "Como calcular o 13º salário proporcional?"
            ],
            'general_conversation': [
                "Olá, como você está?",
                "Bom dia! Tudo bem?",
                "Oi, qual é o seu nome?",
                "Muito obrigado pela ajuda!",
                "Valeu, foi muito útil!",
                "Quem criou você?",
                "Como você funciona?",
                "Você é um robô?",
                "Que legal, adorei!",
                "Tchau, até mais!",
                "Boa tarde!",
                "Boa noite!",
                "Obrigada!",
                "Legal, bacana!",
                "Interessante!"
            ],
            'out_of_scope': [
                "Qual a previsão do tempo para amanhã?",
                "Me conte uma piada engraçada",
                "Qual o melhor time de futebol do Brasil?",
                "Como fazer um bolo de chocolate?",
                "Qual a cotação do dólar hoje?",
                "Como chegar ao shopping center?",
                "Qual o melhor restaurante da cidade?",
                "Como instalar o Windows?",
                "Qual a música mais tocada?",
                "Como emagrecer rapidamente?",
                "Qual o significado da vida?",
                "Como aprender inglês?",
                "Qual a capital da França?",
                "Como funciona um motor de carro?",
                "Qual o melhor celular para comprar?"
            ]
        }
    
    @staticmethod
    def get_rag_evaluation_dataset() -> List[Dict]:
        """Dataset para avaliação do sistema RAG."""
        return [
            {
                "question": "Qual o prazo para reclamar de vícios em produtos duráveis?",
                "expected_keywords": ["noventa dias", "90 dias", "produtos duráveis", "vícios"],
                "expected_source": "Código_de_Defesa_do_Consumidor",
                "difficulty": "easy"
            },
            {
                "question": "O que estabelece o artigo 5º da Constituição sobre direitos fundamentais?",
                "expected_keywords": ["igualdade", "liberdade", "vida", "segurança", "propriedade"],
                "expected_source": "Constituição_Federal",
                "difficulty": "medium"
            },
            {
                "question": "Como funciona a jornada de trabalho segundo a CLT?",
                "expected_keywords": ["8 horas", "44 horas", "jornada", "trabalho"],
                "expected_source": "Consolidação_das_Leis_do_Trabalho",
                "difficulty": "medium"
            },
            {
                "question": "O que é considerado união estável no Código Civil?",
                "expected_keywords": ["união estável", "convivência", "família", "público"],
                "expected_source": "Código_Civil",
                "difficulty": "hard"
            },
            {
                "question": "Quais são os crimes contra a honra no Código Penal?",
                "expected_keywords": ["calúnia", "difamação", "injúria", "honra"],
                "expected_source": "Código_Penal",
                "difficulty": "hard"
            }
        ]
    
    @staticmethod
    def get_edge_cases_dataset() -> List[Dict]:
        """Dataset para teste de casos extremos."""
        return [
            {
                "question": "",
                "type": "empty_query",
                "expected_behavior": "handle_gracefully"
            },
            {
                "question": "a" * 1000,
                "type": "very_long_query",
                "expected_behavior": "handle_gracefully"
            },
            {
                "question": "!@#$%^&*()",
                "type": "special_characters",
                "expected_behavior": "out_of_scope"
            },
            {
                "question": "artigo lei código direito mas não faz sentido nenhum",
                "type": "keywords_without_meaning",
                "expected_behavior": "low_confidence"
            },
            {
                "question": "Me fale sobre tudo relacionado ao direito brasileiro",
                "type": "overly_broad",
                "expected_behavior": "request_specificity"
            }
        ]

# ============================================================================
# 2. AVALIADOR DO CLASSIFICADOR
# ============================================================================

class ClassifierEvaluator:
    """Avalia a performance do classificador de intenções."""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.test_results = []
    
    def run_classification_tests(self) -> Dict:
        """Executa testes completos do classificador."""
        print("🧪 Iniciando avaliação do classificador...")
        
        dataset = TestDatasets.get_classification_dataset()
        true_labels = []
        predicted_labels = []
        
        total_queries = sum(len(queries) for queries in dataset.values())
        current_query = 0
        
        for true_category, queries in dataset.items():
            for query in queries:
                current_query += 1
                print(f"Testando {current_query}/{total_queries}: {query[:50]}...")
                
                # Classificar query
                predicted_category = self.chatbot.classifier.classify(query)
                confidence = self.chatbot.classifier.get_classification_confidence(query)
                
                # Armazenar resultados
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                
                self.test_results.append({
                    'query': query,
                    'true_label': true_category,
                    'predicted_label': predicted_category,
                    'confidence': confidence,
                    'correct': true_category == predicted_category
                })
        
        # Calcular métricas
        accuracy = sum(1 for r in self.test_results if r['correct']) / len(self.test_results)
        
        # Relatório detalhado
        report = classification_report(
            true_labels, 
            predicted_labels, 
            output_dict=True,
            zero_division=0
        )
        
        # Matriz de confusão
        cm = confusion_matrix(true_labels, predicted_labels)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'detailed_results': self.test_results,
            'categories': list(dataset.keys())
        }
        
        print(f"✅ Avaliação concluída! Acurácia: {accuracy:.3f}")
        return results
    
    def analyze_classification_errors(self) -> Dict:
        """Analisa erros de classificação em detalhes."""
        errors = [r for r in self.test_results if not r['correct']]
        
        if not errors:
            return {"message": "Nenhum erro de classificação encontrado!"}
        
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(self.test_results),
            'errors_by_category': {},
            'common_misclassifications': {},
            'examples': errors[:10]  # Primeiros 10 erros
        }
        
        # Análise por categoria
        for error in errors:
            true_cat = error['true_label']
            pred_cat = error['predicted_label']
            
            if true_cat not in error_analysis['errors_by_category']:
                error_analysis['errors_by_category'][true_cat] = 0
            error_analysis['errors_by_category'][true_cat] += 1
            
            # Pares de erro comum
            error_pair = f"{true_cat} -> {pred_cat}"
            if error_pair not in error_analysis['common_misclassifications']:
                error_analysis['common_misclassifications'][error_pair] = 0
            error_analysis['common_misclassifications'][error_pair] += 1
        
        return error_analysis

# ============================================================================
# 3. AVALIADOR DO SISTEMA RAG
# ============================================================================

class RAGEvaluator:
    """Avalia a performance do sistema RAG."""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.rag_results = []
    
    def run_rag_evaluation(self) -> Dict:
        """Executa avaliação completa do sistema RAG."""
        print("🔍 Iniciando avaliação do sistema RAG...")
        
        dataset = TestDatasets.get_rag_evaluation_dataset()
        
        for i, test_case in enumerate(dataset, 1):
            print(f"Testando RAG {i}/{len(dataset)}: {test_case['question'][:50]}...")
            
            # Processar query
            start_time = time.time()
            response = self.chatbot.process_query(test_case['question'])
            processing_time = time.time() - start_time
            
            # Avaliar resposta
            evaluation = self._evaluate_rag_response(test_case, response)
            evaluation['processing_time'] = processing_time
            
            self.rag_results.append(evaluation)
        
        # Calcular métricas agregadas
        metrics = self._calculate_rag_metrics()
        
        print(f"✅ Avaliação RAG concluída!")
        return {
            'individual_results': self.rag_results,
            'aggregate_metrics': metrics,
            'summary': self._generate_rag_summary()
        }
    
    def _evaluate_rag_response(self, test_case: Dict, response: Dict) -> Dict:
        """Avalia uma resposta individual do RAG."""
        evaluation = {
            'question': test_case['question'],
            'expected_keywords': test_case['expected_keywords'],
            'expected_source': test_case['expected_source'],
            'difficulty': test_case['difficulty'],
            'response': response['answer'],
            'sources': response.get('sources', []),
            'confidence': response.get('confidence', 0),
            'context_used': response.get('context_used', False)
        }
        
        # Verificar presença de keywords esperadas
        answer_lower = response['answer'].lower()
        keywords_found = sum(1 for kw in test_case['expected_keywords'] 
                           if kw.lower() in answer_lower)
        evaluation['keyword_score'] = keywords_found / len(test_case['expected_keywords'])
        
        # Verificar se a fonte correta foi usada
        source_correct = any(test_case['expected_source'] in source 
                           for source in response.get('sources', []))
        evaluation['source_correct'] = source_correct
        
        # Score geral (combinação de fatores)
        evaluation['overall_score'] = (
            evaluation['keyword_score'] * 0.4 +
            (1.0 if source_correct else 0.0) * 0.3 +
            response.get('confidence', 0) * 0.3
        )
        
        return evaluation
    
    def _calculate_rag_metrics(self) -> Dict:
        """Calcula métricas agregadas do RAG."""
        if not self.rag_results:
            return {}
        
        return {
            'avg_keyword_score': np.mean([r['keyword_score'] for r in self.rag_results]),
            'avg_confidence': np.mean([r['confidence'] for r in self.rag_results]),
            'avg_overall_score': np.mean([r['overall_score'] for r in self.rag_results]),
            'source_accuracy': np.mean([r['source_correct'] for r in self.rag_results]),
            'avg_processing_time': np.mean([r['processing_time'] for r in self.rag_results]),
            'context_usage_rate': np.mean([r['context_used'] for r in self.rag_results])
        }
    
    def _generate_rag_summary(self) -> Dict:
        """Gera resumo da avaliação RAG."""
        if not self.rag_results:
            return {}
        
        # Análise por dificuldade
        by_difficulty = {}
        for result in self.rag_results:
            diff = result['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result['overall_score'])
        
        difficulty_scores = {
            diff: np.mean(scores) 
            for diff, scores in by_difficulty.items()
        }
        
        # Melhores e piores casos
        sorted_results = sorted(self.rag_results, key=lambda x: x['overall_score'])
        
        return {
            'total_tests': len(self.rag_results),
            'difficulty_performance': difficulty_scores,
            'best_case': sorted_results[-1] if sorted_results else None,
            'worst_case': sorted_results[0] if sorted_results else None,
            'high_confidence_rate': sum(1 for r in self.rag_results if r['confidence'] > 0.7) / len(self.rag_results)
        }

# ============================================================================
# 4. AVALIADOR DE CASOS EXTREMOS
# ============================================================================

class EdgeCaseEvaluator:
    """Avalia o comportamento do sistema em casos extremos."""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.edge_case_results = []
    
    def run_edge_case_tests(self) -> Dict:
        """Executa testes de casos extremos."""
        print("🚨 Iniciando testes de casos extremos...")
        
        dataset = TestDatasets.get_edge_cases_dataset()
        
        for i, test_case in enumerate(dataset, 1):
            print(f"Testando caso extremo {i}/{len(dataset)}: {test_case['type']}")
            
            try:
                # Processar query
                start_time = time.time()
                response = self.chatbot.process_query(test_case['question'])
                processing_time = time.time() - start_time
                
                # Avaliar comportamento
                evaluation = {
                    'test_case': test_case,
                    'response': response,
                    'processing_time': processing_time,
                    'handled_gracefully': True,
                    'error': None
                }
                
            except Exception as e:
                evaluation = {
                    'test_case': test_case,
                    'response': None,
                    'processing_time': None,
                    'handled_gracefully': False,
                    'error': str(e)
                }
            
            self.edge_case_results.append(evaluation)
        
        # Análise dos resultados
        analysis = self._analyze_edge_cases()
        
        print(f"✅ Testes de casos extremos concluídos!")
        return {
            'individual_results': self.edge_case_results,
            'analysis': analysis
        }
    
    def _analyze_edge_cases(self) -> Dict:
        """Analisa os resultados dos casos extremos."""
        total_tests = len(self.edge_case_results)
        handled_gracefully = sum(1 for r in self.edge_case_results if r['handled_gracefully'])
        
        analysis = {
            'total_tests': total_tests,
            'graceful_handling_rate': handled_gracefully / total_tests if total_tests > 0 else 0,
            'errors_found': [r for r in self.edge_case_results if not r['handled_gracefully']],
            'recommendations': []
        }
        
        # Gerar recomendações
        if analysis['graceful_handling_rate'] < 1.0:
            analysis['recommendations'].append("Implementar melhor tratamento de erros para casos extremos")
        
        if any(r['test_case']['type'] == 'empty_query' and not r['handled_gracefully'] 
               for r in self.edge_case_results):
            analysis['recommendations'].append("Adicionar validação para queries vazias")
        
        return analysis

# ============================================================================
# 5. SISTEMA COMPLETO DE AVALIAÇÃO
# ============================================================================

class ComprehensiveEvaluator:
    """Sistema completo de avaliação do chatbot."""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.classifier_evaluator = ClassifierEvaluator(chatbot)
        self.rag_evaluator = RAGEvaluator(chatbot)
        self.edge_case_evaluator = EdgeCaseEvaluator(chatbot)
        
        self.evaluation_results = {}
    
    def run_complete_evaluation(self) -> Dict:
        """Executa avaliação completa do sistema."""
        print("🚀 Iniciando avaliação completa do sistema...")
        print("="*60)
        
        start_time = time.time()
        
        # 1. Avaliar classificador
        print("\n1️⃣ AVALIAÇÃO DO CLASSIFICADOR")
        print("-" * 40)
        classifier_results = self.classifier_evaluator.run_classification_tests()
        classifier_errors = self.classifier_evaluator.analyze_classification_errors()
        
        # 2. Avaliar RAG
        print("\n2️⃣ AVALIAÇÃO DO SISTEMA RAG")
        print("-" * 40)
        rag_results = self.rag_evaluator.run_rag_evaluation()
        
        # 3. Avaliar casos extremos
        print("\n3️⃣ AVALIAÇÃO DE CASOS EXTREMOS")
        print("-" * 40)
        edge_case_results = self.edge_case_evaluator.run_edge_case_tests()
        
        # 4. Compilar resultados
        total_time = time.time() - start_time
        
        self.evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluation_time': total_time,
            'classifier': {
                'results': classifier_results,
                'error_analysis': classifier_errors
            },
            'rag': rag_results,
            'edge_cases': edge_case_results,
            'overall_summary': self._generate_overall_summary()
        }
        
        print(f"\n✅ Avaliação completa concluída em {total_time:.2f}s!")
        return self.evaluation_results
    
    def _generate_overall_summary(self) -> Dict:
        """Gera resumo geral da avaliação."""
        summary = {
            'system_health': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'overall_score': 0.0
        }
        
        # Analisar resultados do classificador
        if 'classifier' in self.evaluation_results:
            classifier_acc = self.evaluation_results['classifier']['results']['accuracy']
            if classifier_acc > 0.9:
                summary['strengths'].append("Excelente precisão do classificador")
            elif classifier_acc > 0.8:
                summary['strengths'].append("Boa precisão do classificador")
            else:
                summary['weaknesses'].append("Classificador precisa de melhorias")
        
        # Analisar resultados do RAG
        if 'rag' in self.evaluation_results:
            rag_score = self.evaluation_results['rag']['aggregate_metrics'].get('avg_overall_score', 0)
            if rag_score > 0.8:
                summary['strengths'].append("Sistema RAG com alta qualidade")
            elif rag_score > 0.6:
                summary['strengths'].append("Sistema RAG funcional")
            else:
                summary['weaknesses'].append("Sistema RAG precisa de otimização")
        
        # Analisar casos extremos
        if 'edge_cases' in self.evaluation_results:
            graceful_rate = self.evaluation_results['edge_cases']['analysis']['graceful_handling_rate']
            if graceful_rate == 1.0:
                summary['strengths'].append("Excelente tratamento de casos extremos")
            elif graceful_rate > 0.8:
                summary['strengths'].append("Bom tratamento de casos extremos")
            else:
                summary['weaknesses'].append("Melhorar robustez para casos extremos")
        
        # Calcular score geral
        scores = []
        if 'classifier' in self.evaluation_results:
            scores.append(self.evaluation_results['classifier']['results']['accuracy'])
        if 'rag' in self.evaluation_results:
            scores.append(self.evaluation_results['rag']['aggregate_metrics'].get('avg_overall_score', 0))
        if 'edge_cases' in self.evaluation_results:
            scores.append(self.evaluation_results['edge_cases']['analysis']['graceful_handling_rate'])
        
        summary['overall_score'] = np.mean(scores) if scores else 0.0
        
        # Determinar saúde do sistema
        if summary['overall_score'] > 0.9:
            summary['system_health'] = 'Excelente'
        elif summary['overall_score'] > 0.8:
            summary['system_health'] = 'Bom'
        elif summary['overall_score'] > 0.6:
            summary['system_health'] = 'Regular'
        else:
            summary['system_health'] = 'Precisa melhorias'
        
        return summary
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """Gera relatório detalhado da avaliação."""
        if not self.evaluation_results:
            return "Nenhuma avaliação foi executada ainda."
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE AVALIAÇÃO - CHATBOT JURÍDICO INTELIGENTE")
        report.append("=" * 80)
        report.append(f"Data/Hora: {self.evaluation_results['timestamp']}")
        report.append(f"Tempo total de avaliação: {self.evaluation_results['total_evaluation_time']:.2f}s")
        report.append("")
        
        # Resumo geral
        summary = self.evaluation_results['overall_summary']
        report.append("📊 RESUMO GERAL")
        report.append("-" * 40)
        report.append(f"Saúde do Sistema: {summary['system_health']}")
        report.append(f"Score Geral: {summary['overall_score']:.3f}")
        report.append("")
        
        # Pontos fortes
        if summary['strengths']:
            report.append("✅ PONTOS FORTES:")
            for strength in summary['strengths']:
                report.append(f"  • {strength}")
            report.append("")
        
        # Pontos fracos
        if summary['weaknesses']:
            report.append("❌ PONTOS FRACOS:")
            for weakness in summary['weaknesses']:
                report.append(f"  • {weakness}")
            report.append("")
        
        # Resultados do classificador
        classifier_results = self.evaluation_results['classifier']['results']
        report.append("🎯 CLASSIFICADOR DE INTENÇÕES")
        report.append("-" * 40)
        report.append(f"Acurácia: {classifier_results['accuracy']:.3f}")
        
        # Métricas por categoria
        class_report = classifier_results['classification_report']
        for category in ['rag_required', 'calculation_required', 'general_conversation', 'out_of_scope']:
            if category in class_report:
                metrics = class_report[category]
                report.append(f"{category}:")
                report.append(f"  Precisão: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  F1-Score: {metrics['f1-score']:.3f}")
        report.append("")
        
        # Resultados do RAG
        rag_metrics = self.evaluation_results['rag']['aggregate_metrics']
        report.append("🔍 SISTEMA RAG")
        report.append("-" * 40)
        report.append(f"Score médio de keywords: {rag_metrics['avg_keyword_score']:.3f}")
        report.append(f"Confiança média: {rag_metrics['avg_confidence']:.3f}")
        report.append(f"Score geral médio: {rag_metrics['avg_overall_score']:.3f}")
        report.append(f"Precisão das fontes: {rag_metrics['source_accuracy']:.3f}")
        report.append(f"Tempo médio de processamento: {rag_metrics['avg_processing_time']:.3f}s")
        report.append("")
        
        # Casos extremos
        edge_analysis = self.evaluation_results['edge_cases']['analysis']
        report.append("🚨 CASOS EXTREMOS")
        report.append("-" * 40)
        report.append(f"Taxa de tratamento adequado: {edge_analysis['graceful_handling_rate']:.3f}")
        report.append(f"Erros encontrados: {len(edge_analysis['errors_found'])}")
        report.append("")
        
        # Recomendações
        if summary.get('recommendations'):
            report.append("💡 RECOMENDAÇÕES")
            report.append("-" * 40)
            for rec in summary['recommendations']:
                report.append(f"  • {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_to_file:
            filename = f"relatorio_avaliacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Relatório salvo em: {filename}")
        
        return report_text
    
    def save_results_json(self, filename: str = None):
        """Salva resultados em formato JSON."""
        if filename is None:
            filename = f"avaliacao_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Resultados salvos em: {filename}")

# ============================================================================
# 6. EXEMPLO DE USO
# ============================================================================

def run_evaluation_example():
    """Exemplo de como usar o sistema de avaliação."""
    try:
        # Importar e inicializar o chatbot
        from chatbot_juridico_inteligente import ChatbotJuridicoInteligente
        
        print("🚀 Inicializando chatbot para avaliação...")
        chatbot = ChatbotJuridicoInteligente()
        
        # Criar avaliador
        evaluator = ComprehensiveEvaluator(chatbot)
        
        # Executar avaliação completa
        results = evaluator.run_complete_evaluation()
        
        # Gerar e mostrar relatório
        report = evaluator.generate_report()
        print("\n" + report)
        
        # Salvar resultados
        evaluator.save_results_json()
        
        return results
        
    except Exception as e:
        print(f"❌ Erro na avaliação: {e}")
        return None

if __name__ == "__main__":
    run_evaluation_example()