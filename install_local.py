#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instalador de Dependências - Chatbot Jurídico Local
Instala apenas as dependências essenciais para execução local.
"""

import subprocess
import sys
import os

def install_package(package):
    """Instala um pacote via pip."""
    try:
        print(f"📦 Instalando {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--quiet"
        ])
        print(f"✅ {package} instalado")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erro ao instalar {package}")
        return False

def check_package(package):
    """Verifica se um pacote está instalado."""
    try:
        if package == 'faiss':
            import faiss
        elif package == 'sklearn':
            import sklearn
        else:
            __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Instala dependências essenciais."""
    print("🚀 INSTALADOR - CHATBOT JURÍDICO LOCAL")
    print("="*50)
    
    # Dependências essenciais (ordem de instalação)
    essential_packages = [
        "torch",
        "transformers", 
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "faiss-cpu",
        "gradio"
    ]
    
    # Dependências opcionais (para funcionalidades extras)
    optional_packages = [
        "langchain",
        "langchain-text-splitters",
        "matplotlib",
        "seaborn"
    ]
    
    print("🔍 Verificando dependências essenciais...")
    
    missing_essential = []
    for package in essential_packages:
        if check_package(package.split('==')[0]):
            print(f"✅ {package} - OK")
        else:
            print(f"❌ {package} - Faltando")
            missing_essential.append(package)
    
    if missing_essential:
        print(f"\n📦 Instalando {len(missing_essential)} pacote(s) essencial(is)...")
        
        failed = []
        for package in missing_essential:
            if not install_package(package):
                failed.append(package)
        
        if failed:
            print(f"\n❌ Falha ao instalar: {failed}")
            print("💡 Tente instalar manualmente:")
            for pkg in failed:
                print(f"   pip install {pkg}")
            return False
    
    print("\n✅ Dependências essenciais OK!")
    
    # Perguntar sobre opcionais
    install_optional = input("\n❓ Instalar dependências opcionais? (s/N): ").lower().strip()
    
    if install_optional in ['s', 'sim', 'y', 'yes']:
        print("\n📦 Instalando dependências opcionais...")
        
        for package in optional_packages:
            if not check_package(package.split('==')[0]):
                install_package(package)
    
    print("\n🎉 Instalação concluída!")
    print("\n🚀 Para executar o chatbot:")
    print("   python run_local.py")
    
    return True

if __name__ == "__main__":
    main()