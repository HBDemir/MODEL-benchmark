"""
Gerekli paketler:
pip install numpy matplotlib seaborn scikit-learn pandas plotly kaleido

Eğer bu paketler yüklü değilse:
pip install numpy==1.24.3 matplotlib==3.7.1 seaborn==0.12.2 scikit-learn==1.3.0 pandas==2.0.3 plotly==5.15.0 kaleido==0.2.1
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ek import'lar
import json
import base64
import io
from typing import List, Tuple, Optional
import math

class DirectEmbeddingVisualizer:
    def __init__(self):
        """Mevcut embeddings ve scorlar ile çalışan görselleştirici"""
        pass
    
    def visualize_single_comparison(self, query_embedding, chunk_embedding, chunk_text, similarity_score, chunk_idx, save_path=None):
        """Tek bir query-chunk karşılaştırmasını görselleştir"""
        
        # Plotly ile interaktif görselleştirme
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Query Embedding Vector (Dim: {len(query_embedding)})",
                f"Chunk {chunk_idx} Embedding Vector (Dim: {len(chunk_embedding)})",
                "Embedding Vectors Overlay Comparison",
                "Cosine Similarity Score"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Query embedding
        fig.add_trace(
            go.Scatter(
                x=list(range(len(query_embedding))),
                y=query_embedding,
                mode='lines+markers',
                name='Query Embedding',
                line=dict(color='blue', width=2),
                marker=dict(size=3)
            ),
            row=1, col=1
        )
        
        # Chunk embedding
        fig.add_trace(
            go.Scatter(
                x=list(range(len(chunk_embedding))),
                y=chunk_embedding,
                mode='lines+markers',
                name=f'Chunk {chunk_idx} Embedding',
                line=dict(color='red', width=2),
                marker=dict(size=3)
            ),
            row=1, col=2
        )
        
        # Overlay comparison
        fig.add_trace(
            go.Scatter(
                x=list(range(len(query_embedding))),
                y=query_embedding,
                mode='lines',
                name='Query',
                line=dict(color='blue', width=2),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(chunk_embedding))),
                y=chunk_embedding,
                mode='lines',
                name=f'Chunk {chunk_idx}',
                line=dict(color='red', width=2),
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Similarity gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=similarity_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Cosine Similarity: {similarity_score:.4f}"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightcoral"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "lightgreen"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Query vs Chunk {chunk_idx} - Similarity: {similarity_score:.4f}",
            height=700,
            width=1200,
            showlegend=True
        )
        
        # Chunk text bilgisi
        chunk_preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
        fig.add_annotation(
            text=f"<b>Chunk {chunk_idx} Text:</b><br>{chunk_preview}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10),
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=1,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_matplotlib_comparison(self, query_embedding, chunk_embedding, chunk_text, similarity_score, chunk_idx, save_path=None):
        """Matplotlib ile static görselleştirme"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Query vs Chunk {chunk_idx} - Cosine Similarity: {similarity_score:.4f}', fontsize=16, fontweight='bold')
        
        # Query embedding
        axes[0, 0].plot(query_embedding, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].fill_between(range(len(query_embedding)), query_embedding, alpha=0.3, color='blue')
        axes[0, 0].set_title('Query Embedding Vector', fontsize=14)
        axes[0, 0].set_xlabel('Dimension Index')
        axes[0, 0].set_ylabel('Embedding Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(min(min(query_embedding), min(chunk_embedding)) - 0.1, 
                           max(max(query_embedding), max(chunk_embedding)) + 0.1)
        
        # Chunk embedding
        axes[0, 1].plot(chunk_embedding, 'r-', linewidth=2, alpha=0.8)
        axes[0, 1].fill_between(range(len(chunk_embedding)), chunk_embedding, alpha=0.3, color='red')
        axes[0, 1].set_title(f'Chunk {chunk_idx} Embedding Vector', fontsize=14)
        axes[0, 1].set_xlabel('Dimension Index')
        axes[0, 1].set_ylabel('Embedding Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(min(min(query_embedding), min(chunk_embedding)) - 0.1, 
                           max(max(query_embedding), max(chunk_embedding)) + 0.1)
        
        # Overlay comparison
        axes[1, 0].plot(query_embedding, 'b-', linewidth=2, alpha=0.8, label='Query')
        axes[1, 0].plot(chunk_embedding, 'r-', linewidth=2, alpha=0.8, label=f'Chunk {chunk_idx}')
        axes[1, 0].set_title('Embedding Vectors Overlay', fontsize=14)
        axes[1, 0].set_xlabel('Dimension Index')
        axes[1, 0].set_ylabel('Embedding Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Similarity visualization
        colors = ['red' if similarity_score < 0.3 else 'orange' if similarity_score < 0.6 else 'yellow' if similarity_score < 0.8 else 'green']
        bars = axes[1, 1].bar(['Cosine Similarity'], [similarity_score], color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Similarity Score', fontsize=14)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Similarity değerini bar üzerine yazdır
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Threshold çizgileri
        axes[1, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Similarity')
        axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Medium Similarity')
        axes[1, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Low Similarity')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Chunk text bilgisi
        chunk_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        fig.text(0.5, 0.02, f'Chunk Text: {chunk_preview}', ha='center', va='bottom', 
                fontsize=10, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_all_comparisons(self, query_embedding, chunk_embeddings, chunk_texts, similarity_scores, query_text="", output_dir="embedding_visualizations"):
        """Tüm chunk'lar için görselleştirme oluştur"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML raporu başlat
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MOF Embedding Similarity Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #007bff; }}
                .chunk-container {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .similarity-high {{ border-left: 4px solid #28a745; }}
                .similarity-medium {{ border-left: 4px solid #ffc107; }}
                .similarity-low {{ border-left: 4px solid #dc3545; }}
                .score {{ font-size: 1.3em; font-weight: bold; }}
                .score-high {{ color: #28a745; }}
                .score-medium {{ color: #ffc107; }}
                .score-low {{ color: #dc3545; }}
                h1 {{ color: #333; margin: 0; }}
                h2 {{ color: #555; }}
                .stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
                .stat-item {{ text-align: center; padding: 10px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>🔬 MOF Embedding Similarity Analysis</h1>
                <p>Deep dive into query-chunk embedding similarities</p>
            </div>
            
            <div class="summary">
                <h2>📊 Analysis Summary</h2>
                <p><strong>Query:</strong> {query_text}</p>
                <div class="stats">
                    <div class="stat-item">
                        <strong>Total Chunks:</strong><br>{len(chunk_embeddings)}
                    </div>
                    <div class="stat-item">
                        <strong>Avg Similarity:</strong><br>{np.mean(similarity_scores):.4f}
                    </div>
                    <div class="stat-item">
                        <strong>Max Similarity:</strong><br>{np.max(similarity_scores):.4f}
                    </div>
                    <div class="stat-item">
                        <strong>Min Similarity:</strong><br>{np.min(similarity_scores):.4f}
                    </div>
                </div>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Her chunk için görselleştirme
        for i, (chunk_emb, chunk_text, sim_score) in enumerate(zip(chunk_embeddings, chunk_texts, similarity_scores)):
            
            # Plotly görselleştirme
            plotly_fig = self.visualize_single_comparison(
                query_embedding, chunk_emb, chunk_text, sim_score, i+1
            )
            plotly_html = plotly_fig.to_html(include_plotlyjs=False, div_id=f"plotly_chunk_{i}")
            
            # Matplotlib görselleştirme
            matplotlib_fig = self.visualize_matplotlib_comparison(
                query_embedding, chunk_emb, chunk_text, sim_score, i+1, 
                save_path=f"{output_dir}/chunk_{i+1}_matplotlib.png"
            )
            plt.close(matplotlib_fig)
            
            # Similarity kategorisi
            if sim_score >= 0.8:
                category = "similarity-high"
                score_class = "score-high"
                category_text = "🟢 High Similarity"
            elif sim_score >= 0.6:
                category = "similarity-medium"
                score_class = "score-medium"
                category_text = "🟡 Medium Similarity"
            else:
                category = "similarity-low"
                score_class = "score-low"
                category_text = "🔴 Low Similarity"
            
            html_content += f"""
            <div class="chunk-container {category}">
                <h2>Chunk {i+1} Analysis</h2>
                <p><strong>Category:</strong> {category_text}</p>
                <p class="score {score_class}">Similarity Score: {sim_score:.4f}</p>
                <p><strong>Text Preview:</strong></p>
                <p style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-style: italic;">
                    {chunk_text[:400]}{'...' if len(chunk_text) > 400 else ''}
                </p>
                
                <h3>📈 Interactive Visualization</h3>
                {plotly_html}
                
                <h3>🖼️ Static Visualization</h3>
                <img src="chunk_{i+1}_matplotlib.png" alt="Chunk {i+1} Matplotlib Visualization" style="max-width: 100%; height: auto;">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # HTML dosyasını kaydet
        with open(f"{output_dir}/embedding_analysis_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Özet CSV
        df = pd.DataFrame({
            'chunk_index': range(1, len(chunk_embeddings)+1),
            'similarity_score': similarity_scores,
            'similarity_category': ['High' if s >= 0.8 else 'Medium' if s >= 0.6 else 'Low' for s in similarity_scores],
            'chunk_text_preview': [text[:200] + "..." if len(text) > 200 else text for text in chunk_texts]
        })
        df.to_csv(f"{output_dir}/similarity_analysis.csv", index=False)
        
        print(f"✅ Analiz tamamlandı!")
        print(f"📁 Dosyalar '{output_dir}' klasöründe:")
        print(f"   📄 HTML raporu: embedding_analysis_report.html")
        print(f"   📊 CSV dosyası: similarity_analysis.csv")
        print(f"   🖼️  PNG dosyaları: chunk_X_matplotlib.png")
        
        return html_content

