"""
Lineage Visualizer
Provides visualization capabilities for data lineage
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

logger = logging.getLogger(__name__)

class LineageVisualizer:
    """
    Provides visualization capabilities for data lineage
    """
    
    def __init__(self):
        self.colors = {
            'dataset': '#3498db',
            'kafka_topic': '#e74c3c',
            's3_dataset': '#2ecc71',
            'ml_model': '#f39c12',
            'process': '#9b59b6',
            'database': '#1abc9c',
            'api': '#e67e22'
        }
        
        logger.info("LineageVisualizer initialized")
    
    def create_lineage_graph(self, lineage_data: Dict[str, Any]) -> nx.DiGraph:
        """Create a NetworkX graph from lineage data"""
        try:
            G = nx.DiGraph()
            
            # Add nodes
            for entity in lineage_data.get('entities', []):
                G.add_node(
                    entity['entity_id'],
                    type=entity['entity_type'],
                    name=entity['name'],
                    location=entity['location'],
                    tags=entity.get('tags', [])
                )
            
            # Add edges
            for event in lineage_data.get('events', []):
                if event['source_entity'] and event['target_entity']:
                    G.add_edge(
                        event['source_entity'],
                        event['target_entity'],
                        process=event['process_name'],
                        event_type=event['event_type'],
                        timestamp=event['timestamp']
                    )
            
            logger.info(f"Created lineage graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to create lineage graph: {str(e)}")
            return nx.DiGraph()
    
    def visualize_lineage_matplotlib(self, 
                                   lineage_data: Dict[str, Any],
                                   output_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 10)) -> str:
        """Create matplotlib visualization of lineage"""
        try:
            G = self.create_lineage_graph(lineage_data)
            
            if G.number_of_nodes() == 0:
                logger.warning("No nodes in lineage graph")
                return ""
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes by type
            node_types = set(nx.get_node_attributes(G, 'type').values())
            for node_type in node_types:
                nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes,
                    node_color=self.colors.get(node_type, '#95a5a6'),
                    node_size=1000,
                    alpha=0.8,
                    label=node_type.replace('_', ' ').title()
                )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color='#7f8c8d',
                arrows=True,
                arrowsize=20,
                alpha=0.6,
                width=2
            )
            
            # Draw labels
            labels = {n: d.get('name', n) for n, d in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            # Add legend and title
            plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1, 1))
            plt.title('Data Lineage Visualization', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Lineage visualization saved to {output_path}")
                return output_path
            else:
                plt.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Failed to create matplotlib visualization: {str(e)}")
            return ""
    
    def visualize_lineage_plotly(self, 
                               lineage_data: Dict[str, Any],
                               output_path: Optional[str] = None,
                               width: int = 1200,
                               height: int = 800) -> str:
        """Create interactive Plotly visualization of lineage"""
        try:
            G = self.create_lineage_graph(lineage_data)
            
            if G.number_of_nodes() == 0:
                logger.warning("No nodes in lineage graph")
                return ""
            
            # Get positions using spring layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Prepare node data
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_hover = []
            
            for node, data in G.nodes(data=True):
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_text.append(data.get('name', node))
                node_type = data.get('type', 'unknown')
                node_colors.append(self.colors.get(node_type, '#95a5a6'))
                node_sizes.append(20)
                
                hover_text = f"""
                <b>{data.get('name', node)}</b><br>
                Type: {node_type}<br>
                Location: {data.get('location', 'N/A')}<br>
                Tags: {', '.join(data.get('tags', []))}
                """
                node_hover.append(hover_text)
            
            # Prepare edge data
            edge_x = []
            edge_y = []
            edge_hover = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                hover_text = f"""
                Process: {edge[2].get('process', 'Unknown')}<br>
                Type: {edge[2].get('event_type', 'Unknown')}<br>
                Timestamp: {edge[2].get('timestamp', 'Unknown')}
                """
                edge_hover.append(hover_text)
            
            # Create the plot
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#7f8c8d'),
                hoverinfo='none',
                mode='lines',
                name='Data Flow'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertext=node_hover,
                hoverinfo='text',
                name='Data Entities'
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Interactive Data Lineage Visualization',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Hover over nodes for details, drag to pan, zoom to explore",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height
            )
            
            # Save or show
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Interactive lineage visualization saved to {output_path}")
                return output_path
            else:
                fig.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Failed to create Plotly visualization: {str(e)}")
            return ""
    
    def create_lineage_dashboard(self, 
                               lineage_data: Dict[str, Any],
                               output_path: Optional[str] = None) -> str:
        """Create a comprehensive lineage dashboard"""
        try:
            G = self.create_lineage_graph(lineage_data)
            
            if G.number_of_nodes() == 0:
                logger.warning("No nodes in lineage graph")
                return ""
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lineage Overview', 'Entity Types Distribution', 
                              'Data Flow Timeline', 'Entity Dependencies'),
                specs=[[{"type": "scatter"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # 1. Lineage Overview (top-left)
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='lightgray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ), row=1, col=1)
            
            # Add nodes by type
            node_types = set(nx.get_node_attributes(G, 'type').values())
            for node_type in node_types:
                nodes = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
                node_x = [pos[n][0] for n in nodes]
                node_y = [pos[n][1] for n in nodes]
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.colors.get(node_type, '#95a5a6'),
                        line=dict(width=2, color='white')
                    ),
                    name=node_type.replace('_', ' ').title(),
                    showlegend=True
                ), row=1, col=1)
            
            # 2. Entity Types Distribution (top-right)
            entity_counts = {}
            for node, data in G.nodes(data=True):
                entity_type = data.get('type', 'unknown')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            fig.add_trace(go.Pie(
                labels=list(entity_counts.keys()),
                values=list(entity_counts.values()),
                marker_colors=[self.colors.get(et, '#95a5a6') for et in entity_counts.keys()]
            ), row=1, col=2)
            
            # 3. Data Flow Timeline (bottom-left)
            timeline_data = {}
            for event in lineage_data.get('events', []):
                date = event['timestamp'][:10]  # Extract date
                timeline_data[date] = timeline_data.get(date, 0) + 1
            
            fig.add_trace(go.Bar(
                x=list(timeline_data.keys()),
                y=list(timeline_data.values()),
                name='Events per Day'
            ), row=2, col=1)
            
            # 4. Entity Dependencies Heatmap (bottom-right)
            # Create adjacency matrix
            nodes = list(G.nodes())
            adj_matrix = nx.adjacency_matrix(G).todense()
            
            fig.add_trace(go.Heatmap(
                z=adj_matrix.tolist(),
                x=nodes,
                y=nodes,
                colorscale='Blues',
                showscale=True
            ), row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Data Lineage Dashboard',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24}
                },
                height=800,
                showlegend=True
            )
            
            # Save or show
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Lineage dashboard saved to {output_path}")
                return output_path
            else:
                fig.show()
                return "displayed"
                
        except Exception as e:
            logger.error(f"Failed to create lineage dashboard: {str(e)}")
            return ""
    
    def generate_lineage_report(self, 
                              lineage_data: Dict[str, Any],
                              output_path: Optional[str] = None) -> str:
        """Generate a comprehensive lineage report"""
        try:
            G = self.create_lineage_graph(lineage_data)
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_entities": G.number_of_nodes(),
                    "total_relationships": G.number_of_edges()
                },
                "entity_summary": {},
                "relationship_summary": {},
                "quality_metrics": {},
                "recommendations": []
            }
            
            # Entity summary
            entity_types = {}
            for node, data in G.nodes(data=True):
                entity_type = data.get('type', 'unknown')
                if entity_type not in entity_types:
                    entity_types[entity_type] = {
                        "count": 0,
                        "entities": []
                    }
                entity_types[entity_type]["count"] += 1
                entity_types[entity_type]["entities"].append({
                    "id": node,
                    "name": data.get('name', node),
                    "location": data.get('location', 'N/A')
                })
            
            report["entity_summary"] = entity_types
            
            # Relationship summary
            relationship_types = {}
            for source, target, data in G.edges(data=True):
                rel_type = data.get('event_type', 'unknown')
                if rel_type not in relationship_types:
                    relationship_types[rel_type] = 0
                relationship_types[rel_type] += 1
            
            report["relationship_summary"] = relationship_types
            
            # Quality metrics
            report["quality_metrics"] = {
                "orphaned_entities": len([n for n in G.nodes() if G.degree(n) == 0]),
                "highly_connected_entities": len([n for n in G.nodes() if G.degree(n) > 5]),
                "circular_dependencies": len(list(nx.simple_cycles(G))),
                "max_depth": self._calculate_max_depth(G)
            }
            
            # Recommendations
            recommendations = []
            
            if report["quality_metrics"]["orphaned_entities"] > 0:
                recommendations.append({
                    "type": "warning",
                    "message": f"Found {report['quality_metrics']['orphaned_entities']} orphaned entities",
                    "action": "Review orphaned entities and establish connections"
                })
            
            if report["quality_metrics"]["circular_dependencies"] > 0:
                recommendations.append({
                    "type": "error",
                    "message": f"Found {report['quality_metrics']['circular_dependencies']} circular dependencies",
                    "action": "Break circular dependencies to prevent infinite loops"
                })
            
            if report["quality_metrics"]["max_depth"] > 10:
                recommendations.append({
                    "type": "warning",
                    "message": f"Maximum lineage depth is {report['quality_metrics']['max_depth']}",
                    "action": "Consider simplifying complex lineage chains"
                })
            
            report["recommendations"] = recommendations
            
            # Save report
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Lineage report saved to {output_path}")
                return output_path
            else:
                return json.dumps(report, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to generate lineage report: {str(e)}")
            return f"Report generation failed: {str(e)}"
    
    def _calculate_max_depth(self, G: nx.DiGraph) -> int:
        """Calculate maximum depth of the lineage graph"""
        try:
            # Find root nodes (nodes with no incoming edges)
            root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            if not root_nodes:
                return 0
            
            max_depth = 0
            for root in root_nodes:
                depth = self._calculate_node_depth(G, root)
                max_depth = max(max_depth, depth)
            
            return max_depth
            
        except Exception as e:
            logger.error(f"Failed to calculate max depth: {str(e)}")
            return 0
    
    def _calculate_node_depth(self, G: nx.DiGraph, node: str, visited: set = None) -> int:
        """Calculate depth of a specific node"""
        if visited is None:
            visited = set()
        
        if node in visited:
            return 0
        
        visited.add(node)
        
        # Get all outgoing neighbors
        neighbors = list(G.successors(node))
        
        if not neighbors:
            return 1
        
        max_child_depth = 0
        for neighbor in neighbors:
            child_depth = self._calculate_node_depth(G, neighbor, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def export_lineage_schema(self, 
                            lineage_data: Dict[str, Any],
                            output_path: Optional[str] = None) -> str:
        """Export lineage schema for external tools"""
        try:
            G = self.create_lineage_graph(lineage_data)
            
            schema = {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "entities": [],
                "relationships": []
            }
            
            # Export entities
            for node, data in G.nodes(data=True):
                entity_schema = {
                    "id": node,
                    "type": data.get('type', 'unknown'),
                    "name": data.get('name', node),
                    "location": data.get('location', 'N/A'),
                    "properties": {
                        "tags": data.get('tags', []),
                        "created_at": data.get('created_at'),
                        "updated_at": data.get('updated_at')
                    }
                }
                schema["entities"].append(entity_schema)
            
            # Export relationships
            for source, target, data in G.edges(data=True):
                relationship_schema = {
                    "source": source,
                    "target": target,
                    "type": data.get('event_type', 'unknown'),
                    "process": data.get('process', 'unknown'),
                    "properties": {
                        "timestamp": data.get('timestamp'),
                        "metadata": data.get('metadata', {})
                    }
                }
                schema["relationships"].append(relationship_schema)
            
            # Save schema
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(schema, f, indent=2)
                logger.info(f"Lineage schema exported to {output_path}")
                return output_path
            else:
                return json.dumps(schema, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to export lineage schema: {str(e)}")
            return f"Schema export failed: {str(e)}"
