import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import json
from src import config


def build_linkage_graph(X: pd.DataFrame, risk_df: pd.DataFrame,
                        selected_features: list, top_n_links: int = 1):
    """
    Build an attribute-based linkage graph.
    Two accounts are linked if they share similar values on
    multiple high-risk features — suggesting coordinated mule behavior.
    """
    # Work only with flagged accounts + a sample of normals for context
    flagged_idx = risk_df[risk_df["risk_tier"].isin(["Red", "Amber"])]["account_idx"].tolist()
    green_sample = risk_df[risk_df["risk_tier"] == "Green"].sample(
        min(50, len(risk_df[risk_df["risk_tier"] == "Green"])),
        random_state=config.RANDOM_STATE
    )["account_idx"].tolist()

    display_idx = flagged_idx + green_sample
    X_display = X.iloc[display_idx][selected_features].copy()

    # Use only numeric, low-missing features for similarity
    num_cols = X_display.select_dtypes(include=[np.number]).columns
    X_num = X_display[num_cols].fillna(X_display[num_cols].median())

    # Normalize
    X_norm = (X_num - X_num.min()) / (X_num.max() - X_num.min() + 1e-9)

    # Build graph
    G = nx.Graph()

    # Add nodes
    for i, idx in enumerate(display_idx):
        row = risk_df[risk_df["account_idx"] == idx].iloc[0]
        G.add_node(
            idx,
            tier=row["risk_tier"],
            risk_score=round(float(row["risk_score"]), 4),
            true_label=int(row["true_label"]),
            xgb_proba=round(float(row["xgb_proba"]), 4),
        )

    # Add edges based on feature similarity
    X_arr = X_norm.values
    indices = list(range(len(display_idx)))

    for i in indices:
        similarities = []
        for j in indices:
            if i == j:
                continue
            dot = np.dot(X_arr[i], X_arr[j])
            norm = np.linalg.norm(X_arr[i]) * np.linalg.norm(X_arr[j])
            sim = dot / (norm + 1e-9)
            similarities.append((j, sim))

        # Link to top_n_links most similar accounts only
        top_links = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n_links]
        for j, sim in top_links:
            if sim > 0.85:
                G.add_edge(display_idx[i], display_idx[j], weight=round(sim, 3))

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, display_idx


def render_pyvis_graph(G: nx.Graph, save_path=None):
    """
    Clean hierarchical layout — no physics, fixed positions.
    Confirmed mules inner ring, risk tiers in concentric rings outward.
    """
    save_path = save_path or (config.OUTPUTS_DIR / "linkage_graph.html")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Separate nodes by tier
    mules = [n for n, d in G.nodes(data=True) if d.get("true_label") == 1]
    red = [n for n, d in G.nodes(data=True)
           if d.get("tier") == "Red" and d.get("true_label") == 0]
    amber = [n for n, d in G.nodes(data=True) if d.get("tier") == "Amber"]
    green = [n for n, d in G.nodes(data=True) if d.get("tier") == "Green"]

    pos = {}

    # Mules — inner ring (radius 350)
    for i, node in enumerate(mules):
        angle = 2 * np.pi * i / max(len(mules), 1)
        pos[node] = (350 * np.cos(angle), 350 * np.sin(angle))

    # Red — middle ring (radius 550)
    for i, node in enumerate(red):
        angle = 2 * np.pi * i / max(len(red), 1)
        pos[node] = (550 * np.cos(angle), 550 * np.sin(angle))

    # Amber — outer ring (radius 750)
    for i, node in enumerate(amber):
        angle = 2 * np.pi * i / max(len(amber), 1)
        pos[node] = (750 * np.cos(angle), 750 * np.sin(angle))

    # Green — outermost ring (radius 950)
    for i, node in enumerate(green):
        angle = 2 * np.pi * i / max(len(green), 1)
        pos[node] = (950 * np.cos(angle), 950 * np.sin(angle))

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0f1117",
        font_color="white",
        directed=False,
    )

    # Disable physics — use fixed positions
    net.toggle_physics(False)

    # Add nodes at fixed positions
    for node, data in G.nodes(data=True):
        tier = data.get("tier", "Green")
        is_mule = data.get("true_label", 0) == 1
        risk_score = data.get("risk_score", 0)
        x, y = pos.get(node, (0, 0))

        if is_mule:
            color = "#ff4444"
            shape = "star"
            size = 22
        elif tier == "Red":
            color = "#ff8888"
            shape = "dot"
            size = 16
        elif tier == "Amber":
            color = "#ffa500"
            shape = "diamond"
            size = 12
        else:
            color = "#51cf66"
            shape = "dot"
            size = 7

        net.add_node(
            node,
            label=f"#{node}",
            color=color,
            size=size,
            shape=shape,
            x=float(x * 1.2),
            y=float(y * 1.2),
            physics=False,
            title=(
                f"Account: #{node}\n"
                f"Risk Tier: {tier}\n"
                f"Risk Score: {risk_score:.4f}\n"
                f"XGB Prob: {data.get('xgb_proba', 0):.4f}\n"
                f"Label: {'⚠️ MULE' if is_mule else '✓ Normal'}"
            )
        )

    # Add edges — only between mules and red/amber accounts
    for u, v, data in G.edges(data=True):
        u_tier = G.nodes[u].get("tier", "Green")
        v_tier = G.nodes[v].get("tier", "Green")
        u_mule = G.nodes[u].get("true_label", 0) == 1
        v_mule = G.nodes[v].get("true_label", 0) == 1

        if (u_mule or v_mule or
                (u_tier in ["Red", "Amber"] and v_tier in ["Red", "Amber"])):
            weight = data.get("weight", 0.5)
            net.add_edge(
                u, v,
                width=max(0.5, weight * 2),
                color="#ffffff22",
                title=f"Similarity: {weight:.3f}"
            )

    net.set_options("""
    {
        "physics": {"enabled": false},
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
        },
        "edges": {
            "smooth": {"enabled": false}
        }
    }
    """)

    legend_html = """
    <div style='position:absolute; top:10px; right:10px;
                background:rgba(15,17,23,0.95); padding:14px 18px;
                border-radius:10px; border:1px solid #444;
                font-family:Arial; font-size:13px; color:white; z-index:999;'>
        <b style='font-size:15px'>🔍 Account Risk Network</b><br><br>
        <span style='color:#ff4444; font-size:20px'>★</span>
        <span style='color:#ff4444'> Confirmed Mule</span><br><br>
        <span style='color:#ff8888; font-size:16px'>●</span>
        <span style='color:#ff8888'> Red — High Risk</span><br><br>
        <span style='color:#ffa500; font-size:16px'>◆</span>
        <span style='color:#ffa500'> Amber — Medium Risk</span><br><br>
        <span style='color:#51cf66; font-size:16px'>●</span>
        <span style='color:#51cf66'> Green — Low Risk</span><br><br>
        <hr style='border-color:#444; margin:8px 0'>
        <i style='color:#888; font-size:11px'>
            Rings: Mules → Red → Amber → Green<br>
            Edges = behavioral similarity<br>
            Scroll to zoom · Drag to pan<br>
            Hover node for full details
        </i>
    </div>
    <div style='position:absolute; bottom:10px; left:50%; transform:translateX(-50%);
                background:rgba(15,17,23,0.95); padding:6px 16px;
                border-radius:20px; border:1px solid #444;
                font-family:Arial; font-size:11px; color:#888; z-index:999;'>
        ⭕ Inner → Confirmed Mules &nbsp;·&nbsp;
        🔴 Middle → High Risk &nbsp;·&nbsp;
        🟠 Outer → Medium Risk &nbsp;·&nbsp;
        🟢 Outermost → Low Risk
    </div>
    """

    net.save_graph(str(save_path))

    with open(save_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", f"{legend_html}</body>")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Linkage graph saved to {save_path}")
    return save_path


def get_connected_components(G: nx.Graph, risk_df: pd.DataFrame):
    """
    Find clusters of connected suspicious accounts.
    Returns summary of each cluster.
    """
    clusters = []
    for component in nx.connected_components(G):
        if len(component) < 2:
            continue

        component_risk = risk_df[risk_df["account_idx"].isin(component)]
        mule_count = component_risk[component_risk["true_label"] == 1].shape[0]
        red_count = component_risk[component_risk["risk_tier"] == "Red"].shape[0]
        avg_risk = component_risk["risk_score"].mean()

        clusters.append({
            "size": len(component),
            "mule_count": mule_count,
            "red_count": red_count,
            "avg_risk_score": round(avg_risk, 4),
            "account_ids": list(component)
        })

    clusters = sorted(clusters, key=lambda x: x["avg_risk_score"], reverse=True)
    return clusters