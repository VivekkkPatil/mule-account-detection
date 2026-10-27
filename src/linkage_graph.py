
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
from src import config


def build_linkage_graph(
    X: pd.DataFrame,
    risk_df: pd.DataFrame,
    selected_features: list,
    top_n_links: int = 1
):
    """
    Build an attribute-based linkage graph.

    Accounts are linked when their selected numeric features have
    high cosine similarity. Account indices are treated as positional
    row indices in X.
    """

    X = X.reset_index(drop=True)
    risk_df = risk_df.copy()

    required_columns = [
        "account_idx",
        "risk_tier",
        "risk_score",
        "true_label",
        "xgb_proba",
    ]

    missing_columns = [
        col for col in required_columns if col not in risk_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing columns in risk_df: {missing_columns}"
        )

    # Convert account indices to valid integers.
    risk_df["account_idx"] = pd.to_numeric(
        risk_df["account_idx"], errors="coerce"
    )
    risk_df = risk_df.dropna(subset=["account_idx"]).copy()
    risk_df["account_idx"] = risk_df["account_idx"].astype(int)

    # Keep only indices that point to existing rows in X.
    risk_df = risk_df[
        (risk_df["account_idx"] >= 0)
        & (risk_df["account_idx"] < len(X))
    ].copy()

    # One risk record per account to avoid duplicate graph nodes.
    risk_df = risk_df.drop_duplicates(
        subset=["account_idx"], keep="first"
    )

    # Select Red and Amber accounts.
    flagged_idx = risk_df.loc[
        risk_df["risk_tier"].isin(["Red", "Amber"]),
        "account_idx",
    ].tolist()

    # Sample Green accounts for context.
    green_df = risk_df[risk_df["risk_tier"] == "Green"]

    green_sample = green_df.sample(
        n=min(50, len(green_df)),
        random_state=config.RANDOM_STATE,
    )["account_idx"].tolist()

    # Combine indices without duplicates.
    display_idx = list(dict.fromkeys(flagged_idx + green_sample))

    if not display_idx:
        print("No eligible accounts found for the linkage graph.")
        return nx.Graph(), []

    # Keep only selected features that exist in X.
    valid_features = [
        feature
        for feature in selected_features
        if feature in X.columns
    ]

    if not valid_features:
        raise ValueError(
            "None of the selected features exist in X. "
            "Check selected_features and the feature dataset."
        )

    # Extract account features using positional indexing.
    X_display = X.iloc[display_idx][valid_features].copy()

    # Keep numeric columns for similarity calculation.
    X_num = X_display.select_dtypes(include=[np.number]).copy()

    if X_num.shape[1] == 0:
        raise ValueError(
            "No numeric selected features are available for the graph."
        )

    # Replace infinite values, then impute missing values.
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.fillna(X_num.median()).fillna(0)

    # Min-max normalization.
    feature_min = X_num.min()
    feature_range = X_num.max() - feature_min
    X_norm = (X_num - feature_min) / (feature_range + 1e-9)

    X_arr = X_norm.to_numpy(dtype=float)

    # Create graph and add account nodes.
    G = nx.Graph()

    risk_lookup = risk_df.set_index("account_idx")

    for idx in display_idx:
        if idx not in risk_lookup.index:
            continue

        row = risk_lookup.loc[idx]

        G.add_node(
            idx,
            tier=str(row["risk_tier"]),
            risk_score=float(row["risk_score"]),
            true_label=int(row["true_label"]),
            xgb_proba=float(row["xgb_proba"]),
        )

    # Build similarity edges.
    # Each account links to its top N most similar accounts.
    node_indices = list(G.nodes())

    # X_arr rows correspond to display_idx positions.
    position_lookup = {
        account_idx: position
        for position, account_idx in enumerate(display_idx)
    }

    for account_idx in node_indices:
        i = position_lookup[account_idx]
        similarities = []

        for other_idx in node_indices:
            if account_idx == other_idx:
                continue

            j = position_lookup[other_idx]

            vector_a = X_arr[i]
            vector_b = X_arr[j]

            denominator = (
                np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            )

            if denominator < 1e-9:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(vector_a, vector_b) / denominator
                )

            similarities.append((other_idx, similarity))

        top_links = sorted(
            similarities,
            key=lambda item: item[1],
            reverse=True,
        )[:max(0, top_n_links)]

        for other_idx, similarity in top_links:
            if similarity > 0.85:
                G.add_edge(
                    account_idx,
                    other_idx,
                    weight=round(similarity, 3),
                )

    print(
        f"Graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    return G, list(G.nodes())


def render_pyvis_graph(G: nx.Graph, save_path=None):
    """
    Render the account linkage graph as an interactive PyVis HTML graph.
    Nodes are positioned in rings based on account risk tier and label.
    """

    if save_path is None:
        save_path = config.OUTPUTS_DIR / "linkage_graph.html"

    save_path = str(save_path)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Separate nodes by tier and true label.
    mules = [
        n for n, d in G.nodes(data=True)
        if d.get("true_label", 0) == 1
    ]

    red = [
        n for n, d in G.nodes(data=True)
        if d.get("tier") == "Red"
        and d.get("true_label", 0) != 1
    ]

    amber = [
        n for n, d in G.nodes(data=True)
        if d.get("tier") == "Amber"
        and d.get("true_label", 0) != 1
    ]

    green = [
        n for n, d in G.nodes(data=True)
        if d.get("tier") == "Green"
        and d.get("true_label", 0) != 1
    ]

    positions = {}

    def assign_ring(nodes, radius):
        count = len(nodes)

        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / max(count, 1)
            positions[node] = (
                radius * np.cos(angle),
                radius * np.sin(angle),
            )

    assign_ring(mules, 350)
    assign_ring(red, 550)
    assign_ring(amber, 750)
    assign_ring(green, 950)

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0f1117",
        font_color="white",
        directed=False,
    )

    # Use fixed positions rather than physics simulation.
    net.toggle_physics(False)

    for node, data in G.nodes(data=True):
        tier = data.get("tier", "Green")
        is_mule = data.get("true_label", 0) == 1
        risk_score = float(data.get("risk_score", 0))
        xgb_proba = float(data.get("xgb_proba", 0))

        x, y = positions.get(node, (0, 0))

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
                f"XGB Probability: {xgb_proba:.4f}\n"
                f"Label: {'MULE' if is_mule else 'Normal'}"
            ),
        )

    # Show links involving mules or high-risk accounts.
    for u, v, data in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        u_tier = u_data.get("tier", "Green")
        v_tier = v_data.get("tier", "Green")

        u_mule = u_data.get("true_label", 0) == 1
        v_mule = v_data.get("true_label", 0) == 1

        show_edge = (
            u_mule
            or v_mule
            or (
                u_tier in ["Red", "Amber"]
                and v_tier in ["Red", "Amber"]
            )
        )

        if show_edge:
            weight = float(data.get("weight", 0.5))

            net.add_edge(
                u,
                v,
                width=max(0.5, weight * 2),
                color="#ffffff22",
                title=f"Similarity: {weight:.3f}",
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
    <div style="position:absolute; top:10px; right:10px;
                background:rgba(15,17,23,0.95); padding:14px 18px;
                border-radius:10px; border:1px solid #444;
                font-family:Arial; font-size:13px; color:white; z-index:999;">
        <b style="font-size:15px">Account Risk Network</b><br><br>
        <span style="color:#ff4444; font-size:20px">★</span>
        <span style="color:#ff4444"> Confirmed Mule</span><br><br>
        <span style="color:#ff8888; font-size:16px">●</span>
        <span style="color:#ff8888"> Red — High Risk</span><br><br>
        <span style="color:#ffa500; font-size:16px">◆</span>
        <span style="color:#ffa500"> Amber — Medium Risk</span><br><br>
        <span style="color:#51cf66; font-size:16px">●</span>
        <span style="color:#51cf66"> Green — Low Risk</span><br><br>
        <hr style="border-color:#444; margin:8px 0">
        <i style="color:#888; font-size:11px">
            Rings: Mules → Red → Amber → Green<br>
            Edges represent feature similarity<br>
            Scroll to zoom; drag to pan<br>
            Hover over a node for details
        </i>
    </div>
    <div style="position:absolute; bottom:10px; left:50%;
                transform:translateX(-50%);
                background:rgba(15,17,23,0.95); padding:6px 16px;
                border-radius:20px; border:1px solid #444;
                font-family:Arial; font-size:11px; color:#888; z-index:999;">
        Inner: Mules &nbsp;·&nbsp; Red: High Risk &nbsp;·&nbsp;
        Amber: Medium Risk &nbsp;·&nbsp; Green: Low Risk
    </div>
    """

    net.save_graph(save_path)

    # Inject legend into the generated HTML.
    with open(save_path, "r", encoding="utf-8") as file:
        html = file.read()

    html = html.replace("</body>", f"{legend_html}</body>")

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Linkage graph saved to {save_path}")

    return save_path


def get_connected_components(G: nx.Graph, risk_df: pd.DataFrame):
    """
    Find connected clusters of linked suspicious accounts.
    Returns a summary of each component with two or more nodes.
    """

    clusters = []

    for component in nx.connected_components(G):
        if len(component) < 2:
            continue

        component_risk = risk_df[
            risk_df["account_idx"].isin(component)
        ]

        mule_count = component_risk[
            component_risk["true_label"] == 1
        ].shape[0]

        red_count = component_risk[
            component_risk["risk_tier"] == "Red"
        ].shape[0]

        avg_risk = component_risk["risk_score"].mean()

        clusters.append({
            "size": len(component),
            "mule_count": mule_count,
            "red_count": red_count,
            "avg_risk_score": (
                round(float(avg_risk), 4)
                if pd.notna(avg_risk)
                else 0.0
            ),
            "account_ids": list(component),
        })

    clusters.sort(
        key=lambda cluster: cluster["avg_risk_score"],
        reverse=True,
    )

    return clusters
