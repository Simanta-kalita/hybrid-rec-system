"""
Simulated Neo4j Property Graph Database for Collaborative Filtering (User Cohorts, Orders, Missing Edge Prediction).
Delegates Candidate User Selection to `CandidateStrategyFactory` (Strategy + Factory LLD Pattern) and exposes
equivalent production Cypher query templates alongside in-memory graph traversal.
"""
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from domain.entities import Product, User, UserClusterMembership
from domain.enums import CandidateSelectionStrategy
from infrastructure.interfaces import IGraphDatabase
from services.candidate_strategies import CandidateStrategyFactory


class Neo4jMemoryGraphDB(IGraphDatabase):
    """
    Property Graph Database modeling:
      Nodes:
        - (:User {user_id, name, preferred_categories})
        - (:Product {product_id, title, content_type, price})
        - (:Cluster {cluster_id, aspect})
      Edges:
        - (:User)-[:PURCHASED {rating, timestamp}]->(:Product)
        - (:User)-[:VIEWED {dwell_time}]->(:Product)
        - (:User)-[:BELONGS_TO {aspect, similarity_score}]->(:Cluster)
        - (:User)-[:SIMILAR_TO {aspect, similarity}]->(:User)
    """

    CYPHER_CANDIDATE_USERS_TOP_N = """
    MATCH (u:User {user_id: $user_id})-[:BELONGS_TO]->(c:Cluster)<-[:BELONGS_TO]-(v:User)
    WHERE u <> v
    OPTIONAL MATCH (u)-[s:SIMILAR_TO {aspect: c.aspect}]->(v)
    RETURN v.user_id AS candidate_id, COALESCE(s.similarity, 0.35) AS similarity, c.aspect AS aspect
    ORDER BY similarity DESC
    LIMIT $top_n
    """

    CYPHER_MISSING_EDGE_CANDIDATE_ITEMS = """
    UNWIND $candidate_users AS cand
    MATCH (v:User {user_id: cand.user_id})-[p:PURCHASED]->(item:Product)
    MATCH (u:User {user_id: $target_user_id})
    WHERE NOT (u)-[:PURCHASED]->(item)
    RETURN item.product_id AS product_id, v.user_id AS endorsed_by,
           cand.similarity AS user_similarity, p.rating AS rating, cand.aspect AS aspect
    """

    def __init__(self, random_seed: int = 42) -> None:
        self._strategy_factory = CandidateStrategyFactory(random_seed=random_seed)
        self._user_nodes: Dict[str, Dict[str, Any]] = {}
        self._product_nodes: Dict[str, Dict[str, Any]] = {}
        self._cluster_nodes: Dict[str, Dict[str, Any]] = {}

        self._purchased_edges: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._viewed_edges: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._user_to_clusters: Dict[str, List[UserClusterMembership]] = defaultdict(list)
        self._cluster_to_users: Dict[str, Set[str]] = defaultdict(set)
        self._similar_users: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    def clear_clusters_and_similarities(self) -> None:
        self._cluster_nodes.clear()
        self._user_to_clusters.clear()
        self._cluster_to_users.clear()
        self._similar_users.clear()

    def upsert_user_node(self, user: User) -> None:
        self._user_nodes[user.user_id] = {
            "user_id": user.user_id,
            "name": user.name,
            "preferred_categories": [c.value for c in user.preferred_categories],
        }

    def upsert_product_node(self, product: Product) -> None:
        self._product_nodes[product.product_id] = {
            "product_id": product.product_id,
            "title": product.title,
            "content_type": product.content_type.value,
            "price": product.price,
            "avg_rating": product.avg_rating,
        }

    def upsert_cluster_node(self, cluster_id: str, aspect: str, metadata: Dict[str, Any]) -> None:
        self._cluster_nodes[cluster_id] = {
            "cluster_id": cluster_id,
            "aspect": aspect,
            **metadata,
        }

    def add_purchased_edge(self, user_id: str, product_id: str, rating: float, timestamp: str) -> None:
        self._purchased_edges[user_id][product_id] = {
            "rating": rating,
            "timestamp": timestamp,
        }

    def add_viewed_edge(self, user_id: str, product_id: str, dwell_time: float) -> None:
        existing = self._viewed_edges[user_id].get(product_id, {"dwell_time": 0.0, "views": 0})
        self._viewed_edges[user_id][product_id] = {
            "dwell_time": existing["dwell_time"] + dwell_time,
            "views": existing["views"] + 1,
        }

    def add_belongs_to_edge(self, membership: UserClusterMembership) -> None:
        self._user_to_clusters[membership.user_id] = [
            m for m in self._user_to_clusters[membership.user_id]
            if m.cluster_id != membership.cluster_id
        ]
        self._user_to_clusters[membership.user_id].append(membership)
        self._cluster_to_users[membership.cluster_id].add(membership.user_id)

    def add_similar_user_edge(self, user_a: str, user_b: str, aspect: str, similarity: float) -> None:
        if user_a == user_b:
            return
        self._similar_users[user_a][user_b].append({
            "aspect": aspect,
            "similarity": round(float(similarity), 4),
        })

    def get_user_clusters(self, user_id: str) -> List[UserClusterMembership]:
        return list(self._user_to_clusters.get(user_id, []))

    def get_user_purchased_products(self, user_id: str) -> Set[str]:
        return set(self._purchased_edges.get(user_id, {}).keys())

    def _get_pair_similarity(self, user_a: str, user_b: str, aspect: str) -> float:
        edges = self._similar_users.get(user_a, {}).get(user_b, [])
        if not edges:
            return 0.35
        aspect_matches = [e["similarity"] for e in edges if e["aspect"] == aspect]
        if aspect_matches:
            return max(aspect_matches)
        return max(e["similarity"] for e in edges)

    def query_candidate_users(
        self,
        user_id: str,
        strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.TOP_N_SIMILAR,
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        memberships = self._user_to_clusters.get(user_id, [])
        if not memberships:
            return []
        strategy_impl = self._strategy_factory.get_strategy(strategy)
        return strategy_impl.select_candidates(
            target_user_id=user_id,
            memberships=memberships,
            cluster_to_users=self._cluster_to_users,
            similarity_lookup=self._get_pair_similarity,
            top_n=top_n,
            small_cluster_threshold=small_cluster_threshold,
        )

    def query_candidate_items(
        self,
        target_user_id: str,
        candidate_users: List[Tuple[str, float, str]],
    ) -> Dict[str, List[Tuple[str, float, float, str]]]:
        already_purchased = self.get_user_purchased_products(target_user_id)
        candidate_items: Dict[str, List[Tuple[str, float, float, str]]] = defaultdict(list)

        for cand_user_id, sim_score, aspect in candidate_users:
            cand_purchases = self._purchased_edges.get(cand_user_id, {})
            for product_id, edge_props in cand_purchases.items():
                if product_id in already_purchased or product_id not in self._product_nodes:
                    continue
                rating = float(edge_props.get("rating", 4.5))
                candidate_items[product_id].append((cand_user_id, sim_score, rating, aspect))

        return dict(candidate_items)

    def get_graph_summary(self) -> Dict[str, Any]:
        total_purchased_edges = sum(len(v) for v in self._purchased_edges.values())
        total_viewed_edges = sum(len(v) for v in self._viewed_edges.values())
        total_belongs_to_edges = sum(len(v) for v in self._user_to_clusters.values())
        total_similar_edges = sum(
            sum(len(edges) for edges in targets.values())
            for targets in self._similar_users.values()
        )
        return {
            "nodes": {
                "users": len(self._user_nodes),
                "products": len(self._product_nodes),
                "clusters": len(self._cluster_nodes),
            },
            "edges": {
                "PURCHASED": total_purchased_edges,
                "VIEWED": total_viewed_edges,
                "BELONGS_TO": total_belongs_to_edges,
                "SIMILAR_TO": total_similar_edges,
            },
            "clusters": {
                cid: {
                    "aspect": meta.get("aspect"),
                    "member_count": len(self._cluster_to_users.get(cid, set())),
                    "members": sorted(list(self._cluster_to_users.get(cid, set()))),
                }
                for cid, meta in self._cluster_nodes.items()
            },
        }
