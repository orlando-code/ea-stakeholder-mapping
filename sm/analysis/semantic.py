"""Semantic similarity analysis for cause area clustering.

This module provides tools to:
- Generate embeddings for cause areas using sentence transformers
- Cluster similar cause areas together
- Create 2D coordinates for visualization
- Label clusters based on EA cause categories

The semantic network visualization shows cause areas positioned by similarity,
with node size representing mention frequency.

Note: sentence-transformers and scikit-learn are optional dependencies.
They are imported lazily when SemanticAnalyzer is instantiated to avoid
import errors from pyarrow/datasets compatibility issues.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sm import config

# These will be imported lazily to avoid pyarrow compatibility issues
SentenceTransformer = None
AgglomerativeClustering = None
TSNE = None
cosine_similarity = None


def _ensure_embeddings_available():
    """Lazily import sentence-transformers."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST

            SentenceTransformer = ST
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"sentence-transformers not available: {e}\n"
                "Install with: pip install sentence-transformers\n"
                "If you see pyarrow errors, try: pip install 'datasets>=3.0'"
            ) from e


def _ensure_sklearn_available():
    """Lazily import scikit-learn components."""
    global AgglomerativeClustering, TSNE, cosine_similarity
    if AgglomerativeClustering is None:
        try:
            from sklearn.cluster import AgglomerativeClustering as AC
            from sklearn.manifold import TSNE as T
            from sklearn.metrics.pairwise import cosine_similarity as cs

            AgglomerativeClustering = AC
            TSNE = T
            cosine_similarity = cs
        except ImportError as e:
            raise ImportError(
                f"scikit-learn not available: {e}\nInstall with: pip install scikit-learn"
            ) from e


@dataclass
class CauseAreaCluster:
    """A cluster of semantically related cause areas."""

    cluster_id: int
    name: str  # Display name for the cluster
    members: list[str] = field(default_factory=list)
    total_mentions: int = 0

    def top_members(self, n: int = 5) -> list[str]:
        """Get top N members by mention count."""
        return self.members[:n]


@dataclass
class SemanticAnalysisResult:
    """Results of semantic similarity analysis."""

    cause_areas: list[str]
    mention_counts: dict[str, int]
    embeddings: Optional[np.ndarray] = None
    similarity_matrix: Optional[np.ndarray] = None
    clusters: list[CauseAreaCluster] = field(default_factory=list)
    coordinates_2d: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for visualization."""
        data = []

        for i, area in enumerate(self.cause_areas):
            row = {
                "cause_area": area,
                "mentions": self.mention_counts.get(area, 0),
            }

            # Add 2D coordinates if available
            if self.coordinates_2d is not None and i < len(self.coordinates_2d):
                row["x"] = float(self.coordinates_2d[i, 0])
                row["y"] = float(self.coordinates_2d[i, 1])

            # Find cluster membership
            for cluster in self.clusters:
                if area in cluster.members:
                    row["cluster"] = cluster.cluster_id
                    row["cluster_name"] = cluster.name
                    break
            else:
                row["cluster"] = -1
                row["cluster_name"] = "Other"

            data.append(row)

        return pd.DataFrame(data)

    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of clusters."""
        data = []
        for cluster in self.clusters:
            data.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "name": cluster.name,
                    "n_members": len(cluster.members),
                    "total_mentions": cluster.total_mentions,
                    "top_members": ", ".join(cluster.top_members(3)),
                }
            )
        return pd.DataFrame(data)


class SemanticAnalyzer:
    """Analyze semantic similarity between cause areas.

    Uses sentence transformers for embeddings and hierarchical clustering
    to group similar cause areas together.

    Example:
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze(cause_areas, mention_counts)

        # Get DataFrame for visualization
        df = result.to_dataframe()

        # Get similar cause areas
        similar = analyzer.find_similar("AI alignment", cause_areas)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_clusters: Optional[int] = None,
    ):
        """Initialize semantic analyzer.

        Args:
            embedding_model: Sentence transformer model name
            n_clusters: Number of clusters (auto-determined if None)

        Raises:
            ImportError: If required packages not installed
        """
        # Lazy import to avoid pyarrow/datasets compatibility issues at module load
        _ensure_embeddings_available()
        _ensure_sklearn_available()

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. Install with:\n  pip install sentence-transformers"
            )
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn required. Install with:\n  pip install scikit-learn")

        self.model = SentenceTransformer(embedding_model)
        self.n_clusters = n_clusters

    def analyze(
        self,
        cause_areas: list[str],
        mention_counts: Optional[dict[str, int]] = None,
        use_predefined_categories: bool = False,
        similarity_threshold: float = 0.3,
        min_category_size: int = 1,
    ) -> SemanticAnalysisResult:
        """Analyze semantic similarity of cause areas.

        Args:
            cause_areas: List of cause area strings
            mention_counts: Dict mapping cause area to count (computed if None)
            use_predefined_categories: If True, assign cause areas to predefined EA
                categories (from config.EA_CAUSE_CATEGORIES) based on semantic similarity
                instead of using unsupervised clustering.
            similarity_threshold: When using predefined categories, minimum similarity
                score to assign an item to a category (items below threshold go to "Other").
            min_category_size: When using predefined categories, minimum number of members
                per category. Categories with fewer members will have their items
                reassigned to the next-closest category that meets the size threshold.
        Returns:
            SemanticAnalysisResult with embeddings, clusters, and 2D coordinates
        """
        if not cause_areas:
            return SemanticAnalysisResult(cause_areas=[], mention_counts={})

        # Deduplicate while preserving order
        unique_areas = list(dict.fromkeys(cause_areas))

        # Count mentions if not provided
        if mention_counts is None:
            mention_counts = dict(Counter(cause_areas))

        # Sort by mentions (descending)
        unique_areas = sorted(
            unique_areas,
            key=lambda x: mention_counts.get(x, 0),
            reverse=True,
        )

        # Generate embeddings
        embeddings = self.model.encode(unique_areas, show_progress_bar=False)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Cluster cause areas
        if use_predefined_categories:
            clusters = self._assign_to_predefined_categories(
                unique_areas,
                embeddings,
                mention_counts,
                similarity_threshold=similarity_threshold,
                min_category_size=min_category_size,
            )
        else:
            n_clusters = self.n_clusters or self._estimate_n_clusters(len(unique_areas))
            clusters = self._cluster_cause_areas(
                unique_areas, embeddings, mention_counts, n_clusters
            )

        # Generate 2D coordinates
        coords_2d = self._reduce_to_2d(embeddings)

        return SemanticAnalysisResult(
            cause_areas=unique_areas,
            mention_counts=mention_counts,
            embeddings=embeddings,
            similarity_matrix=similarity_matrix,
            clusters=clusters,
            coordinates_2d=coords_2d,
        )

    def _estimate_n_clusters(self, n_items: int) -> int:
        """Estimate appropriate number of clusters."""
        if n_items < 5:
            return max(1, n_items)
        elif n_items < 15:
            return min(5, n_items // 2)
        elif n_items < 50:
            return min(8, n_items // 4)
        else:
            return min(12, n_items // 6)

    def _cluster_cause_areas(
        self,
        areas: list[str],
        embeddings: np.ndarray,
        mention_counts: dict[str, int],
        n_clusters: int,
    ) -> list[CauseAreaCluster]:
        """Cluster cause areas using hierarchical clustering."""
        if len(areas) <= n_clusters:
            # Each area is its own cluster
            return [
                CauseAreaCluster(
                    cluster_id=i,
                    name=self._name_single_cluster(area),
                    members=[area],
                    total_mentions=mention_counts.get(area, 0),
                )
                for i, area in enumerate(areas)
            ]

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Group areas by cluster
        cluster_members = {i: [] for i in range(n_clusters)}
        for area, label in zip(areas, labels):
            cluster_members[label].append(area)

        # Create cluster objects with unique names
        clusters = []
        used_names: set[str] = set()

        for cluster_id, members in cluster_members.items():
            # Sort members by mention count
            members_sorted = sorted(
                members,
                key=lambda x: mention_counts.get(x, 0),
                reverse=True,
            )

            # Name the cluster (ensuring uniqueness)
            cluster_name = self._name_cluster(members_sorted, used_names)
            used_names.add(cluster_name)

            # Calculate total mentions
            total_mentions = sum(mention_counts.get(m, 0) for m in members)

            clusters.append(
                CauseAreaCluster(
                    cluster_id=cluster_id,
                    name=cluster_name,
                    members=members_sorted,
                    total_mentions=total_mentions,
                )
            )

        # Sort clusters by total mentions
        clusters.sort(key=lambda c: c.total_mentions, reverse=True)

        # Reassign cluster IDs after sorting
        for i, cluster in enumerate(clusters):
            cluster.cluster_id = i

        return clusters

    def _name_cluster(self, members: list[str], used_names: Optional[set[str]] = None) -> str:
        """Generate a unique name for a cluster based on its members.

        Args:
            members: List of cause area strings in the cluster
            used_names: Set of names already assigned to other clusters
        """
        if not members:
            return "Other"

        used_names = used_names or set()

        # Score each EA category by how well it matches the cluster
        category_scores: list[tuple[str, int]] = []
        for category, info in config.EA_CAUSE_CATEGORIES.items():
            keywords = info["keywords"]
            label = info["label"]

            # Count how many members match this category
            matches = sum(
                1
                for m in members
                for kw in keywords
                if kw.lower() in m.lower() or m.lower() in kw.lower()
            )

            # Require significant matches
            if matches >= len(members) * 0.3 or matches >= 2:
                category_scores.append((label, matches))

        # Sort by score (highest first)
        category_scores.sort(key=lambda x: x[1], reverse=True)

        # Pick the best unused category label
        for label, score in category_scores:
            if label not in used_names:
                return label

        # All matching categories are used - use top member name
        fallback = members[0].title()

        # Ensure fallback is also unique
        if fallback in used_names:
            for i, member in enumerate(members[1:], 2):
                alt_name = member.title()
                if alt_name not in used_names:
                    return alt_name
            # Last resort: add numeric suffix
            counter = 2
            base_name = fallback
            while fallback in used_names:
                fallback = f"{base_name} ({counter})"
                counter += 1

        return fallback

    def _name_single_cluster(self, area: str) -> str:
        """Name a single-member cluster."""
        # Check against EA categories
        for category, info in config.EA_CAUSE_CATEGORIES.items():
            for keyword in info["keywords"]:
                if keyword.lower() in area.lower() or area.lower() in keyword.lower():
                    return info["label"]
        return area.title()

    def _assign_to_predefined_categories(
        self,
        areas: list[str],
        embeddings: np.ndarray,
        mention_counts: dict[str, int],
        similarity_threshold: float = 0.3,
        min_category_size: int = 1,
    ) -> list[CauseAreaCluster]:
        """Assign cause areas to predefined EA categories based on embedding similarity.

        Args:
            areas: List of cause area strings
            embeddings: Embeddings for each cause area
            mention_counts: Dict mapping cause area to count
            similarity_threshold: Minimum similarity to assign to a category (below this -> "Other")
            min_category_size: Minimum members per category. Categories with fewer members
                will have their items reassigned to the next-closest category.
        Returns:
            List of CauseAreaCluster objects
        """
        # Get category labels from config
        category_labels = [info["label"] for info in config.EA_CAUSE_CATEGORIES.values()]

        # Create category centroids by embedding the category labels
        category_embeddings = self.model.encode(category_labels, show_progress_bar=False)

        # For each area, compute similarity to all categories
        similarities = cosine_similarity(embeddings, category_embeddings)

        # Track assignments: area_index -> (category_name, similarity)
        assignments: dict[int, tuple[str, float]] = {}

        for i, area in enumerate(areas):
            # Sort categories by similarity (descending)
            sorted_indices = np.argsort(similarities[i])[::-1]
            best_idx = sorted_indices[0]
            best_sim = similarities[i, best_idx]

            if best_sim >= similarity_threshold:
                assignments[i] = (category_labels[best_idx], best_sim)
            else:
                assignments[i] = ("Other", best_sim)

        # Group areas by category
        category_members: dict[str, list[tuple[int, str, float]]] = {
            label: [] for label in category_labels
        }
        category_members["Other"] = []

        for i, (cat_name, sim) in assignments.items():
            category_members[cat_name].append((i, areas[i], sim))

        # Reassign items from small categories to next-closest category
        if min_category_size > 1:
            categories_to_check = list(category_members.keys())
            for cat_name in categories_to_check:
                members = category_members[cat_name]
                if 0 < len(members) < min_category_size and cat_name != "Other":
                    # Reassign each member to next-closest category
                    for area_idx, area, _ in members:
                        sorted_indices = np.argsort(similarities[area_idx])[::-1]

                        # Find next-closest category that has enough members or is "Other"
                        for cat_idx in sorted_indices:
                            alt_cat = category_labels[cat_idx]
                            if alt_cat == cat_name:
                                continue
                            # Check if this category has enough members (or will after reassignment)
                            if (
                                len(category_members[alt_cat]) >= min_category_size
                                or alt_cat == "Other"
                            ):
                                category_members[alt_cat].append(
                                    (area_idx, area, similarities[area_idx, cat_idx])
                                )
                                break
                        else:
                            # Fall back to "Other" if no suitable category found
                            category_members["Other"].append((area_idx, area, 0.0))

                    # Clear the small category
                    category_members[cat_name] = []

        # Create cluster objects (skip empty clusters)
        clusters = []
        for name, members in category_members.items():
            if not members:
                continue

            # Sort members by mention count
            members_sorted = sorted(
                [area for _, area, _ in members],
                key=lambda x: mention_counts.get(x, 0),
                reverse=True,
            )

            total_mentions = sum(mention_counts.get(m, 0) for m in members_sorted)

            clusters.append(
                CauseAreaCluster(
                    cluster_id=0,  # Will be reassigned
                    name=name,
                    members=members_sorted,
                    total_mentions=total_mentions,
                )
            )

        # Sort clusters by total mentions
        clusters.sort(key=lambda c: c.total_mentions, reverse=True)

        # Reassign cluster IDs after sorting
        for i, cluster in enumerate(clusters):
            cluster.cluster_id = i

        return clusters

    def _reduce_to_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to 2D using t-SNE."""
        if len(embeddings) < 2:
            return np.zeros((len(embeddings), 2))

        # Adjust perplexity based on sample size
        perplexity = min(30, max(5, len(embeddings) - 1))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,  # renamed from n_iter in sklearn 1.5+
        )

        return tsne.fit_transform(embeddings)

    def find_similar(
        self,
        query: str,
        cause_areas: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find cause areas most similar to a query.

        Args:
            query: Query string
            cause_areas: List of cause areas to search
            top_k: Number of results to return
        Returns:
            List of (cause_area, similarity_score) tuples
        """
        if not cause_areas:
            return []

        # Encode query and cause areas
        query_embedding = self.model.encode([query])[0]
        area_embeddings = self.model.encode(cause_areas)

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], area_embeddings)[0]

        # Get top-k results
        indices = np.argsort(similarities)[::-1][:top_k]

        return [(cause_areas[i], float(similarities[i])) for i in indices]


# =============================================================================
# Convenience Functions
# =============================================================================


def _token_similarity(term1: str, term2: str) -> float:
    """Calculate token-based similarity between two terms.
    
    Returns a score between 0 and 1 based on shared tokens.
    """
    tokens1 = set(term1.lower().split())
    tokens2 = set(term2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Jaccard similarity on tokens
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def _is_variant_of(term: str, canonical: str, min_similarity: float = 0.5) -> bool:
    """Check if term is a variant of canonical term.
    
    Uses multiple heuristics:
    1. Substring matching (one contains the other)
    2. Token-based similarity (Jaccard on word tokens)
    3. Shared core phrases (2+ consecutive words)
    4. Levenshtein-like distance for very similar strings
    
    Args:
        term: Term to check
        canonical: Canonical term to compare against
        min_similarity: Minimum token similarity threshold
    Returns:
        True if term appears to be a variant
    """
    term_lower = term.lower().strip()
    canonical_lower = canonical.lower().strip()
    
    # Skip if identical (after normalization)
    if term_lower == canonical_lower:
        return True
    
    # Exact substring match (one contains the other, but not too short)
    # This catches: "ai alignment" and "ai alignment research"
    if len(term_lower) >= 5 and len(canonical_lower) >= 5:
        if term_lower in canonical_lower or canonical_lower in term_lower:
            return True
    
    # Token-based similarity
    similarity = _token_similarity(term, canonical)
    if similarity >= min_similarity:
        return True
    
    # Check if they share a significant core phrase (2+ consecutive words)
    term_tokens = term_lower.split()
    canonical_tokens = canonical_lower.split()
    
    # Find common 2-word phrases
    if len(term_tokens) >= 2 and len(canonical_tokens) >= 2:
        term_bigrams = set(tuple(term_tokens[i:i+2]) for i in range(len(term_tokens) - 1))
        canonical_bigrams = set(tuple(canonical_tokens[i:i+2]) for i in range(len(canonical_tokens) - 1))
        
        if term_bigrams & canonical_bigrams:  # Any overlap
            return True
    
    # Check for single-word overlap on important terms (3+ chars)
    # This helps catch: "ai safety" and "technical ai safety"
    important_terms = {t for t in term_tokens if len(t) >= 3}
    important_canonical = {t for t in canonical_tokens if len(t) >= 3}
    
    if important_terms and important_canonical:
        overlap = important_terms & important_canonical
        # If they share at least 2 important words, likely variants
        if len(overlap) >= 2:
            return True
        # Or if one important word appears and similarity is decent
        if len(overlap) >= 1 and similarity >= 0.3:
            return True
    
    return False


def normalize_cause_areas_fuzzy(
    cause_areas: list[str],
    mention_counts: dict[str, int],
    min_similarity: float = 0.5,
    min_mentions_for_canonical: int = 2,
) -> tuple[list[str], dict[str, int]]:
    """Normalize cause areas using fuzzy matching to group variants.
    
    Groups similar terms together by finding canonical forms. Terms are grouped
    if they share significant token overlap or one is a substring of the other.
    Processes terms in frequency order so common terms become canonicals.
    
    Args:
        cause_areas: List of cause area strings
        mention_counts: Dict mapping cause area to mention count
        min_similarity: Minimum token similarity to group terms (0.0-1.0)
        min_mentions_for_canonical: Minimum mentions for a term to be considered canonical
    Returns:
        Tuple of (normalized cause areas list, normalized mention counts dict)
    """
    if not cause_areas:
        return [], {}
    
    # Sort by frequency (most common first) - these become canonical candidates
    sorted_areas = sorted(
        cause_areas,
        key=lambda x: mention_counts.get(x, 0),
        reverse=True,
    )
    
    # Map: original_term -> normalized_name (canonical form)
    term_to_normalized: dict[str, str] = {}
    # Set of canonical forms we've established
    canonicals: set[str] = set()
    
    for term in sorted_areas:
        term_count = mention_counts.get(term, 0)
        normalized = None
        
        # Check if this term is a variant of an existing canonical
        # (only check against canonicals we've already established)
        for canonical in canonicals:
            if _is_variant_of(term, canonical, min_similarity):
                normalized = canonical
                break
        
        # If not a variant, this term becomes a new canonical
        if normalized is None:
            # Use this term as canonical if it has enough mentions
            # OR if it's the first term (to ensure we always have at least one)
            if term_count >= min_mentions_for_canonical or not canonicals:
                normalized = term
                canonicals.add(normalized)
            else:
                # Low-frequency term with no similar canonical - keep as-is (its own canonical)
                normalized = term
                canonicals.add(normalized)
        
        term_to_normalized[term] = normalized
    
    # Build normalized counts by summing counts of all variants
    normalized_counts: dict[str, int] = {}
    for original, count in mention_counts.items():
        normalized = term_to_normalized.get(original, original)
        normalized_counts[normalized] = normalized_counts.get(normalized, 0) + count
    
    # Return sorted list
    normalized_areas = sorted(
        normalized_counts.keys(),
        key=lambda x: normalized_counts.get(x, 0),
        reverse=True,
    )
    
    return normalized_areas, normalized_counts


def normalize_cause_area(area: str) -> str:
    """Normalize a single cause area (legacy function - use normalize_cause_areas_fuzzy for batch).
    
    This is a simple pass-through that just capitalizes. For proper fuzzy normalization,
    use normalize_cause_areas_fuzzy() which processes all terms together.
    """
    return area.strip().title()


def aggregate_cause_areas(
    df: pd.DataFrame,
    cause_area_columns: Optional[list[str]] = None,
    min_mentions: int = 1,
    normalize: bool = False,
) -> tuple[list[str], dict[str, int]]:
    """Aggregate cause areas from DataFrame columns.

    Args:
        df: DataFrame with cause area columns (containing lists)
        cause_area_columns: Columns to aggregate from
        min_mentions: Minimum mentions to include
        normalize: If True, normalize similar terms together (e.g., group AI variants)
    Returns:
        Tuple of (list of cause areas, dict of mention counts)
    """
    if cause_area_columns is None:
        # Auto-detect cause area columns
        cause_area_columns = [
            c
            for c in df.columns
            if "cause" in c.lower() or "expertise" in c.lower() or "interest" in c.lower()
        ]

    all_areas = []

    for col in cause_area_columns:
        if col not in df.columns:
            continue

        for items in df[col]:
            if isinstance(items, list):
                all_areas.extend(items)
            elif isinstance(items, str) and items:
                all_areas.append(items)

    # Normalize
    all_areas = [a.lower().strip() for a in all_areas if a]
    
    # Apply normalization if requested
    if normalize:
        all_areas = [normalize_cause_area(a) for a in all_areas]

    # Count mentions
    mention_counts = dict(Counter(all_areas))

    # Filter by minimum mentions
    unique_areas = [area for area, count in mention_counts.items() if count >= min_mentions]

    # Sort by count
    unique_areas = sorted(
        unique_areas,
        key=lambda x: mention_counts.get(x, 0),
        reverse=True,
    )

    return unique_areas, mention_counts
